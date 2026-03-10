import triton_python_backend_utils as pb_utils
import numpy as np
import json
import gc
import torch
import threading
import time
import re


class TritonPythonModel:
    def initialize(self, args):
        """Initialize the streaming diarization model."""
        self.model_config = json.loads(args["model_config"])
        self.device_id = int(args.get("model_instance_device_id", "0"))
        self.device = f"cuda:{self.device_id}"

        print(f"[Streaming Diar] Loading model on {self.device}...")

        from nemo.collections.asr.models import SortformerEncLabelModel

        self.diar_model = SortformerEncLabelModel.from_pretrained(
            "nvidia/diar_streaming_sortformer_4spk-v2.1"
        )
        self.diar_model = self.diar_model.to(self.device)
        self.diar_model.eval()

        # Configure for low-latency streaming
        self._configure_streaming()

        self.target_sr = 16000

        # Per-session state: accumulated audio + last diarization result
        self._session_audio = {}       # session_id -> np.array of accumulated audio
        self._session_results = {}     # session_id -> last diarization segments
        self._session_lock = threading.Lock()
        self._session_timestamps = {}

        # Minimum audio duration (seconds) before running diarization
        # Sortformer needs enough context for meaningful speaker detection
        self.min_diar_duration = 3.0
        self.diar_interval_seconds = 5.0
        # Keep diarization cost bounded for long sessions.
        # We only need near-real-time speaker turns for live updates; full-call
        # diarization still runs in the post-call batch pipeline.
        self.max_diar_window_seconds = 90.0

        # Warmup
        self._warmup()

        print("[Streaming Diar] Initialization complete")

    def _configure_streaming(self):
        """Configure Sortformer for low-latency streaming."""
        try:
            sm = self.diar_model.sortformer_modules
            # Low-latency preset from HuggingFace model card
            sm.chunk_len = 6
            sm.chunk_right_context = 7
            sm.fifo_len = 188
            sm.spkcache_update_period = 144
            sm.spkcache_len = 188
            # Validate if method exists
            if hasattr(sm, "_check_streaming_parameters"):
                sm._check_streaming_parameters()
            print("[Streaming Diar] Configured low-latency streaming (1.04s latency)")
        except Exception as e:
            print(f"[Streaming Diar] Streaming config warning: {e}")

    def _warmup(self):
        """Warm up the model with dummy audio."""
        print("[Streaming Diar] Warming up...")
        try:
            import soundfile as sf
            import tempfile
            import os

            # 3 seconds of dummy audio
            dummy_audio = np.random.randn(self.target_sr * 3).astype(np.float32) * 0.01
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, dummy_audio, self.target_sr)
                temp_path = f.name

            try:
                with torch.no_grad():
                    self.diar_model.diarize(audio=[temp_path], batch_size=1)
                print("[Streaming Diar] Warmup complete")
            finally:
                os.unlink(temp_path)
        except Exception as e:
            print(f"[Streaming Diar] Warmup failed (non-critical): {e}")
            import traceback
            traceback.print_exc()

    def _cleanup_stale_sessions(self, max_age_seconds=3600):
        """Remove session state older than max_age_seconds."""
        now = time.time()
        stale = [
            sid
            for sid, ts in self._session_timestamps.items()
            if now - ts > max_age_seconds
        ]
        for sid in stale:
            self._session_audio.pop(sid, None)
            self._session_results.pop(sid, None)
            self._session_timestamps.pop(sid, None)
        if stale:
            print(f"[Streaming Diar] Cleaned up {len(stale)} stale sessions")

    def _run_diarization(self, audio_data, time_offset=0.0):
        """Run diarization on audio data and return normalized speaker segments."""
        import soundfile as sf
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, self.target_sr)
            temp_path = f.name

        try:
            with torch.no_grad():
                predicted_segments = self.diar_model.diarize(
                    audio=[temp_path],
                    batch_size=1,
                )
        finally:
            os.unlink(temp_path)

        segments = []
        if predicted_segments and len(predicted_segments) > 0:
            raw_segments = predicted_segments[0]
            for seg in raw_segments:
                parsed = self._parse_segment(seg)
                if not parsed:
                    continue
                start_t, end_t, spk = parsed
                segments.append(
                    {
                        "speaker": spk,
                        "start": round(start_t + time_offset, 3),
                        "end": round(end_t + time_offset, 3),
                    }
                )

            if not segments:
                print(
                    "[Streaming Diar] Warning: diarization ran but produced no parseable segments; "
                    f"raw type={type(raw_segments)}"
                )

        segments = self._normalize_speaker_ids(segments)
        num_speakers = len({seg["speaker"] for seg in segments}) if segments else 0
        return segments, num_speakers

    def _parse_segment(self, seg):
        """
        Parse one diarization segment in a tolerant way.
        Accepts tuple/list, dict, and object-style segments.
        """
        try:
            if isinstance(seg, str):
                parsed = self._parse_segment_from_text(seg)
                if not parsed:
                    return None
                start_t, end_t, spk = parsed
            elif isinstance(seg, dict):
                start_t = float(seg.get("start", seg.get("start_time", 0.0)))
                end_val = seg.get("end", seg.get("end_time"))
                if end_val is None and "duration" in seg:
                    end_t = start_t + float(seg["duration"])
                else:
                    end_t = float(end_val or 0.0)
                spk = self._parse_speaker_id(
                    seg.get(
                        "speaker",
                        seg.get("speaker_id", seg.get("spk_id", seg.get("label"))),
                    )
                )
            elif hasattr(seg, "start") and hasattr(seg, "end"):
                start_t = float(seg.start)
                end_t = float(seg.end)
                spk = self._parse_speaker_id(
                    getattr(seg, "speaker", getattr(seg, "label", 0))
                )
            elif hasattr(seg, "__iter__") and not isinstance(seg, (str, bytes)):
                # Sortformer commonly yields tuples like:
                # (Segment(start, end), track_id, "speaker_0")
                items = list(seg)
                if len(items) >= 1 and hasattr(items[0], "start") and hasattr(items[0], "end"):
                    start_t = float(items[0].start)
                    end_t = float(items[0].end)
                    label = items[2] if len(items) >= 3 else (items[1] if len(items) >= 2 else 0)
                    spk = self._parse_speaker_id(label)
                elif len(items) >= 3:
                    start_t = float(items[0])
                    end_t = float(items[1])
                    spk = self._parse_speaker_id(items[2])
                else:
                    parsed = self._parse_segment_from_text(str(seg))
                    if not parsed:
                        return None
                    start_t, end_t, spk = parsed
            else:
                parsed = self._parse_segment_from_text(str(seg))
                if not parsed:
                    return None
                start_t, end_t, spk = parsed

            if spk is None:
                return None
            if end_t <= start_t:
                return None
            return start_t, end_t, spk
        except Exception:
            return None

    def _parse_segment_from_text(self, text):
        """
        Parse textual segment representations.
        Common formats include:
          "0.52 1.80 speaker_1"
          "[0.52 - 1.80] speaker_1"
        """
        if not text:
            return None

        text = text.strip()
        parts = text.split()
        # RTTM line format:
        # SPEAKER <file-id> <channel-id> <start> <duration> <...> <speaker_id> <...>
        if len(parts) >= 8 and parts[0].upper() == "SPEAKER":
            try:
                start_t = float(parts[3])
                duration = float(parts[4])
                end_t = start_t + duration
                spk = self._parse_speaker_id(parts[7])
                if spk is not None and end_t > start_t:
                    return start_t, end_t, spk
            except Exception:
                pass

        values = re.findall(r"-?\d+(?:\.\d+)?", text)
        if len(values) < 2:
            return None

        start_t = float(values[0])
        end_t = float(values[1])

        # Prefer explicit "speaker_*" labels when present.
        label_match = re.search(r"(speaker[_\s-]*\d+|spk[_\s-]*\d+)", text, flags=re.IGNORECASE)
        if label_match:
            spk = self._parse_speaker_id(label_match.group(1))
        elif len(values) >= 3:
            # If label is implicit in the text, last numeric token is usually speaker id.
            spk = int(float(values[-1]))
        else:
            spk = None

        return start_t, end_t, spk

    def _parse_speaker_id(self, raw_speaker):
        """Convert speaker labels like 0, '1', 'speaker_2' to int."""
        if raw_speaker is None:
            return None

        if isinstance(raw_speaker, (int, np.integer)):
            return max(0, int(raw_speaker))
        if isinstance(raw_speaker, (float, np.floating)):
            return max(0, int(raw_speaker))

        if hasattr(raw_speaker, "item"):
            try:
                return max(0, int(raw_speaker.item()))
            except Exception:
                pass

        text = str(raw_speaker).strip()
        if not text:
            return None

        speaker_match = re.search(r"(?:speaker[_\s-]*|spk[_\s-]*)(\d+)", text, flags=re.IGNORECASE)
        if speaker_match:
            return int(speaker_match.group(1))

        if text.isdigit():
            return max(0, int(text))

        float_like = re.fullmatch(r"-?\d+(?:\.\d+)?", text)
        if float_like:
            try:
                return max(0, int(float(text)))
            except Exception:
                return None

        numbers = re.findall(r"\d+", text)
        if numbers:
            return max(0, int(numbers[-1]))

        return None

    def _normalize_speaker_ids(self, segments):
        """
        Normalize speaker IDs to a contiguous 0-based range.
        This keeps downstream formatting stable when model labels are 1-based
        or sparse/non-contiguous.
        """
        if not segments:
            return []

        unique_raw = sorted({int(seg.get("speaker", 0)) for seg in segments})
        speaker_map = {raw_id: idx for idx, raw_id in enumerate(unique_raw)}

        normalized = []
        for seg in segments:
            raw_id = int(seg.get("speaker", 0))
            normalized.append(
                {
                    "speaker": speaker_map.get(raw_id, 0),
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                }
            )

        return normalized

    def execute(self, requests):
        """Execute streaming diarization on audio chunks."""
        if not requests:
            return []

        responses = []

        for request in requests:
            try:
                # Extract inputs
                audio_tensor = pb_utils.get_input_tensor_by_name(request, "audio_data")
                sr_tensor = pb_utils.get_input_tensor_by_name(request, "sample_rate")
                sid_tensor = pb_utils.get_input_tensor_by_name(request, "session_id")
                final_tensor = pb_utils.get_input_tensor_by_name(request, "is_final")

                audio_data = audio_tensor.as_numpy().flatten().astype(np.float32)
                sample_rate = int(sr_tensor.as_numpy().flat[0])
                session_id = sid_tensor.as_numpy().flat[0]
                if isinstance(session_id, bytes):
                    session_id = session_id.decode("utf-8")
                is_final = bool(final_tensor.as_numpy().flat[0])

                # Resample to 16kHz if needed
                if sample_rate != self.target_sr:
                    from scipy.signal import resample
                    num_samples = int(len(audio_data) * self.target_sr / sample_rate)
                    audio_data = resample(audio_data, num_samples).astype(np.float32)

                # Accumulate audio for this session
                with self._session_lock:
                    if session_id not in self._session_audio:
                        self._session_audio[session_id] = np.array([], dtype=np.float32)
                    self._session_audio[session_id] = np.concatenate(
                        [self._session_audio[session_id], audio_data]
                    )
                    self._session_timestamps[session_id] = time.time()
                    accumulated_audio = self._session_audio[session_id].copy()

                audio_duration = len(accumulated_audio) / self.target_sr

                # Decide whether to run diarization
                # - Always run on final chunk
                # - Run if we have enough audio (>= min_diar_duration)
                # - Only re-run every ~5 seconds of audio to avoid re-processing overhead
                should_run = False
                if is_final:
                    should_run = True
                elif audio_duration >= self.min_diar_duration:
                    # Check if we've accumulated ~N seconds more since last run
                    last_result = self._session_results.get(session_id)
                    if last_result is None:
                        should_run = True
                    else:
                        last_duration = last_result.get("audio_duration", 0)
                        if audio_duration - last_duration >= self.diar_interval_seconds:
                            should_run = True

                if should_run:
                    # Run diarization on a bounded trailing window so work stays stable
                    # for long-running calls.
                    window_offset = 0.0
                    audio_for_diar = accumulated_audio
                    if audio_duration > self.max_diar_window_seconds:
                        window_offset = audio_duration - self.max_diar_window_seconds
                        start_idx = int(window_offset * self.target_sr)
                        audio_for_diar = accumulated_audio[start_idx:]

                    segments, num_speakers = self._run_diarization(
                        audio_for_diar,
                        time_offset=window_offset,
                    )
                else:
                    # Return cached result
                    last_result = self._session_results.get(session_id, {})
                    segments = last_result.get("segments", [])
                    num_speakers = last_result.get("num_speakers", 0)

                result = {
                    "segments": segments,
                    "num_speakers": num_speakers,
                    "session_id": session_id,
                    "audio_duration": audio_duration,
                    "is_final": is_final,
                    "diar_ran": should_run,
                    "diar_window_seconds": min(audio_duration, self.max_diar_window_seconds),
                }

                # Cache result
                if should_run:
                    with self._session_lock:
                        self._session_results[session_id] = result

                # Clean up session on final chunk
                if is_final:
                    with self._session_lock:
                        self._session_audio.pop(session_id, None)
                        self._session_results.pop(session_id, None)
                        self._session_timestamps.pop(session_id, None)
                    print(f"[Streaming Diar] Session {session_id} finalized ({audio_duration:.1f}s, {num_speakers} speakers)")

                # Periodic stale cleanup
                if len(self._session_timestamps) > 100:
                    self._cleanup_stale_sessions()

                output_json = json.dumps(result)
                output_array = np.array([[output_json]], dtype=object)
                output_tensor = pb_utils.Tensor("diarization_output", output_array)
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[output_tensor])
                )

            except Exception as e:
                print(f"[Streaming Diar] Error: {e}")
                import traceback
                traceback.print_exc()
                error_data = {
                    "segments": [],
                    "num_speakers": 0,
                    "error": str(e),
                }
                error_json = json.dumps(error_data)
                error_array = np.array([[error_json]], dtype=object)
                error_tensor = pb_utils.Tensor("diarization_output", error_array)
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[error_tensor])
                )

        return responses

    def finalize(self):
        """Clean up resources."""
        print("[Streaming Diar] Finalizing...")
        try:
            del self.diar_model
            self._session_audio.clear()
            self._session_results.clear()
            torch.cuda.empty_cache()
            gc.collect()
            print("[Streaming Diar] Finalized successfully")
        except Exception as e:
            print(f"[Streaming Diar] Error during finalization: {e}")
