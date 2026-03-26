import triton_python_backend_utils as pb_utils
import numpy as np
import nemo.collections.asr as nemo_asr
import json
import gc
import torch
import threading
import time
from omegaconf import open_dict


class TritonPythonModel:
    def initialize(self, args):
        """Initialize the streaming ASR model with cache-aware inference."""
        self.model_config = json.loads(args["model_config"])
        self.device_id = int(args.get("model_instance_device_id", "0"))
        self.device = f"cuda:{self.device_id}"

        print(f"[Streaming ASR] Loading model on {self.device}...")

        # Load the streaming ASR model
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/nemotron-speech-streaming-en-0.6b"
        )
        self.asr_model = self.asr_model.to(self.device)
        self.asr_model.eval()

        # H100 optimization: bfloat16
        if torch.cuda.is_bf16_supported():
            self.asr_model = self.asr_model.to(dtype=torch.bfloat16)
            self.amp_dtype = torch.bfloat16
            print("[Streaming ASR] Using bfloat16 precision")
        else:
            self.amp_dtype = torch.float16
            print("[Streaming ASR] Using float16 precision")

        # Enable Flash Attention
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            print("[Streaming ASR] Enabled Flash Attention")

        self.target_sr = 16000

        # Configure for cache-aware streaming
        self._configure_streaming()

        # Determine pre-encode cache size from model config
        try:
            self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
        except Exception:
            self.pre_encode_cache_size = 2  # safe default
        self.num_features = self.asr_model.cfg.preprocessor.features

        # Per-session cache storage
        self._sessions = {}  # session_id -> dict of cache tensors
        self._session_lock = threading.Lock()
        self._session_timestamps = {}
        self._last_cleanup_at = 0.0
        self._cleanup_interval_seconds = 60.0

        # Warmup
        self._warmup()

        print(f"[Streaming ASR] Initialization complete (pre_encode_cache_size={self.pre_encode_cache_size}, features={self.num_features})")

    def _configure_streaming(self):
        """Configure the model for cache-aware streaming inference."""
        try:
            # Configure decoding strategy for streaming
            decoding_cfg = self.asr_model.cfg.decoding
            with open_dict(decoding_cfg):
                decoding_cfg.strategy = "greedy"
                decoding_cfg.preserve_alignments = False
                if hasattr(decoding_cfg, "greedy"):
                    with open_dict(decoding_cfg.greedy):
                        decoding_cfg.greedy.loop_labels = True
                        decoding_cfg.greedy.use_cuda_graph_decoder = False
                        # Allow longer token emission per streaming step to reduce
                        # truncation on dense/long utterances.
                        decoding_cfg.greedy.max_symbols = 50
                if hasattr(decoding_cfg, "fused_batch_size"):
                    decoding_cfg.fused_batch_size = -1
            self.asr_model.change_decoding_strategy(decoding_cfg)

            # Configure streaming chunk size on encoder
            # att_context_size=[70, 13] = 1.12s latency, best accuracy
            if hasattr(self.asr_model, "encoder") and hasattr(
                self.asr_model.encoder, "setup_streaming_params"
            ):
                self.asr_model.encoder.setup_streaming_params(
                    att_context_size=[70, 13]
                )
                print("[Streaming ASR] Configured streaming with att_context_size=[70, 13]")
            else:
                print("[Streaming ASR] Warning: Could not configure streaming params on encoder")

            print("[Streaming ASR] Streaming configuration applied")
        except Exception as e:
            print(f"[Streaming ASR] Streaming config failed: {e}")

    def _init_session_cache(self, session_id):
        """Initialize all cache tensors for a new streaming session."""
        # Encoder caches
        cache_last_channel, cache_last_time, cache_last_channel_len = \
            self.asr_model.encoder.get_initial_cache_state(batch_size=1)

        # Move caches to device
        cache_last_channel = cache_last_channel.to(self.device)
        cache_last_time = cache_last_time.to(self.device)
        cache_last_channel_len = cache_last_channel_len.to(self.device)

        # Pre-encode cache (mel-spectrogram tail for continuity across chunks)
        cache_pre_encode = torch.zeros(
            (1, self.num_features, self.pre_encode_cache_size),
            device=self.device,
            dtype=self.amp_dtype,
        )

        session_state = {
            "cache_last_channel": cache_last_channel,
            "cache_last_time": cache_last_time,
            "cache_last_channel_len": cache_last_channel_len,
            "cache_pre_encode": cache_pre_encode,
            "previous_hypotheses": None,
            "pred_out_stream": None,
            "step_num": 0,
            "cumulative_text": "",
        }

        with self._session_lock:
            self._sessions[session_id] = session_state
            self._session_timestamps[session_id] = time.time()

        print(f"[Streaming ASR] Initialized cache for session {session_id}")
        return session_state

    def _warmup(self):
        """Warm up the model with a dummy streaming step."""
        print("[Streaming ASR] Warming up...")
        try:
            # Initialize a dummy session cache
            cache_last_channel, cache_last_time, cache_last_channel_len = \
                self.asr_model.encoder.get_initial_cache_state(batch_size=1)
            cache_last_channel = cache_last_channel.to(self.device)
            cache_last_time = cache_last_time.to(self.device)
            cache_last_channel_len = cache_last_channel_len.to(self.device)

            cache_pre_encode = torch.zeros(
                (1, self.num_features, self.pre_encode_cache_size),
                device=self.device,
                dtype=self.amp_dtype,
            )

            # Create 1s of dummy audio
            dummy_audio = torch.randn(1, self.target_sr, device=self.device, dtype=self.amp_dtype)
            dummy_length = torch.tensor([self.target_sr], device=self.device)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.amp_dtype):
                processed_signal, processed_signal_length = self.asr_model.preprocessor(
                    input_signal=dummy_audio.float(), length=dummy_length
                )
                processed_signal = processed_signal.to(self.amp_dtype)
                processed_signal = torch.cat([cache_pre_encode, processed_signal], dim=-1)
                processed_signal_length += cache_pre_encode.shape[2]

                self.asr_model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=None,
                    previous_pred_out=None,
                    drop_extra_pre_encoded=self.pre_encode_cache_size,
                    return_transcription=True,
                )

            print("[Streaming ASR] Warmup complete")
        except Exception as e:
            print(f"[Streaming ASR] Warmup failed (non-critical): {e}")
            import traceback
            traceback.print_exc()

    def _cleanup_stale_sessions(self, max_age_seconds=3600):
        """Remove session caches older than max_age_seconds."""
        now = time.time()
        with self._session_lock:
            stale = [
                sid
                for sid, ts in self._session_timestamps.items()
                if now - ts > max_age_seconds
            ]
            for sid in stale:
                self._sessions.pop(sid, None)
                self._session_timestamps.pop(sid, None)
        if stale:
            print(f"[Streaming ASR] Cleaned up {len(stale)} stale sessions")
            torch.cuda.empty_cache()

    def _maybe_cleanup_stale_sessions(self):
        now = time.time()
        if now - self._last_cleanup_at < self._cleanup_interval_seconds:
            return
        self._last_cleanup_at = now
        self._cleanup_stale_sessions()

    def execute(self, requests):
        """Execute cache-aware streaming inference on audio chunks."""
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

                # Resample if needed
                if sample_rate != self.target_sr:
                    from scipy.signal import resample
                    num_samples = int(len(audio_data) * self.target_sr / sample_rate)
                    audio_data = resample(audio_data, num_samples).astype(np.float32)

                # Get or create session cache
                with self._session_lock:
                    session = self._sessions.get(session_id)
                    self._session_timestamps[session_id] = time.time()

                if session is None:
                    session = self._init_session_cache(session_id)

                # Convert audio to tensor
                audio_torch = torch.tensor(audio_data, device=self.device, dtype=torch.float32).unsqueeze(0)
                audio_length = torch.tensor([len(audio_data)], device=self.device)

                with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    # Preprocess to mel-spectrogram
                    processed_signal, processed_signal_length = self.asr_model.preprocessor(
                        input_signal=audio_torch, length=audio_length
                    )
                    processed_signal = processed_signal.to(self.amp_dtype)

                    # Prepend pre-encode cache for mel continuity across chunks
                    processed_signal = torch.cat(
                        [session["cache_pre_encode"], processed_signal], dim=-1
                    )
                    processed_signal_length += session["cache_pre_encode"].shape[2]

                    # Save tail of current signal as pre-encode cache for next chunk
                    session["cache_pre_encode"] = processed_signal[:, :, -self.pre_encode_cache_size:]

                    # Core streaming call with all cache state
                    (
                        pred_out_stream,
                        transcribed_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                    ) = self.asr_model.conformer_stream_step(
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        cache_last_channel=session["cache_last_channel"],
                        cache_last_time=session["cache_last_time"],
                        cache_last_channel_len=session["cache_last_channel_len"],
                        keep_all_outputs=False,
                        previous_hypotheses=session["previous_hypotheses"],
                        previous_pred_out=session["pred_out_stream"],
                        drop_extra_pre_encoded=self.pre_encode_cache_size,
                        return_transcription=True,
                    )

                    # Update session cache for next chunk
                    session["cache_last_channel"] = cache_last_channel
                    session["cache_last_time"] = cache_last_time
                    session["cache_last_channel_len"] = cache_last_channel_len
                    session["previous_hypotheses"] = previous_hypotheses
                    session["pred_out_stream"] = pred_out_stream
                    session["step_num"] += 1

                # Extract transcription — conformer_stream_step returns Hypothesis objects
                if transcribed_texts:
                    hyp = transcribed_texts[0]
                    if hasattr(hyp, "text"):
                        text = hyp.text
                    elif isinstance(hyp, str):
                        text = hyp
                    else:
                        text = str(hyp)
                else:
                    text = ""
                session["cumulative_text"] = text

                output_data = {
                    "text": text,
                    "word_timestamps": [],  # conformer_stream_step doesn't return per-word timestamps
                    "session_id": session_id,
                    "is_final": is_final,
                    "step_num": session["step_num"],
                }

                # Clean up session on final chunk
                if is_final:
                    with self._session_lock:
                        self._sessions.pop(session_id, None)
                        self._session_timestamps.pop(session_id, None)
                    print(f"[Streaming ASR] Session {session_id} finalized after {output_data['step_num']} steps")

                # Periodic cleanup of stale sessions, even under light load.
                self._maybe_cleanup_stale_sessions()

                output_json = json.dumps(output_data)
                output_array = np.array([[output_json]], dtype=object)
                output_tensor = pb_utils.Tensor("transcription", output_array)
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[output_tensor])
                )

            except Exception as e:
                print(f"[Streaming ASR] Error: {e}")
                import traceback
                traceback.print_exc()
                error_data = {
                    "text": "",
                    "word_timestamps": [],
                    "error": str(e),
                }
                error_json = json.dumps(error_data)
                error_array = np.array([[error_json]], dtype=object)
                error_tensor = pb_utils.Tensor("transcription", error_array)
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[error_tensor])
                )

        return responses

    def finalize(self):
        """Clean up resources."""
        print("[Streaming ASR] Finalizing...")
        try:
            del self.asr_model
            self._sessions.clear()
            self._session_timestamps.clear()
            torch.cuda.empty_cache()
            gc.collect()
            print("[Streaming ASR] Finalized successfully")
        except Exception as e:
            print(f"[Streaming ASR] Error during finalization: {e}")
