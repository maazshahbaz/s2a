"""
Triton Inference Server — Speaker Diarization
Model:  diar_msdd_telephonic  (NVIDIA NeMo)
Domain: 2-speaker telephonic calls
Target: Any CUDA GPU

MSDD CONFIGURATION
──────────────────
Based on 5 temporal scales tuned for telephonic speech:

  Scale  Window   Hop      Purpose
  ─────  ──────   ───      ───────
  0      1.50 s   0.75 s   Coarsest — stable long-range speaker identity
  1      1.25 s   0.625 s
  2      1.00 s   0.50 s
  3      0.75 s   0.375 s
  4      0.50 s   0.25 s   Finest   — defines 0.25 s base temporal resolution

The base grid is driven by Scale 4: windows are centred every 0.25 s,
giving a default temporal resolution of 0.25 s.  All five TitaNet
embeddings are extracted for every grid step and stacked into a
multi-scale sequence that the MSDD LSTM decoder uses to assign speaker
labels.

PIPELINE (7 steps)
──────────────────
  1. VAD         → MarbleNet — speech/non-speech regions
  2. Embeddings  → TitaNet   — [T, 5, 192] multi-scale embeddings
                               one batched forward pass per scale
  3. Clustering  → SpectralClustering on cosine affinity
  4. Cluster-avg → ms_avg_embs [5, 192, 2] — per-speaker per-scale mean
  5. MSDD        → CNN + 3-layer LSTM decoder — refined speaker labels [T]
                   fallback: cosine-similarity assignment
  6. Segments    → midpoint-boundary grouping (zero overlap, zero gap)
  7. Merge       → fuse same-speaker segments separated by ≤ 0.3 s
"""

import os
import json
import uuid
import logging
import warnings
from typing import List, Dict, Any, Tuple

import numpy as np
import triton_python_backend_utils as pb_utils

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


class TritonPythonModel:

    # Five temporal scales (window_s, hop_s) — MSDD telephonic configuration.
    # Scale 4 is the finest and defines the base 0.25 s temporal grid.
    MSDD_SCALES = [
        (1.50, 0.75),
        (1.25, 0.625),
        (1.00, 0.50),
        (0.75, 0.375),
        (0.50, 0.25),   # ← base scale; hop = temporal resolution
    ]
    SCALE_N    = 5
    EMB_DIM    = 192
    NUM_SPKS   = 2
    SAMPLE_RATE = 16_000

    # TitaNet batch size — how many windows are forwarded in one call.
    # Lower this if you run into GPU OOM.
    TITANET_BATCH = 128

    def initialize(self, args):
        self.model_config        = json.loads(args["model_config"])
        self.model_instance_name = args["model_instance_name"]
        self.device_id           = int(args["model_instance_device_id"])
        self.device              = f"cuda:{self.device_id}"

        print(f"[Diarizer] Initialising on {self.device}")

        self.work_dir = f"/tmp/diarizer_{self.model_instance_name}"
        os.makedirs(self.work_dir, exist_ok=True)

        self._load_models()
        print("[Diarizer] ✓ Ready")

    def _load_models(self):
        import torch
        from nemo.collections.asr.models import (
            EncDecClassificationModel,
            EncDecDiarLabelModel,
        )

        print("[Diarizer] Loading VAD (MarbleNet)...")
        self.vad_model = (
            EncDecClassificationModel
            .from_pretrained("vad_multilingual_marblenet")
            .to(self.device)
            .eval()
        )
        print("  ✓ VAD loaded")

        # ── MSDD checkpoint — contains joint-trained TitaNet + decoder ────
        print("[Diarizer] Loading MSDD telephonic checkpoint...")
        msdd_full = (
            EncDecDiarLabelModel
            .from_pretrained("diar_msdd_telephonic")
            .to(self.device)
            .eval()
        )

        # TitaNet speaker encoder (used for multi-scale embedding extraction)
        self.titanet = msdd_full.msdd._speaker_model.to(self.device).eval()

        # MSDD decoder: CNN scale-weighter + 3-layer LSTM
        # forward(ms_emb_seq, ms_avg_embs, length) → pairwise speaker probs
        self.msdd_module = msdd_full.msdd.to(self.device).eval()

        # Hold a reference so the sub-models stay alive
        self._msdd_ref = msdd_full

        print("  ✓ MSDD checkpoint loaded (TitaNet + decoder)")

    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Resample to 16 kHz, mono, normalise to ±0.95."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sample_rate != self.SAMPLE_RATE:
            from scipy.signal import resample
            audio = resample(
                audio, int(len(audio) * self.SAMPLE_RATE / sample_rate)
            )
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return np.clip(audio * 0.95, -1.0, 1.0).astype(np.float32)


    def run_vad(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """
        Run MarbleNet VAD and return speech segments as (start_s, end_s) pairs.

        MarbleNet operates on 25 ms frames with 10 ms hop (standard NeMo
        settings), outputting a speech probability per frame.
        """
        import torch

        n      = len(audio)
        audio_t = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        length  = torch.tensor([n], device=self.device)

        with torch.no_grad():
            logits = self.vad_model(
                input_signal=audio_t,
                input_signal_length=length,
            )
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        probs = torch.softmax(logits.float(), dim=-1)

        if probs.dim() == 3:
            # Frame-level output [1, T, 2]: speech is class 1
            return self._probs_to_segments(probs[0, :, 1].cpu().numpy())
        elif probs.dim() == 2 and probs.shape[0] == 1:
            # Single-chunk utterance-level output [1, 2]
            if probs[0, 1].item() > 0.5:
                return [(0.0, n / self.SAMPLE_RATE)]
            return []
        # Fallback: treat the whole file as speech
        return [(0.0, n / self.SAMPLE_RATE)]

    def _probs_to_segments(
        self,
        speech_probs:   np.ndarray,
        threshold:      float = 0.5,
        min_duration:   float = 0.2,
        frame_duration: float = 0.01,   # 10 ms NeMo VAD hop
    ) -> List[Tuple[float, float]]:
        is_speech = speech_probs > threshold
        segments, in_speech, start = [], False, 0
        for i, active in enumerate(is_speech):
            if active and not in_speech:
                in_speech, start = True, i
            elif not active and in_speech:
                in_speech = False
                s, e = start * frame_duration, i * frame_duration
                if e - s >= min_duration:
                    segments.append((s, e))
        if in_speech:
            s = start * frame_duration
            e = len(is_speech) * frame_duration
            if e - s >= min_duration:
                segments.append((s, e))
        return segments


    def _build_window_plan(
        self,
        audio_dur:    float,
        vad_segments: List[Tuple[float, float]],
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """
        Build the base temporal grid using Scale 4 (0.50 s window, 0.25 s hop).

        Returns
        ───────
        centres        : list of window-centre times [T]
        embedding_times: list of (start_s, end_s) pairs [T]
        """
        base_win, base_hop = self.MSDD_SCALES[-1]  # 0.50 s, 0.25 s
        centres, times = [], []
        for seg_start, seg_end in vad_segments:
            c = seg_start + base_win / 2
            while c + base_win / 2 <= seg_end:
                centres.append(c)
                times.append((
                    round(c - base_win / 2, 4),
                    round(c + base_win / 2, 4),
                ))
                c += base_hop
        return centres, times

    def _extract_scale(
        self,
        audio:    np.ndarray,
        centres:  List[float],
        win_len:  float,
    ) -> np.ndarray:
        """
        Extract TitaNet embeddings for all T windows at one scale.

        Windows are gathered into batches of TITANET_BATCH and forwarded
        together — one batched forward pass per TITANET_BATCH windows rather
        than T sequential single-sample calls.

        Returns [T, 192] float32.
        """
        import torch

        T         = len(centres)
        half      = win_len / 2
        nom_n     = int(win_len * self.SAMPLE_RATE)
        audio_dur = len(audio) / self.SAMPLE_RATE
        embs_out  = np.empty((T, self.EMB_DIM), dtype=np.float32)

        for batch_start in range(0, T, self.TITANET_BATCH):
            batch_end = min(batch_start + self.TITANET_BATCH, T)
            bs        = batch_end - batch_start

            batch   = torch.zeros(bs, nom_n, dtype=torch.float32)
            lengths = torch.zeros(bs, dtype=torch.long)

            for i, c in enumerate(centres[batch_start:batch_end]):
                s = int(max(0.0, c - half) * self.SAMPLE_RATE)
                e = int(min(audio_dur, c + half) * self.SAMPLE_RATE)
                n = e - s
                if n > 0:
                    batch[i, :n] = torch.from_numpy(audio[s:e])
                    lengths[i]   = n
                else:
                    lengths[i] = 1

            batch   = batch.to(self.device)
            lengths = lengths.to(self.device)

            with torch.no_grad():
                _, embs = self.titanet(
                    input_signal=batch,
                    input_signal_length=lengths,
                )
            embs_out[batch_start:batch_end] = embs.float().cpu().numpy()

        return embs_out  # [T, 192]

    def extract_embeddings(
        self,
        audio:        np.ndarray,
        vad_segments: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
        """
        Extract multi-scale TitaNet embeddings.

        Executes SCALE_N=5 batched TitaNet forward passes (one per scale),
        each processing all T base-grid windows together.

        Returns
        ───────
        emb_per_scale  : [T, 5, 192]    — per-scale embeddings
        emb_flat       : [T, 960]       — concatenated scales (for clustering)
        embedding_times: list of (start_s, end_s) length T
        """
        audio_dur          = len(audio) / self.SAMPLE_RATE
        centres, emb_times = self._build_window_plan(audio_dur, vad_segments)
        T                  = len(centres)

        if T == 0:
            return np.array([]), np.array([]), []

        emb_per_scale = np.empty((T, self.SCALE_N, self.EMB_DIM), dtype=np.float32)
        for s_idx, (win_len, _) in enumerate(self.MSDD_SCALES):
            print(f"    scale {s_idx} (win={win_len:.2f}s): {T} windows, "
                  f"batch={self.TITANET_BATCH}")
            emb_per_scale[:, s_idx, :] = self._extract_scale(
                audio, centres, win_len
            )

        emb_flat = emb_per_scale.reshape(T, -1)  # [T, 960]
        return emb_per_scale, emb_flat, emb_times


    def cluster_embeddings(
        self,
        emb_flat:     np.ndarray,
        num_speakers: int = 2,
    ) -> np.ndarray:
        """
        SpectralClustering with cosine affinity on the flattened [T, 960]
        multi-scale embeddings.
        """
        import torch
        import torch.nn.functional as F
        from sklearn.cluster import SpectralClustering

        if len(emb_flat) < num_speakers:
            return np.zeros(len(emb_flat), dtype=int)

        emb_t    = torch.from_numpy(emb_flat).float()
        emb_norm = F.normalize(emb_t, dim=-1)
        aff      = ((torch.mm(emb_norm, emb_norm.T) + 1.0) / 2.0).numpy()

        try:
            return SpectralClustering(
                n_clusters=num_speakers,
                affinity="precomputed",
                n_init=10,
                random_state=42,
            ).fit_predict(aff)
        except Exception:
            labels = np.zeros(len(emb_flat), dtype=int)
            labels[len(emb_flat) // 2:] = 1
            return labels


    def compute_cluster_avg_embs(
        self,
        emb_per_scale:  np.ndarray,
        cluster_labels: np.ndarray,
        num_speakers:   int = 2,
    ) -> np.ndarray:
        """
        Per-speaker per-scale mean embedding.

        ms_avg_embs[s, :, k] = mean of emb_per_scale[t, s, :]
                                for all t where cluster_labels[t] == k

        Returns [SCALE_N, EMB_DIM, NUM_SPKS] = [5, 192, 2].
        """
        ms_avg = np.zeros(
            (self.SCALE_N, self.EMB_DIM, num_speakers), dtype=np.float32
        )
        for k in range(num_speakers):
            mask = cluster_labels == k
            src  = emb_per_scale[mask] if mask.any() else emb_per_scale
            ms_avg[:, :, k] = src.mean(axis=0)   # [5, 192]
        return ms_avg


    def run_msdd_decoder(
        self,
        emb_per_scale:  np.ndarray,   # [T, 5, 192]
        ms_avg_embs_np: np.ndarray,   # [5, 192, 2]
    ) -> np.ndarray:
        """
        MSDD neural decoder — CNN scale-weighter + 3-layer LSTM.

        Inputs (batched as single sequence of length T):
          ms_emb_seq  [1, T, 960]    — flattened multi-scale embeddings
          ms_avg_embs [1, 5, 192, 2] — per-speaker per-scale mean embeddings
          length      [1]            — sequence length T

        Output:
          labels [T] int — speaker assignment per base-grid step
        """
        import torch

        T = emb_per_scale.shape[0]

        ms_emb_seq = (
            torch.from_numpy(emb_per_scale.reshape(T, -1))
            .unsqueeze(0)
            .float()
            .to(self.device)
        )  # [1, T, 960]

        ms_avg_embs = (
            torch.from_numpy(ms_avg_embs_np)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )  # [1, 5, 192, 2]

        length = torch.tensor([T], dtype=torch.long, device=self.device)

        with torch.no_grad():
            output = self.msdd_module(
                ms_emb_seq  = ms_emb_seq,
                ms_avg_embs = ms_avg_embs,
                length      = length,
            )

        preds = output[0] if isinstance(output, (tuple, list)) else output
        # preds: [1, T, 2] — sigmoid pairwise probabilities
        return preds.squeeze(0).float().cpu().argmax(dim=-1).numpy().astype(int)

    def _cosine_fallback_labels(
        self,
        emb_per_scale:  np.ndarray,
        ms_avg_embs_np: np.ndarray,
    ) -> np.ndarray:
        """
        Fallback when MSDD decoder fails: assign each window to the speaker
        whose cluster-average embedding has the highest mean cosine similarity
        across all 5 scales.

        cos_sim[t, s, k] = normalised_dot(emb_per_scale[t,s,:], avg[s,:,k])
        label[t]         = argmax_k  mean_s  cos_sim[t, s, k]
        """
        import torch
        import torch.nn.functional as F

        emb = torch.from_numpy(emb_per_scale).float()   # [T, 5, 192]
        avg = torch.from_numpy(ms_avg_embs_np).float()  # [5, 192, 2]

        emb_n   = F.normalize(emb, dim=-1)
        avg_n   = F.normalize(avg, dim=1)
        cos_sim = torch.einsum("tse,sek->tsk", emb_n, avg_n)  # [T, 5, 2]
        return cos_sim.mean(dim=1).argmax(dim=-1).numpy().astype(int)

    # ──────────────────────── Labels → Segments ─────────────────────────────

    def _labels_to_segments(
        self,
        times:        List[Tuple[float, float]],
        labels:       np.ndarray,
        min_duration: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Convert per-window labels to speaker segments.

        At each speaker transition, the boundary is set to the midpoint
        between the end of the last window of the outgoing speaker and the
        start of the first window of the incoming speaker.  This guarantees
        zero overlap and zero gap between adjacent segments.
        """
        if len(labels) == 0:
            return []

        segments       = []
        current_label  = labels[0]
        seg_start      = times[0][0]

        for i in range(1, len(labels)):
            if labels[i] != current_label:
                boundary = (times[i - 1][1] + times[i][0]) / 2
                if boundary - seg_start >= min_duration:
                    segments.append({
                        "speaker":  f"speaker_{current_label}",
                        "start":    round(seg_start, 4),
                        "end":      round(boundary,  4),
                        "duration": round(boundary - seg_start, 4),
                    })
                seg_start     = boundary
                current_label = labels[i]

        end_time = times[-1][1]
        if end_time - seg_start >= min_duration:
            segments.append({
                "speaker":  f"speaker_{current_label}",
                "start":    round(seg_start, 4),
                "end":      round(end_time,  4),
                "duration": round(end_time - seg_start, 4),
            })
        return segments


    def merge_segments(
        self,
        segments:      List[Dict[str, Any]],
        gap_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Merge same-speaker segments separated by ≤ gap_threshold seconds.
        0.3 s retains short genuine turns ("Okay.", "Yes.").
        """
        if not segments:
            return []
        segments = sorted(segments, key=lambda x: x["start"])
        merged   = [segments[0].copy()]
        for seg in segments[1:]:
            prev = merged[-1]
            if (seg["speaker"] == prev["speaker"] and
                    seg["start"] - prev["end"] <= gap_threshold):
                prev["end"]      = seg["end"]
                prev["duration"] = round(prev["end"] - prev["start"], 3)
            else:
                merged.append(seg.copy())
        return merged


    def process_audio(
        self, audio: np.ndarray, request_id: str
    ) -> Dict[str, Any]:
        import time
        duration = len(audio) / self.SAMPLE_RATE
        t0       = time.time()

        print(f"\n{'='*70}")
        print(f"[{request_id}] Processing {duration:.2f}s audio")
        print(f"{'='*70}")

        try:
            # Step 1 — VAD
            print(f"[{request_id}] Step 1: VAD...")
            vad_segs = self.run_vad(audio)
            print(f"[{request_id}]   ✓ {len(vad_segs)} speech segment(s)")
            if not vad_segs:
                return self._empty_result(duration, time.time() - t0)

            # Step 2 — Multi-scale embeddings
            print(f"[{request_id}] Step 2: Multi-scale embeddings "
                  f"({self.SCALE_N} scales, 0.25 s base resolution)...")
            emb_per_scale, emb_flat, emb_times = self.extract_embeddings(
                audio, vad_segs
            )
            T = len(emb_times)
            print(f"[{request_id}]   ✓ {T} base-grid steps "
                  f"(temporal resolution = 0.25 s)")
            if T == 0:
                return self._empty_result(duration, time.time() - t0)

            # Step 3 — Spectral clustering
            print(f"[{request_id}] Step 3: SpectralClustering...")
            clus_labels = self.cluster_embeddings(emb_flat, self.NUM_SPKS)
            print(f"[{request_id}]   ✓ {len(set(clus_labels.tolist()))} "
                  f"cluster(s) found")

            # Step 4 — Cluster-average embeddings
            print(f"[{request_id}] Step 4: Cluster-average embeddings...")
            ms_avg_embs = self.compute_cluster_avg_embs(
                emb_per_scale, clus_labels, self.NUM_SPKS
            )

            # Step 5 — MSDD decoder
            print(f"[{request_id}] Step 5: MSDD decoder...")
            try:
                labels = self.run_msdd_decoder(emb_per_scale, ms_avg_embs)
            except Exception as e:
                print(f"[{request_id}]   [WARN] MSDD failed ({e}), "
                      f"using cosine-sim fallback")
                labels = self._cosine_fallback_labels(emb_per_scale, ms_avg_embs)
            print(f"[{request_id}]   ✓ {len(set(labels.tolist()))} "
                  f"speaker(s) assigned")

            # Step 6 — Build segments
            print(f"[{request_id}] Step 6: Building segments...")
            segments = self._labels_to_segments(emb_times, labels)
            print(f"[{request_id}]   ✓ {len(segments)} raw segment(s)")

            # Step 7 — Merge
            print(f"[{request_id}] Step 7: Merging (gap ≤ 0.3 s)...")
            segments = self.merge_segments(segments)

            elapsed = time.time() - t0
            spks    = {s["speaker"] for s in segments} if segments else set()
            print(f"[{request_id}] ✓ Done: {len(segments)} segment(s), "
                  f"{len(spks)} speaker(s) in {elapsed:.2f}s")
            print(f"{'='*70}\n")

            return {
                "segments":       segments,
                "num_speakers":   len(spks),
                "total_segments": len(segments),
                "duration":       round(duration, 3),
                "inference_time": round(elapsed, 3),
                "status":         "success",
            }

        except Exception as e:
            import traceback
            print(f"[{request_id}] ✗ Error: {e}\n{traceback.format_exc()}")
            raise

    @staticmethod
    def _empty_result(duration: float, elapsed: float) -> Dict[str, Any]:
        return {
            "segments":       [],
            "num_speakers":   0,
            "total_segments": 0,
            "duration":       round(duration, 3),
            "inference_time": round(elapsed, 3),
            "status":         "success",
        }


    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                audio_in = pb_utils.get_input_tensor_by_name(request, "audio_input")
                sr_in    = pb_utils.get_input_tensor_by_name(request, "sample_rate")

                audio       = audio_in.as_numpy()
                if audio.ndim == 2:
                    audio = audio[0]

                sample_rate = int(sr_in.as_numpy().flat[0])
                request_id  = f"{self.model_instance_name}_{uuid.uuid4().hex[:8]}"

                audio  = self.preprocess_audio(audio, sample_rate)
                result = self.process_audio(audio, request_id)

                print(f"diar_result: {json.dumps(result)}\n", flush=True)

                out = pb_utils.Tensor(
                    "diarization_output",
                    np.array([json.dumps(result)], dtype=object),
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))

            except Exception as e:
                import traceback
                print(f"[ERROR] {e}\n{traceback.format_exc()}")
                err = {
                    "status":       "error",
                    "error":        str(e),
                    "segments":     [],
                    "num_speakers": 0,
                }
                out = pb_utils.Tensor(
                    "diarization_output",
                    np.array([json.dumps(err)], dtype=object),
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[out],
                        error=pb_utils.TritonError(str(e)),
                    )
                )
        return responses

    def finalize(self):
        import shutil
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)
        print(f"[Diarizer] Shutdown: {self.model_instance_name}")