import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import time
from loguru import logger
import librosa
import soundfile as sf
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from .chunking_utils import ChunkingManager, AudioChunk

# NeMo imports - required for this service
try:
    import nemo.collections.asr as nemo_asr
    from nemo.core.config import hydra_runner
    logger.info("NeMo toolkit available")
except ImportError as e:
    logger.error(f"NeMo toolkit is required but not available: {e}")
    raise ImportError("NeMo toolkit is required for this ASR service. Please install with: pip install nemo_toolkit[asr]")

@dataclass
class TranscriptionResult:
    text: str
    duration: float
    rtf: float
    chunks: List[Dict] = None
    confidence: float = None
    processing_time: float = None
    model_used: str = None
    chunks_processed: int = 0

class NeMoASRService:
    def __init__(self,
                 model_name: str,
                 device: str,
                 batch_size: int,
                 max_chunk_duration: float,
                 min_audio_duration: float,
                 overlap_duration: float,
                 target_sample_rate: int,
                 words_per_second: float,
                 overlap_similarity_threshold: float):

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_chunk_duration = max_chunk_duration
        self.min_audio_duration = min_audio_duration
        self.overlap_duration = overlap_duration
        self.target_sample_rate = target_sample_rate
        self.words_per_second = words_per_second
        self.overlap_similarity_threshold = overlap_similarity_threshold
        self.model_type = None
        self.model = None

        # Initialize chunking manager for NeMo (using config values)
        self.chunking_manager = ChunkingManager(
            max_chunk_duration=self.max_chunk_duration,
            overlap_duration=self.overlap_duration,
            words_per_second=words_per_second,
            overlap_similarity_threshold=overlap_similarity_threshold
        )
        
        logger.info(f"Initializing ASR service with model: {model_name}")
        logger.info(f"Device: {device}, Batch size: {batch_size}")
        
        self._load_nemo_model()
        self._setup_h100_optimizations()
        
    def _load_nemo_model(self):
        """Load NeMo Parakeet model - the only model supported"""
        try:
            logger.info(f"Loading NeMo Parakeet model: {self.model_name}")
            
            # Load NeMo Parakeet model
            self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
            
            # Move to device and optimize for H100
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
                # Enable mixed precision for H100 (compute capability >= 8.0)
                try:
                    device_capability = torch.cuda.get_device_capability()
                    if device_capability and device_capability[0] >= 8:
                        self.model = self.model.half()
                        logger.info("Enabled FP16 precision for H100 GPU")
                except Exception as e:
                    logger.warning(f"Could not check GPU capability: {e}. Using FP32.")
            
            self.model.eval()
            self.model_type = "nemo"
            
            logger.info(f"NeMo Parakeet model {self.model_name} loaded successfully")
            self._warmup_nemo_model()
            
        except Exception as e:
            logger.error(f"Failed to load NeMo model {self.model_name}: {e}")
            raise RuntimeError(f"NeMo model loading failed: {e}")
    
    
    def _setup_h100_optimizations(self):
        """Setup H100-specific optimizations"""
        if self.device == "cuda" and torch.cuda.is_available():
            # Enable optimized attention for H100
            torch.backends.cuda.enable_flash_sdp(True)

            # Set memory format for optimal H100 performance
            torch.backends.cudnn.benchmark = True

            # Enable Tensor Core usage (TF32 for Tensor Cores)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            device_name = torch.cuda.get_device_name(0)
            if "H100" in device_name:
                logger.info(f"H100 optimizations enabled for {device_name}")
                logger.info(f"Tensor Cores, Flash Attention, and TF32 enabled")
                logger.info(f"Using configured batch_size: {self.batch_size}")
    
    def _warmup_nemo_model(self):
        logger.info("Warming up NeMo model...")
        try:
            # Create dummy audio for warmup
            dummy_audio = np.random.randn(self.target_sample_rate * 10).astype(np.float32)  # 10 seconds
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, dummy_audio, self.target_sample_rate)
                # Transcribe dummy audio
                _ = self.model.transcribe([tmp_file.name])
                os.unlink(tmp_file.name)
            logger.info("NeMo model warmup completed")
        except Exception as e:
            logger.warning(f"NeMo model warmup failed: {e}")
    
    
    
    def preprocess_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, float, bool]:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio using librosa for consistent preprocessing
            audio, sr = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)
            duration = len(audio) / sr

            # Check minimum duration
            if duration < self.min_audio_duration:
                logger.warning(f"Audio duration {duration:.2f}s is below minimum {self.min_audio_duration}s")
                return audio, duration, False

            # Normalize audio
            audio = librosa.util.normalize(audio)

            logger.info(f"Preprocessed audio: duration={duration:.2f}s, sample_rate={self.target_sample_rate}Hz, model={self.model_type}")
            return audio, duration, True

        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
    
    def chunk_audio_intelligent(self, audio: np.ndarray, sr: int = None) -> Tuple[List[AudioChunk], callable]:
        """Use intelligent chunking for NeMo model (up to 24 minutes per chunk)"""
        if sr is None:
            sr = self.target_sample_rate
        total_duration = len(audio) / sr

        # Use intelligent chunking for NeMo (24-minute chunks)
        logger.info(f"Using intelligent chunking for NeMo model (24-min chunks)")
        return self.chunking_manager.process_long_audio(audio, sr)
    
    def chunk_audio_simple(self, audio: np.ndarray, sr: int = None) -> List[np.ndarray]:
        """Fallback simple chunking method for NeMo"""
        if sr is None:
            sr = self.target_sample_rate
        total_duration = len(audio) / sr

        # Use NeMo chunk duration (24 minutes)
        chunk_duration = self.max_chunk_duration

        if total_duration <= chunk_duration:
            return [audio]

        chunk_samples = int(chunk_duration * sr)
        chunks = []

        # NeMo overlap (5 seconds)
        overlap_samples = int(self.overlap_duration * sr)

        for start in range(0, len(audio), chunk_samples - overlap_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            # Skip chunks that are too short
            if len(chunk) >= int(self.min_audio_duration * sr):
                chunks.append(chunk)

        logger.info(f"Split {total_duration:.1f}s audio into {len(chunks)} chunks")
        return chunks
    
    def transcribe_batch(self, audio_chunks: List[AudioChunk]) -> List[Dict]:
        """Transcribe a batch of audio chunks using NeMo model"""
        return self.transcribe_batch_nemo(audio_chunks)
    
    def transcribe_batch_nemo(self, audio_chunks: List[AudioChunk]) -> List[Dict]:
        """Transcribe chunks using NeMo model with optimized batching"""
        results = []

        # Prepare audio files for NeMo (it expects file paths)
        temp_files = []
        try:
            # Save chunks to temporary files
            for i, chunk in enumerate(audio_chunks):
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(temp_file.name, chunk.audio_data, self.target_sample_rate)
                temp_files.append(temp_file.name)
                temp_file.close()
            
            # Process in batches
            for i in range(0, len(temp_files), self.batch_size):
                batch_files = temp_files[i:i + self.batch_size]
                batch_chunks = audio_chunks[i:i + self.batch_size]
                
                start_time = time.time()
                
                # NeMo batch transcription
                transcriptions = self.model.transcribe(batch_files)
                
                processing_time = time.time() - start_time
                
                # Process results
                for j, (transcription, chunk) in enumerate(zip(transcriptions, batch_chunks)):
                    rtf = processing_time / (chunk.duration * len(batch_chunks))
                    
                    # Convert NeMo Hypothesis to text
                    if hasattr(transcription, 'text'):
                        text = transcription.text
                    elif isinstance(transcription, str):
                        text = transcription
                    else:
                        text = str(transcription)
                    
                    results.append({
                        "text": text.strip(),
                        "duration": chunk.duration,
                        "rtf": rtf,
                        "processing_time": processing_time / len(batch_chunks),
                        "chunk_index": i + j,
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time
                    })
                
                logger.info(f"Processed NeMo batch of {len(batch_chunks)} chunks, RTF: {rtf:.3f}")
        
        except Exception as e:
            logger.error(f"Error in NeMo batch processing: {e}")
            # Return error results
            for i, chunk in enumerate(audio_chunks):
                results.append({
                    "text": "",
                    "duration": chunk.duration,
                    "rtf": float('inf'),
                    "processing_time": 0,
                    "chunk_index": i,
                    "error": str(e)
                })
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        
        return results

    def stitch_transcriptions(self, chunk_results: List[Dict], stitch_function: callable = None) -> str:
        """Stitch transcription results using intelligent or simple method"""
        if stitch_function:
            # Use intelligent stitching
            logger.info("Using intelligent stitching for NeMo transcription")
            stitched_result = stitch_function(chunk_results)
            return stitched_result.get("text", "")
        else:
            # Use simple stitching as fallback
            logger.info("Using simple stitching")
            valid_chunks = [r for r in chunk_results if r.get("text") and not r.get("error")]

            if not valid_chunks:
                logger.warning("No valid transcriptions to stitch")
                return ""

            # Simple stitching with cleanup
            texts = [chunk["text"] for chunk in valid_chunks]
            stitched_text = " ".join(texts)
            stitched_text = " ".join(stitched_text.split())  # Remove extra whitespace

            logger.info(f"Stitched {len(valid_chunks)} chunks into final transcription")
            return stitched_text
    
    async def transcribe_audio(self, audio_path: Union[str, Path]) -> TranscriptionResult:
        start_time = time.time()

        try:
            # Preprocess audio
            audio, duration, is_valid = self.preprocess_audio(audio_path)

            if not is_valid:
                logger.warning(f"Skipping invalid audio: {audio_path}")
                return TranscriptionResult(
                    text="",
                    duration=duration,
                    rtf=0,
                    processing_time=time.time() - start_time,
                    model_used=self.model_type
                )

            # Transcribe using NeMo
            result = await self._transcribe_with_nemo(audio, duration, start_time)

            logger.info(f"Transcription completed: duration={duration:.2f}s, "
                       f"RTF={result.rtf:.3f}, chunks={result.chunks_processed}")

            return result

        except Exception as e:
            logger.error(f"Error transcribing audio {audio_path}: {e}")
            return TranscriptionResult(
                text="",
                duration=0,
                rtf=float('inf'),
                processing_time=time.time() - start_time,
                model_used=self.model_type
            )
    
    async def _transcribe_with_nemo(self, audio: np.ndarray, duration: float, start_time: float) -> TranscriptionResult:
        """Transcribe using NeMo model with intelligent chunking"""
        try:
            # Use intelligent chunking for NeMo (up to 24 minutes per chunk)
            audio_chunks, stitch_function = self.chunk_audio_intelligent(audio, self.target_sample_rate)
            
            # Transcribe chunks using NeMo
            chunk_results = self.transcribe_batch_nemo(audio_chunks)
            
            # Use intelligent stitching
            final_text = self.stitch_transcriptions(chunk_results, stitch_function)
            
            processing_time = time.time() - start_time
            rtf = processing_time / duration if duration > 0 else float('inf')
            
            # Calculate average confidence if available
            valid_chunks = [r for r in chunk_results if not r.get("error")]
            avg_confidence = None
            if valid_chunks and all("confidence" in r for r in valid_chunks):
                avg_confidence = sum(r["confidence"] for r in valid_chunks) / len(valid_chunks)
            
            return TranscriptionResult(
                text=final_text,
                duration=duration,
                rtf=rtf,
                chunks=chunk_results,
                confidence=avg_confidence,
                processing_time=processing_time,
                model_used="nemo",
                chunks_processed=len(audio_chunks)
            )
        
        except Exception as e:
            logger.error(f"Error in NeMo transcription: {e}")
            # Fallback to simple chunking
            return await self._transcribe_with_simple_chunking(audio, duration, start_time)

    async def _transcribe_with_simple_chunking(self, audio: np.ndarray, duration: float, start_time: float) -> TranscriptionResult:
        """Fallback transcription with simple chunking for NeMo"""
        logger.info("Using fallback simple chunking")

        # Use simple chunking as fallback
        audio_chunks = self.chunk_audio_simple(audio)

        # Convert to AudioChunk objects for NeMo
        audio_chunk_objects = []
        for i, chunk in enumerate(audio_chunks):
            chunk_duration = len(chunk) / self.target_sample_rate
            audio_chunk_objects.append(AudioChunk(
                audio_data=chunk,
                start_time=i * (chunk_duration - 5),  # Approximate
                end_time=(i + 1) * (chunk_duration - 5),
                duration=chunk_duration,
                chunk_id=i
            ))
        chunk_results = self.transcribe_batch_nemo(audio_chunk_objects)

        # Simple stitching
        final_text = self.stitch_transcriptions(chunk_results)

        processing_time = time.time() - start_time
        rtf = processing_time / duration if duration > 0 else float('inf')

        return TranscriptionResult(
            text=final_text,
            duration=duration,
            rtf=rtf,
            chunks=chunk_results,
            confidence=None,
            processing_time=processing_time,
            model_used=self.model_type,
            chunks_processed=len(audio_chunks)
        )
    
    def get_model_info(self) -> Dict:
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "gpu_compute_capability": torch.cuda.get_device_capability(0)
            }
        else:
            gpu_info = {"gpu_available": False}
        
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_chunk_duration": self.max_chunk_duration,
            "min_audio_duration": self.min_audio_duration,
            "nemo_available": True,  # Always true in NeMo-only service
            "chunking_strategy": "intelligent",  # Always intelligent for NeMo
            "h100_optimizations": "H100" in torch.cuda.get_device_name(0) if torch.cuda.is_available() else False,
            **gpu_info
        }