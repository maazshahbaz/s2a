import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import time
from loguru import logger
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import soundfile as sf
from pydub import AudioSegment
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from chunking_utils import ChunkingManager, AudioChunk

# NeMo imports with fallback
try:
    import nemo.collections.asr as nemo_asr
    from nemo.core.config import hydra_runner
    NEMO_AVAILABLE = True
    logger.info("NeMo toolkit available")
except ImportError as e:
    NEMO_AVAILABLE = False
    logger.warning(f"NeMo toolkit not available: {e}. Will use Whisper fallback.")

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
                 model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 4,
                 max_chunk_duration: float = 24 * 60,  # 24 minutes in seconds
                 min_audio_duration: float = 5.0):
        
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_chunk_duration = max_chunk_duration
        self.min_audio_duration = min_audio_duration
        self.model_type = None
        self.model = None
        self.processor = None
        
        # Initialize chunking managers for different models
        self.nemo_chunking_manager = ChunkingManager(
            max_chunk_duration=24 * 60,  # 24 minutes for NeMo
            overlap_duration=5.0
        )
        self.whisper_chunking_manager = ChunkingManager(
            max_chunk_duration=30,  # 30 seconds for Whisper optimal performance
            overlap_duration=2.0
        )
        
        logger.info(f"Initializing ASR service with model: {model_name}")
        logger.info(f"Device: {device}, Batch size: {batch_size}")
        
        self._load_model()
        self._setup_h100_optimizations()
        
    def _load_model(self):
        # Try loading NeMo Parakeet first
        if NEMO_AVAILABLE and "parakeet" in self.model_name.lower():
            if self._try_load_nemo():
                return
        
        # Fallback to Whisper
        self._load_whisper_fallback()
    
    def _try_load_nemo(self):
        try:
            logger.info(f"Attempting to load NeMo Parakeet model: {self.model_name}")
            
            # Load NeMo Parakeet model
            self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
            
            # Move to device and optimize for H100
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                # Enable mixed precision for H100
                self.model = self.model.half() if torch.cuda.get_device_capability()[0] >= 8 else self.model
            
            self.model.eval()
            self.model_type = "nemo"
            
            logger.info(f"NeMo Parakeet model {self.model_name} loaded successfully")
            self._warmup_nemo_model()
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load NeMo model {self.model_name}: {e}")
            logger.info("Falling back to Whisper model...")
            return False
    
    def _load_whisper_fallback(self):
        try:
            logger.info("Loading Whisper Large V3 as fallback model...")
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                # Enable mixed precision for H100
                if torch.cuda.get_device_capability()[0] >= 8:
                    self.model = self.model.half()
            
            self.model.eval()
            self.model_type = "whisper"
            
            logger.info("Whisper model loaded successfully")
            self._warmup_whisper_model()
            
        except Exception as e:
            logger.error(f"Failed to load both NeMo and Whisper models: {e}")
            raise
    
    def _setup_h100_optimizations(self):
        """Setup H100-specific optimizations"""
        if self.device == "cuda" and torch.cuda.is_available():
            # Enable optimized attention for H100
            torch.backends.cuda.enable_flash_sdp(True)
            
            # Set memory format for optimal H100 performance
            torch.backends.cudnn.benchmark = True
            
            # Enable Tensor Core usage
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            device_name = torch.cuda.get_device_name(0)
            if "H100" in device_name:
                logger.info(f"H100 optimizations enabled for {device_name}")
                # Increase batch size for H100's larger memory
                if self.model_type == "nemo":
                    self.batch_size = min(self.batch_size * 2, 16)
                    logger.info(f"Increased batch size to {self.batch_size} for H100")
    
    def _warmup_nemo_model(self):
        logger.info("Warming up NeMo model...")
        try:
            # Create dummy audio for warmup
            dummy_audio = np.random.randn(16000 * 10).astype(np.float32)  # 10 seconds
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, dummy_audio, 16000)
                # Transcribe dummy audio
                _ = self.model.transcribe([tmp_file.name])
                os.unlink(tmp_file.name)
            logger.info("NeMo model warmup completed")
        except Exception as e:
            logger.warning(f"NeMo model warmup failed: {e}")
    
    def _warmup_whisper_model(self):
        logger.info("Warming up Whisper model...")
        try:
            dummy_audio = torch.randn(1, 16000 * 10).to(self.device)  # 10 seconds
            with torch.no_grad():
                features = self.processor(dummy_audio.cpu().numpy().squeeze(), 
                                        sampling_rate=16000, 
                                        return_tensors="pt").input_features.to(self.device)
                _ = self.model.generate(features, max_length=50)
            logger.info("Whisper model warmup completed")
        except Exception as e:
            logger.warning(f"Whisper model warmup failed: {e}")
    
    
    def preprocess_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, float, bool]:
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio using librosa for consistent preprocessing
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(audio) / sr
            
            # Check minimum duration
            if duration < self.min_audio_duration:
                logger.warning(f"Audio duration {duration:.2f}s is below minimum {self.min_audio_duration}s")
                return audio, duration, False
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            logger.info(f"Preprocessed audio: duration={duration:.2f}s, sample_rate=16000Hz, model={self.model_type}")
            return audio, duration, True
            
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
    
    def chunk_audio_intelligent(self, audio: np.ndarray, sr: int = 16000) -> Tuple[List[AudioChunk], callable]:
        """Use intelligent chunking for NeMo model (up to 24 minutes per chunk)"""
        total_duration = len(audio) / sr
        
        if self.model_type == "nemo":
            # Use intelligent chunking for NeMo (24-minute chunks)
            logger.info(f"Using intelligent chunking for NeMo model (24-min chunks)")
            return self.nemo_chunking_manager.process_long_audio(audio, sr)
        else:
            # Use smaller chunks for Whisper (30-second optimal)
            logger.info(f"Using optimal chunking for Whisper model (30-sec chunks)")
            return self.whisper_chunking_manager.process_long_audio(audio, sr)
    
    def chunk_audio_simple(self, audio: np.ndarray, sr: int = 16000) -> List[np.ndarray]:
        """Fallback simple chunking method"""
        total_duration = len(audio) / sr
        
        # Use model-specific chunk duration
        chunk_duration = self.max_chunk_duration if self.model_type == "nemo" else 30
        
        if total_duration <= chunk_duration:
            return [audio]
        
        chunk_samples = int(chunk_duration * sr)
        chunks = []
        
        # Model-specific overlap
        overlap_samples = int((5 if self.model_type == "nemo" else 2) * sr)
        
        for start in range(0, len(audio), chunk_samples - overlap_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Skip chunks that are too short
            if len(chunk) >= int(self.min_audio_duration * sr):
                chunks.append(chunk)
        
        logger.info(f"Split {total_duration:.1f}s audio into {len(chunks)} chunks using {self.model_type} chunking")
        return chunks
    
    def transcribe_batch_nemo(self, audio_chunks: List[AudioChunk]) -> List[Dict]:
        """Transcribe chunks using NeMo model with optimized batching"""
        results = []
        
        # Prepare audio files for NeMo (it expects file paths)
        temp_files = []
        try:
            # Save chunks to temporary files
            for i, chunk in enumerate(audio_chunks):
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(temp_file.name, chunk.audio_data, 16000)
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
                    
                    results.append({
                        "text": transcription.strip(),
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
    
    def transcribe_batch_whisper(self, audio_chunks: List[np.ndarray]) -> List[Dict]:
        """Transcribe chunks using Whisper model with optimized batching"""
        results = []
        
        for i in range(0, len(audio_chunks), self.batch_size):
            batch = audio_chunks[i:i + self.batch_size]
            batch_results = self._process_whisper_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_whisper_batch(self, audio_batch: List[np.ndarray]) -> List[Dict]:
        """Process a batch of audio chunks using Whisper model"""
        batch_results = []
        
        try:
            # Prepare batch inputs
            features_batch = []
            for audio in audio_batch:
                features = self.processor(audio, 
                                        sampling_rate=16000, 
                                        return_tensors="pt").input_features
                features_batch.append(features)
            
            # Stack features for batch processing
            batch_features = torch.cat(features_batch, dim=0).to(self.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                # Generate transcriptions with H100 optimizations
                generated_tokens = self.model.generate(
                    batch_features,
                    max_length=448,
                    num_beams=1,
                    do_sample=False,
                    temperature=0.0,
                    use_cache=True  # Enable KV cache for H100
                )
                
                # Decode transcriptions
                transcriptions = self.processor.batch_decode(generated_tokens, 
                                                           skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            # Process results
            for i, (audio, transcription) in enumerate(zip(audio_batch, transcriptions)):
                audio_duration = len(audio) / 16000
                rtf = processing_time / (audio_duration * len(audio_batch))
                
                batch_results.append({
                    "text": transcription.strip(),
                    "duration": audio_duration,
                    "rtf": rtf,
                    "processing_time": processing_time / len(audio_batch),
                    "chunk_index": i
                })
            
            logger.info(f"Processed Whisper batch of {len(audio_batch)} chunks, RTF: {rtf:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing Whisper batch: {e}")
            # Return empty results for failed batch
            for i, audio in enumerate(audio_batch):
                batch_results.append({
                    "text": "",
                    "duration": len(audio) / 16000,
                    "rtf": float('inf'),
                    "processing_time": 0,
                    "chunk_index": i,
                    "error": str(e)
                })
        
        return batch_results
    
    def stitch_transcriptions(self, chunk_results: List[Dict], stitch_function: callable = None) -> str:
        """Stitch transcription results using intelligent or simple method"""
        if stitch_function and self.model_type == "nemo":
            # Use intelligent stitching for NeMo
            logger.info("Using intelligent stitching for NeMo transcription")
            stitched_result = stitch_function(chunk_results)
            return stitched_result.get("text", "")
        else:
            # Use simple stitching for Whisper or fallback
            logger.info(f"Using simple stitching for {self.model_type} transcription")
            valid_chunks = [r for r in chunk_results if r.get("text") and not r.get("error")]
            
            if not valid_chunks:
                logger.warning("No valid transcriptions to stitch")
                return ""
            
            # Simple stitching with better cleanup
            texts = [chunk["text"] for chunk in valid_chunks]
            stitched_text = " ".join(texts)
            
            # Enhanced cleanup for Whisper
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
            
            # Use model-specific chunking and processing
            if self.model_type == "nemo":
                result = await self._transcribe_with_nemo(audio, duration, start_time)
            else:
                result = await self._transcribe_with_whisper(audio, duration, start_time)
            
            logger.info(f"Transcription completed with {self.model_type}: duration={duration:.2f}s, "
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
            audio_chunks, stitch_function = self.chunk_audio_intelligent(audio, 16000)
            
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
    
    async def _transcribe_with_whisper(self, audio: np.ndarray, duration: float, start_time: float) -> TranscriptionResult:
        """Transcribe using Whisper model with optimal chunking"""
        total_duration = len(audio) / 16000
        
        if total_duration <= 30:  # Single chunk for short audio
            chunk_results = self.transcribe_batch_whisper([audio])
            final_text = chunk_results[0].get("text", "") if chunk_results else ""
            chunks_count = 1
        else:
            # Use intelligent chunking for longer audio
            audio_chunks, stitch_function = self.chunk_audio_intelligent(audio, 16000)
            
            # Convert AudioChunk objects to numpy arrays for Whisper
            audio_arrays = [chunk.audio_data for chunk in audio_chunks]
            
            # Transcribe chunks using Whisper
            chunk_results = self.transcribe_batch_whisper(audio_arrays)
            
            # Use simple stitching for Whisper
            final_text = self.stitch_transcriptions(chunk_results)
            chunks_count = len(audio_chunks)
        
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
            model_used="whisper",
            chunks_processed=chunks_count
        )
    
    async def _transcribe_with_simple_chunking(self, audio: np.ndarray, duration: float, start_time: float) -> TranscriptionResult:
        """Fallback transcription with simple chunking"""
        logger.info("Using fallback simple chunking")
        
        # Use simple chunking as fallback
        audio_chunks = self.chunk_audio_simple(audio)
        
        # Transcribe based on model type
        if self.model_type == "nemo":
            # Convert to AudioChunk objects for NeMo
            audio_chunk_objects = []
            for i, chunk in enumerate(audio_chunks):
                chunk_duration = len(chunk) / 16000
                audio_chunk_objects.append(AudioChunk(
                    audio_data=chunk,
                    start_time=i * (chunk_duration - 5),  # Approximate
                    end_time=(i + 1) * (chunk_duration - 5),
                    duration=chunk_duration,
                    chunk_id=i
                ))
            chunk_results = self.transcribe_batch_nemo(audio_chunk_objects)
        else:
            chunk_results = self.transcribe_batch_whisper(audio_chunks)
        
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
            "nemo_available": NEMO_AVAILABLE,
            "chunking_strategy": "intelligent" if self.model_type == "nemo" else "optimal",
            "h100_optimizations": "H100" in torch.cuda.get_device_name(0) if torch.cuda.is_available() else False,
            **gpu_info
        }