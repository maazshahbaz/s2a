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

@dataclass
class TranscriptionResult:
    text: str
    duration: float
    rtf: float
    chunks: List[Dict] = None
    confidence: float = None
    processing_time: float = None

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
        
        logger.info(f"Initializing ASR service with model: {model_name}")
        logger.info(f"Device: {device}, Batch size: {batch_size}")
        
        self._load_model()
        
    def _load_model(self):
        try:
            # Load the Parakeet model from HuggingFace
            # Note: Using Whisper as fallback since direct NeMo integration requires specific setup
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
            self.model.to(self.device)
            self.model.eval()
            
            # Warm up the model
            self._warmup_model()
            
            logger.info("ASR model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            raise
    
    def _warmup_model(self):
        logger.info("Warming up model...")
        dummy_audio = torch.randn(1, 16000 * 10).to(self.device)  # 10 seconds of dummy audio
        with torch.no_grad():
            features = self.processor(dummy_audio.cpu().numpy().squeeze(), 
                                    sampling_rate=16000, 
                                    return_tensors="pt").input_features.to(self.device)
            _ = self.model.generate(features, max_length=50)
        logger.info("Model warmup completed")
    
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
            
            logger.info(f"Preprocessed audio: duration={duration:.2f}s, sample_rate=16000Hz")
            return audio, duration, True
            
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
    
    def chunk_audio(self, audio: np.ndarray, sr: int = 16000) -> List[np.ndarray]:
        total_duration = len(audio) / sr
        
        if total_duration <= self.max_chunk_duration:
            return [audio]
        
        chunk_samples = int(self.max_chunk_duration * sr)
        chunks = []
        
        # Add overlap between chunks to avoid cutting words
        overlap_samples = int(2 * sr)  # 2 seconds overlap
        
        for start in range(0, len(audio), chunk_samples - overlap_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Skip chunks that are too short
            if len(chunk) >= int(self.min_audio_duration * sr):
                chunks.append(chunk)
        
        logger.info(f"Split {total_duration:.1f}s audio into {len(chunks)} chunks")
        return chunks
    
    def transcribe_batch(self, audio_chunks: List[np.ndarray]) -> List[Dict]:
        results = []
        
        for i in range(0, len(audio_chunks), self.batch_size):
            batch = audio_chunks[i:i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, audio_batch: List[np.ndarray]) -> List[Dict]:
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
                # Generate transcriptions
                generated_tokens = self.model.generate(
                    batch_features,
                    max_length=448,
                    num_beams=1,
                    do_sample=False,
                    temperature=0.0
                )
                
                # Decode transcriptions
                transcriptions = self.processor.batch_decode(generated_tokens, 
                                                           skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            # Process results
            for i, (audio, transcription) in enumerate(zip(audio_batch, transcriptions)):
                audio_duration = len(audio) / 16000
                rtf = processing_time / (audio_duration * len(audio_batch))  # Approximate RTF
                
                batch_results.append({
                    "text": transcription.strip(),
                    "duration": audio_duration,
                    "rtf": rtf,
                    "processing_time": processing_time / len(audio_batch),
                    "chunk_index": i
                })
            
            logger.info(f"Processed batch of {len(audio_batch)} chunks, RTF: {rtf:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
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
    
    def stitch_transcriptions(self, chunk_results: List[Dict]) -> str:
        valid_chunks = [r for r in chunk_results if r.get("text") and not r.get("error")]
        
        if not valid_chunks:
            logger.warning("No valid transcriptions to stitch")
            return ""
        
        # Simple stitching - could be enhanced with sentence boundary detection
        texts = [chunk["text"] for chunk in valid_chunks]
        stitched_text = " ".join(texts)
        
        # Basic cleanup
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
                    processing_time=time.time() - start_time
                )
            
            # Chunk audio if necessary
            audio_chunks = self.chunk_audio(audio)
            
            # Transcribe chunks in batches
            chunk_results = self.transcribe_batch(audio_chunks)
            
            # Stitch results
            final_text = self.stitch_transcriptions(chunk_results)
            
            processing_time = time.time() - start_time
            rtf = processing_time / duration if duration > 0 else float('inf')
            
            # Calculate average confidence if available
            valid_chunks = [r for r in chunk_results if not r.get("error")]
            avg_confidence = None
            if valid_chunks and all("confidence" in r for r in valid_chunks):
                avg_confidence = sum(r["confidence"] for r in valid_chunks) / len(valid_chunks)
            
            result = TranscriptionResult(
                text=final_text,
                duration=duration,
                rtf=rtf,
                chunks=chunk_results,
                confidence=avg_confidence,
                processing_time=processing_time
            )
            
            logger.info(f"Transcription completed: duration={duration:.2f}s, "
                       f"RTF={rtf:.3f}, chunks={len(audio_chunks)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio {audio_path}: {e}")
            return TranscriptionResult(
                text="",
                duration=0,
                rtf=float('inf'),
                processing_time=time.time() - start_time
            )
    
    def get_model_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_chunk_duration": self.max_chunk_duration,
            "min_audio_duration": self.min_audio_duration,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }