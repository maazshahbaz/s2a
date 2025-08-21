import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import tempfile
import os
from pydub import AudioSegment
from loguru import logger
import webrtcvad
from scipy import signal
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class AudioProcessor:
    def __init__(self, 
                 target_sr: int = 16000,
                 vad_aggressiveness: int = 3):
        self.target_sr = target_sr
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
    def convert_to_wav(self, input_path: Union[str, Path], 
                      output_path: Optional[Union[str, Path]] = None) -> Path:
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = input_path.with_suffix('.wav')
        else:
            output_path = Path(output_path)
        
        if input_path.suffix.lower() == '.wav':
            return input_path
        
        try:
            # Use pydub for format conversion
            audio = AudioSegment.from_file(str(input_path))
            
            # Convert to mono and target sample rate
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(self.target_sr)
            
            # Export as WAV
            audio.export(str(output_path), format="wav")
            
            logger.info(f"Converted {input_path} to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            raise
    
    def normalize_audio(self, audio: np.ndarray, method: str = "peak") -> np.ndarray:
        if method == "peak":
            # Peak normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
        elif method == "rms":
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio / rms * 0.1  # Target RMS level
        
        return audio
    
    def apply_preemphasis(self, audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    
    def remove_silence(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
        # Use librosa for silence removal
        intervals = librosa.effects.split(audio, 
                                         top_db=20, 
                                         frame_length=2048, 
                                         hop_length=512)
        
        if len(intervals) == 0:
            return audio, 1.0
        
        # Concatenate non-silent intervals
        trimmed_audio = np.concatenate([audio[start:end] for start, end in intervals])
        
        # Calculate compression ratio
        compression_ratio = len(trimmed_audio) / len(audio)
        
        logger.debug(f"Silence removal: {compression_ratio:.2f} compression ratio")
        return trimmed_audio, compression_ratio
    
    def detect_voice_activity(self, audio: np.ndarray, sr: int, 
                            frame_duration_ms: int = 30) -> np.ndarray:
        # Convert audio to 16-bit PCM for WebRTC VAD
        audio_int16 = (audio * 32767).astype(np.int16)
        
        frame_length = int(sr * frame_duration_ms / 1000)
        frames = []
        
        for i in range(0, len(audio_int16) - frame_length + 1, frame_length):
            frame = audio_int16[i:i + frame_length]
            
            # WebRTC VAD requires specific sample rates
            if sr in [8000, 16000, 32000, 48000]:
                is_speech = self.vad.is_speech(frame.tobytes(), sr)
            else:
                # Fallback to energy-based VAD
                is_speech = np.mean(frame**2) > 0.01
            
            frames.append(is_speech)
        
        return np.array(frames)
    
    def apply_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Simple spectral subtraction noise reduction
        # Estimate noise from first 0.5 seconds
        noise_duration = min(int(0.5 * sr), len(audio) // 4)
        noise_sample = audio[:noise_duration]
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise spectrum
        noise_stft = librosa.stft(noise_sample, n_fft=2048, hop_length=512)
        noise_magnitude = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        
        # Spectral subtraction
        alpha = 2.0  # Over-subtraction factor
        enhanced_magnitude = magnitude - alpha * noise_magnitude
        
        # Ensure non-negative values
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
        
        return enhanced_audio
    
    def apply_band_pass_filter(self, audio: np.ndarray, sr: int, 
                              low_freq: float = 80, high_freq: float = 8000) -> np.ndarray:
        # Design Butterworth band-pass filter
        nyquist = sr / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def enhance_audio(self, audio: np.ndarray, sr: int, 
                     apply_noise_reduction: bool = True,
                     apply_filtering: bool = True,
                     remove_silence: bool = False) -> Tuple[np.ndarray, Dict]:
        processing_info = {}
        original_length = len(audio)
        
        # Apply preemphasis
        audio = self.apply_preemphasis(audio)
        
        # Apply band-pass filter to remove low/high frequency noise
        if apply_filtering:
            audio = self.apply_band_pass_filter(audio, sr)
            processing_info['filtered'] = True
        
        # Apply noise reduction
        if apply_noise_reduction:
            audio = self.apply_noise_reduction(audio, sr)
            processing_info['noise_reduced'] = True
        
        # Remove silence
        if remove_silence:
            audio, compression_ratio = self.remove_silence(audio, sr)
            processing_info['silence_removed'] = True
            processing_info['compression_ratio'] = compression_ratio
        
        # Normalize
        audio = self.normalize_audio(audio, method="peak")
        
        processing_info['original_length'] = original_length
        processing_info['final_length'] = len(audio)
        processing_info['duration_change'] = (len(audio) - original_length) / sr
        
        return audio, processing_info
    
    def validate_audio_quality(self, audio: np.ndarray, sr: int) -> Dict:
        # Calculate audio quality metrics
        metrics = {}
        
        # Signal-to-noise ratio estimation
        # Use voice activity detection to separate speech from silence
        vad_frames = self.detect_voice_activity(audio, sr)
        frame_length = int(sr * 30 / 1000)  # 30ms frames
        
        speech_segments = []
        silence_segments = []
        
        for i, is_speech in enumerate(vad_frames):
            start = i * frame_length
            end = min((i + 1) * frame_length, len(audio))
            segment = audio[start:end]
            
            if is_speech:
                speech_segments.extend(segment)
            else:
                silence_segments.extend(segment)
        
        if speech_segments and silence_segments:
            speech_power = np.mean(np.array(speech_segments)**2)
            noise_power = np.mean(np.array(silence_segments)**2)
            
            if noise_power > 0:
                snr_db = 10 * np.log10(speech_power / noise_power)
                metrics['snr_db'] = snr_db
            else:
                metrics['snr_db'] = float('inf')
        
        # Dynamic range
        metrics['dynamic_range_db'] = 20 * np.log10(np.max(np.abs(audio)) / 
                                                   (np.mean(np.abs(audio)) + 1e-10))
        
        # Zero crossing rate (indicates speech vs noise)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        metrics['mean_zcr'] = np.mean(zcr)
        
        # RMS energy
        metrics['rms_energy'] = np.sqrt(np.mean(audio**2))
        
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        metrics['spectral_centroid_hz'] = np.mean(spectral_centroids)
        
        # Voice activity ratio
        metrics['voice_activity_ratio'] = np.mean(vad_frames) if len(vad_frames) > 0 else 0
        
        return metrics
    
    def process_audio_file(self, input_path: Union[str, Path], 
                          enhance: bool = True,
                          validate: bool = True) -> Tuple[np.ndarray, int, Dict]:
        input_path = Path(input_path)
        
        try:
            # Convert to WAV if necessary
            if input_path.suffix.lower() != '.wav':
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    wav_path = self.convert_to_wav(input_path, tmp_file.name)
            else:
                wav_path = input_path
            
            # Load audio
            audio, sr = librosa.load(wav_path, sr=self.target_sr, mono=True)
            
            info = {
                'original_format': input_path.suffix,
                'sample_rate': sr,
                'duration': len(audio) / sr,
                'channels': 1
            }
            
            # Enhance audio if requested
            if enhance:
                audio, enhancement_info = self.enhance_audio(audio, sr)
                info.update(enhancement_info)
            
            # Validate audio quality if requested
            if validate:
                quality_metrics = self.validate_audio_quality(audio, sr)
                info['quality_metrics'] = quality_metrics
            
            # Clean up temporary file
            if wav_path != input_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            
            logger.info(f"Processed audio: duration={info['duration']:.2f}s, "
                       f"SNR={info.get('quality_metrics', {}).get('snr_db', 'N/A')}")
            
            return audio, sr, info
            
        except Exception as e:
            logger.error(f"Error processing audio file {input_path}: {e}")
            raise