#!/usr/bin/env python3
"""
ASR Dataset Analysis Script
Comprehensive analysis of audio dataset for ASR model evaluation preparation.
Analyzes audio quality, distribution, and characteristics before transcription.
"""

import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict
import librosa
import soundfile as sf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ASRDatasetAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.audio_files = list(self.dataset_path.glob("*.wav"))
        self.analysis_results = {}
        
    def basic_file_analysis(self):
        """Basic file count and size analysis"""
        print("=" * 60)
        print("BASIC DATASET OVERVIEW")
        print("=" * 60)
        
        total_files = len(self.audio_files)
        total_size = sum(f.stat().st_size for f in self.audio_files)
        
        print(f"Total audio files: {total_files:,}")
        print(f"Total dataset size: {total_size / (1024**3):.2f} GB")
        print(f"Average file size: {total_size / total_files / (1024**2):.2f} MB")
        
        self.analysis_results['basic'] = {
            'total_files': total_files,
            'total_size_gb': total_size / (1024**3),
            'avg_file_size_mb': total_size / total_files / (1024**2)
        }
        
    def audio_format_analysis(self):
        """Analyze audio format consistency"""
        print("\n" + "=" * 60)
        print("AUDIO FORMAT ANALYSIS")
        print("=" * 60)
        
        formats = defaultdict(int)
        sample_rates = defaultdict(int)
        channels = defaultdict(int)
        bit_depths = defaultdict(int)
        corrupted_files = []
        
        print("Analyzing audio formats... (sampling first 100 files)")
        
        # Sample first 100 files for format analysis
        sample_files = self.audio_files[:min(100, len(self.audio_files))]
        
        for audio_file in sample_files:
            try:
                with wave.open(str(audio_file), 'r') as wav:
                    sample_rate = wav.getframerate()
                    n_channels = wav.getnchannels()
                    sample_width = wav.getsampwidth()
                    
                    sample_rates[sample_rate] += 1
                    channels[n_channels] += 1
                    bit_depths[sample_width * 8] += 1
                    
            except Exception as e:
                corrupted_files.append(str(audio_file))
        
        print(f"Sample rates found:")
        for sr, count in sorted(sample_rates.items()):
            print(f"  {sr} Hz: {count} files ({count/len(sample_files)*100:.1f}%)")
            
        print(f"\nChannel configurations:")
        for ch, count in sorted(channels.items()):
            ch_type = "Mono" if ch == 1 else "Stereo" if ch == 2 else f"{ch}-channel"
            print(f"  {ch_type}: {count} files ({count/len(sample_files)*100:.1f}%)")
            
        print(f"\nBit depths:")
        for bd, count in sorted(bit_depths.items()):
            print(f"  {bd}-bit: {count} files ({count/len(sample_files)*100:.1f}%)")
            
        if corrupted_files:
            print(f"\nCorrupted/unreadable files found: {len(corrupted_files)}")
            
        self.analysis_results['format'] = {
            'sample_rates': dict(sample_rates),
            'channels': dict(channels),
            'bit_depths': dict(bit_depths),
            'corrupted_count': len(corrupted_files)
        }
        
    def duration_distribution_analysis(self):
        """Analyze duration distribution and create visualizations"""
        print("\n" + "=" * 60)
        print("DURATION DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        durations = []
        print("Extracting durations from all files...")
        
        for i, audio_file in enumerate(self.audio_files):
            if i % 500 == 0:
                print(f"  Progress: {i+1}/{len(self.audio_files)} files...")
                
            try:
                with wave.open(str(audio_file), 'r') as wav:
                    frames = wav.getnframes()
                    sample_rate = wav.getframerate()
                    duration = frames / sample_rate
                    durations.append(duration)
            except:
                continue
        
        durations = np.array(durations)
        
        # Statistics
        print(f"\nDuration Statistics:")
        print(f"  Total recordings: {len(durations):,}")
        print(f"  Mean duration: {np.mean(durations):.2f} seconds ({np.mean(durations)/60:.2f} minutes)")
        print(f"  Median duration: {np.median(durations):.2f} seconds ({np.median(durations)/60:.2f} minutes)")
        print(f"  Std deviation: {np.std(durations):.2f} seconds")
        print(f"  Min duration: {np.min(durations):.2f} seconds")
        print(f"  Max duration: {np.max(durations):.2f} seconds ({np.max(durations)/60:.2f} minutes)")
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\nDuration Percentiles:")
        for p in percentiles:
            val = np.percentile(durations, p)
            print(f"  {p:2d}th percentile: {val:6.2f}s ({val/60:5.2f}m)")
        
        # Identify outliers and very short/long files
        q1, q3 = np.percentile(durations, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = durations[(durations < lower_bound) | (durations > upper_bound)]
        very_short = durations[durations < 1.0]  # Less than 1 second
        very_long = durations[durations > 600]   # More than 10 minutes
        
        print(f"\nData Quality Indicators:")
        print(f"  Outliers (IQR method): {len(outliers):,} files ({len(outliers)/len(durations)*100:.2f}%)")
        print(f"  Very short (<1s): {len(very_short):,} files ({len(very_short)/len(durations)*100:.2f}%)")
        print(f"  Very long (>10m): {len(very_long):,} files ({len(very_long)/len(durations)*100:.2f}%)")
        
        # Create duration distribution plot
        plt.figure(figsize=(15, 10))
        
        # Histogram
        plt.subplot(2, 2, 1)
        plt.hist(durations, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Frequency')
        plt.title('Duration Distribution')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(durations)
        plt.ylabel('Duration (seconds)')
        plt.title('Duration Box Plot')
        plt.grid(True, alpha=0.3)
        
        # Log scale histogram
        plt.subplot(2, 2, 3)
        plt.hist(durations, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.title('Duration Distribution (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 4)
        sorted_durations = np.sort(durations)
        cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
        plt.plot(sorted_durations, cumulative)
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Duration Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/sj/work_space/ai/duration_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nDuration analysis plot saved to: duration_analysis.png")
        
        self.analysis_results['duration'] = {
            'mean': float(np.mean(durations)),
            'median': float(np.median(durations)),
            'std': float(np.std(durations)),
            'min': float(np.min(durations)),
            'max': float(np.max(durations)),
            'total_hours': float(np.sum(durations) / 3600),
            'outliers_count': len(outliers),
            'very_short_count': len(very_short),
            'very_long_count': len(very_long)
        }
        
    def audio_quality_analysis(self):
        """Analyze audio quality indicators"""
        print("\n" + "=" * 60)
        print("AUDIO QUALITY ANALYSIS")
        print("=" * 60)
        
        # Sample 50 files for detailed quality analysis
        sample_size = min(50, len(self.audio_files))
        sample_files = np.random.choice(self.audio_files, sample_size, replace=False)
        
        quality_metrics = {
            'clipping_detected': 0,
            'silence_ratio': [],
            'snr_estimates': [],
            'zero_crossings': [],
            'spectral_centroids': []
        }
        
        print(f"Analyzing audio quality on {sample_size} random samples...")
        
        for i, audio_file in enumerate(sample_files):
            try:
                # Load audio with librosa
                y, sr = librosa.load(str(audio_file), sr=None)
                
                # Clipping detection
                if np.any(np.abs(y) >= 0.99):
                    quality_metrics['clipping_detected'] += 1
                
                # Silence ratio (frames below -40dB)
                silence_threshold = 0.01  # -40dB approximately
                silence_frames = np.sum(np.abs(y) < silence_threshold)
                silence_ratio = silence_frames / len(y)
                quality_metrics['silence_ratio'].append(silence_ratio)
                
                # Simple SNR estimate (signal power vs noise floor)
                signal_power = np.mean(y**2)
                noise_floor = np.percentile(np.abs(y), 10)**2  # Bottom 10% as noise estimate
                if noise_floor > 0:
                    snr_db = 10 * np.log10(signal_power / noise_floor)
                    quality_metrics['snr_estimates'].append(snr_db)
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                quality_metrics['zero_crossings'].append(np.mean(zcr))
                
                # Spectral centroid
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                quality_metrics['spectral_centroids'].append(np.mean(spectral_centroids))
                
            except Exception as e:
                print(f"  Error analyzing {audio_file}: {e}")
                continue
        
        # Report quality metrics
        print(f"\nQuality Analysis Results:")
        print(f"  Files with clipping: {quality_metrics['clipping_detected']}/{sample_size} ({quality_metrics['clipping_detected']/sample_size*100:.1f}%)")
        
        if quality_metrics['silence_ratio']:
            silence_ratios = np.array(quality_metrics['silence_ratio'])
            print(f"  Average silence ratio: {np.mean(silence_ratios):.3f} ({np.mean(silence_ratios)*100:.1f}%)")
            print(f"  Silence ratio range: {np.min(silence_ratios):.3f} - {np.max(silence_ratios):.3f}")
        
        if quality_metrics['snr_estimates']:
            snr_values = np.array(quality_metrics['snr_estimates'])
            print(f"  Estimated SNR: {np.mean(snr_values):.1f} ± {np.std(snr_values):.1f} dB")
            print(f"  SNR range: {np.min(snr_values):.1f} - {np.max(snr_values):.1f} dB")
        
        if quality_metrics['zero_crossings']:
            zcr_values = np.array(quality_metrics['zero_crossings'])
            print(f"  Average zero crossing rate: {np.mean(zcr_values):.4f}")
        
        if quality_metrics['spectral_centroids']:
            sc_values = np.array(quality_metrics['spectral_centroids'])
            print(f"  Average spectral centroid: {np.mean(sc_values):.0f} Hz")
        
        self.analysis_results['quality'] = quality_metrics
        
    def asr_readiness_assessment(self):
        """Assess dataset readiness for ASR evaluation"""
        print("\n" + "=" * 60)
        print("ASR EVALUATION READINESS ASSESSMENT")
        print("=" * 60)
        
        recommendations = []
        warnings = []
        
        # Check duration distribution
        if 'duration' in self.analysis_results:
            dur = self.analysis_results['duration']
            
            if dur['very_short_count'] > len(self.audio_files) * 0.05:  # >5% very short
                warnings.append(f"High number of very short files ({dur['very_short_count']:,}) may affect WER calculation")
            
            if dur['very_long_count'] > 0:
                recommendations.append(f"Consider segmenting {dur['very_long_count']} very long files for better ASR processing")
            
            if dur['mean'] < 5:
                warnings.append("Short average duration may limit ASR model context")
            elif dur['mean'] > 30:
                recommendations.append("Long average duration - consider chunking for real-time factor evaluation")
        
        # Check format consistency
        if 'format' in self.analysis_results:
            fmt = self.analysis_results['format']
            
            if len(fmt['sample_rates']) > 1:
                warnings.append("Multiple sample rates detected - standardization recommended")
                recommendations.append("Resample all files to consistent sample rate (e.g., 16kHz for ASR)")
            
            if len(fmt['channels']) > 1:
                warnings.append("Mixed mono/stereo files - standardization recommended")
                recommendations.append("Convert all files to mono for ASR evaluation")
            
            if fmt['corrupted_count'] > 0:
                warnings.append(f"{fmt['corrupted_count']} corrupted files need to be removed or fixed")
        
        # Quality assessment
        if 'quality' in self.analysis_results:
            qual = self.analysis_results['quality']
            
            if qual['clipping_detected'] > 0:
                warnings.append(f"Clipping detected in {qual['clipping_detected']} sample files")
            
            if qual['silence_ratio'] and np.mean(qual['silence_ratio']) > 0.3:
                recommendations.append("High silence ratio - consider voice activity detection preprocessing")
        
        print("WARNINGS:")
        if warnings:
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        else:
            print("  No major issues detected!")
        
        print("\nRECOMMENDATIONS:")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("  Dataset appears ready for ASR evaluation!")
        
        # ASR-specific recommendations
        print("\nASR EVALUATION SPECIFIC RECOMMENDATIONS:")
        print("  1. Prepare ground truth transcriptions for WER/CER calculation")
        print("  2. Consider creating train/validation/test splits")
        print("  3. Document speaker demographics and recording conditions")
        print("  4. Plan for RTFx measurement infrastructure")
        print("  5. Consider phonetic diversity analysis for comprehensive evaluation")
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "=" * 60)
        print("DATASET SUMMARY REPORT")
        print("=" * 60)
        
        if 'basic' in self.analysis_results:
            basic = self.analysis_results['basic']
            print(f"Dataset Size: {basic['total_files']:,} files, {basic['total_size_gb']:.2f} GB")
        
        if 'duration' in self.analysis_results:
            dur = self.analysis_results['duration']
            print(f"Total Audio: {dur['total_hours']:.1f} hours")
            print(f"Duration Range: {dur['min']:.1f}s - {dur['max']:.1f}s (avg: {dur['mean']:.1f}s)")
        
        print(f"\nNext Steps:")
        print(f"  1. Address any warnings from readiness assessment")
        print(f"  2. Begin transcription process (manual or automated)")
        print(f"  3. Prepare ASR evaluation pipeline")
        print(f"  4. Set up WER/CER/RTFx measurement tools")
        
    def run_full_analysis(self):
        """Run complete dataset analysis"""
        print("Starting comprehensive ASR dataset analysis...")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Found {len(self.audio_files):,} audio files")
        
        self.basic_file_analysis()
        self.audio_format_analysis()
        self.duration_distribution_analysis()
        self.audio_quality_analysis()
        self.asr_readiness_assessment()
        self.generate_summary_report()
        
        print(f"\nAnalysis complete! Check duration_analysis.png for visualizations.")

if __name__ == "__main__":
    # Initialize analyzer
    dataset_path = "/home/sj/work_space/ai/csrRecords/csrRecords"
    analyzer = ASRDatasetAnalyzer(dataset_path)
    
    # Run full analysis
    analyzer.run_full_analysis()
