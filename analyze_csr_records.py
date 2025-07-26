#!/usr/bin/env python3
"""
CSR Records Analysis Script
Analyzes audio files in the CSR records directory to provide:
- Total number of files
- Maximum, minimum, and average duration
"""

import os
import wave
import statistics
from pathlib import Path

def get_wav_duration(file_path):
    """Get duration of a WAV file in seconds"""
    try:
        with wave.open(str(file_path), 'r') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / sample_rate
            return duration
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def analyze_csr_records(directory_path):
    """Analyze CSR records directory"""
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist!")
        return
    
    # Find all WAV files
    wav_files = list(directory.glob("*.wav"))
    
    print(f"CSR Records Analysis")
    print(f"=" * 50)
    print(f"Directory: {directory_path}")
    print(f"Total number of files: {len(wav_files)}")
    
    if not wav_files:
        print("No WAV files found in the directory!")
        return
    
    # Get durations for all files
    durations = []
    processed_files = 0
    
    print(f"\nAnalyzing file durations...")
    
    for i, wav_file in enumerate(wav_files):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing file {i+1}/{len(wav_files)}...")
        
        duration = get_wav_duration(wav_file)
        if duration is not None:
            durations.append(duration)
            processed_files += 1
    
    if not durations:
        print("Could not read duration from any files!")
        return
    
    # Calculate statistics
    min_duration = min(durations)
    max_duration = max(durations)
    avg_duration = statistics.mean(durations)
    
    # Convert to minutes and seconds for better readability
    def format_duration(seconds):
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s ({seconds:.2f}s)"
    
    print(f"\nDuration Analysis Results:")
    print(f"=" * 50)
    print(f"Files successfully processed: {processed_files}")
    print(f"Files with errors: {len(wav_files) - processed_files}")
    print(f"")
    print(f"Minimum duration: {format_duration(min_duration)}")
    print(f"Maximum duration: {format_duration(max_duration)}")
    print(f"Average duration: {format_duration(avg_duration)}")
    
    # Additional statistics
    median_duration = statistics.median(durations)
    total_duration = sum(durations)
    
    print(f"Median duration: {format_duration(median_duration)}")
    print(f"Total duration: {format_duration(total_duration)}")
    print(f"Total duration in hours: {total_duration / 3600:.2f} hours")

if __name__ == "__main__":
    # CSR records directory path
    csr_directory = "/home/sj/work_space/ai/csrRecords/csrRecords"
    
    analyze_csr_records(csr_directory)
