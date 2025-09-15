#!/usr/bin/env python3
"""
S2A Speech-to-Text CLI Tool

A command-line interface for the S2A ASR microservice that provides
high-performance speech transcription using NVIDIA NeMo models.
"""

import click
import asyncio
import json
import time
from pathlib import Path
from typing import Optional, List
import requests
from loguru import logger

from services.asr_service import NeMoASRService
from services.audio_utils import AudioProcessor
from config import get_settings

@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, log_level):
    """S2A Speech-to-Text CLI"""
    logger.remove()
    logger.add(lambda msg: click.echo(msg, err=True), level=log_level)
    ctx.ensure_object(dict)

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--model', default='nvidia/parakeet-tdt-0.6b-v2', help='Model name to use')
@click.option('--device', default='auto', help='Device to use (cuda/cpu/auto)')
@click.option('--batch-size', default=4, help='Batch size for processing')
@click.option('--enhance', is_flag=True, help='Apply audio enhancement')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', 'output_format', default='text', 
              type=click.Choice(['text', 'json', 'srt']), help='Output format')
def transcribe(audio_file, model, device, batch_size, enhance, output, output_format):
    """Transcribe an audio file"""
    
    async def _transcribe():
        # Initialize services
        if device == 'auto':
            import torch
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device_name = device
            
        click.echo(f"Initializing ASR service with {model} on {device_name}...")
        
        asr_service = NeMoASRService(
            model_name=model,
            device=device_name,
            batch_size=batch_size
        )
        
        audio_processor = AudioProcessor()
        
        # Process audio file
        click.echo(f"Processing audio file: {audio_file}")
        start_time = time.time()
        
        if enhance:
            click.echo("Applying audio enhancement...")
            audio, sr, info = audio_processor.process_audio_file(audio_file, enhance=True)
            click.echo(f"Audio quality: SNR={info.get('quality_metrics', {}).get('snr_db', 'N/A')} dB")
        
        # Transcribe
        result = await asr_service.transcribe_audio(audio_file)
        processing_time = time.time() - start_time
        
        # Format output
        if output_format == 'text':
            output_content = result.text
        elif output_format == 'json':
            output_content = json.dumps({
                'text': result.text,
                'duration': result.duration,
                'rtf': result.rtf,
                'processing_time': processing_time,
                'confidence': result.confidence,
                'chunks': len(result.chunks) if result.chunks else 1
            }, indent=2)
        elif output_format == 'srt':
            # Simple SRT format (would need word-level timestamps for proper SRT)
            output_content = f"1\n00:00:00,000 --> {format_timestamp(result.duration)}\n{result.text}\n"
        
        # Output results
        if output:
            Path(output).write_text(output_content)
            click.echo(f"Transcription saved to {output}")
        else:
            click.echo("\n--- Transcription ---")
            click.echo(output_content)
        
        # Show stats
        click.echo(f"\n--- Statistics ---")
        click.echo(f"Audio duration: {result.duration:.2f}s")
        click.echo(f"Processing time: {processing_time:.2f}s")
        click.echo(f"Real-time factor: {result.rtf:.3f}")
        click.echo(f"Throughput: {result.duration / processing_time:.1f}x real-time")
    
    asyncio.run(_transcribe())

@cli.command()
@click.argument('audio_files', nargs=-1, type=click.Path(exists=True))
@click.option('--model', default='nvidia/parakeet-tdt-0.6b-v2', help='Model name to use')
@click.option('--device', default='auto', help='Device to use (cuda/cpu/auto)')
@click.option('--batch-size', default=8, help='Batch size for processing')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--format', 'output_format', default='text',
              type=click.Choice(['text', 'json']), help='Output format')
@click.option('--enhance', is_flag=True, help='Apply audio enhancement')
def batch_transcribe(audio_files, model, device, batch_size, output_dir, output_format, enhance):
    """Transcribe multiple audio files in batch"""
    
    if not audio_files:
        click.echo("No audio files provided")
        return
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    async def _batch_transcribe():
        # Initialize services
        if device == 'auto':
            import torch
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device_name = device
        
        click.echo(f"Initializing ASR service for batch processing...")
        
        asr_service = NeMoASRService(
            model_name=model,
            device=device_name,
            batch_size=batch_size
        )
        
        total_duration = 0
        total_processing_time = 0
        
        with click.progressbar(audio_files, label='Transcribing files') as files:
            for audio_file in files:
                try:
                    start_time = time.time()
                    result = await asr_service.transcribe_audio(audio_file)
                    processing_time = time.time() - start_time
                    
                    total_duration += result.duration
                    total_processing_time += processing_time
                    
                    # Save output
                    if output_dir:
                        output_path = output_dir / f"{Path(audio_file).stem}.{output_format}"
                        
                        if output_format == 'text':
                            output_path.write_text(result.text)
                        elif output_format == 'json':
                            output_data = {
                                'file': str(audio_file),
                                'text': result.text,
                                'duration': result.duration,
                                'rtf': result.rtf,
                                'processing_time': processing_time,
                                'confidence': result.confidence
                            }
                            output_path.write_text(json.dumps(output_data, indent=2))
                    else:
                        click.echo(f"\n{audio_file}: {result.text}")
                        
                except Exception as e:
                    click.echo(f"Error processing {audio_file}: {e}", err=True)
        
        # Show summary
        avg_rtf = total_processing_time / total_duration if total_duration > 0 else 0
        click.echo(f"\n--- Batch Summary ---")
        click.echo(f"Files processed: {len(audio_files)}")
        click.echo(f"Total audio duration: {total_duration:.1f}s")
        click.echo(f"Total processing time: {total_processing_time:.1f}s")
        click.echo(f"Average RTF: {avg_rtf:.3f}")
    
    asyncio.run(_batch_transcribe())

@cli.command()
@click.option('--url', default='http://localhost:8000', help='API server URL')
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--enhance', is_flag=True, help='Apply audio enhancement')
@click.option('--async-mode', is_flag=True, help='Use async API')
def api_transcribe(url, audio_file, enhance, async_mode):
    """Transcribe using the API server"""
    
    endpoint = f"{url}/v1/transcription/transcribe" + ("/async" if async_mode else "")
    
    with open(audio_file, 'rb') as f:
        files = {'audio_file': f}
        data = {'enhance_audio': enhance}
        
        try:
            response = requests.post(endpoint, files=files, data=data, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            
            if async_mode:
                job_id = result['job_id']
                click.echo(f"Job submitted: {job_id}")
                
                # Poll for result
                status_url = f"{url}/v1/status/{job_id}"
                while True:
                    time.sleep(2)
                    status_response = requests.get(status_url)
                    status_data = status_response.json()
                    
                    if status_data['status'] == 'completed':
                        click.echo("Transcription completed!")
                        click.echo(status_data['result']['text'])
                        break
                    elif status_data['status'] == 'failed':
                        click.echo(f"Transcription failed: {status_data.get('error')}")
                        break
                    else:
                        click.echo(f"Status: {status_data['status']}")
            else:
                click.echo("Transcription completed!")
                click.echo(f"Text: {result['text']}")
                click.echo(f"Duration: {result['duration']:.2f}s")
                click.echo(f"RTF: {result['rtf']:.3f}")
                
        except requests.RequestException as e:
            click.echo(f"API request failed: {e}", err=True)

@cli.command()
@click.option('--url', default='http://localhost:8000', help='API server URL')
def status(url):
    """Check API server status"""
    
    try:
        # Health check
        health_response = requests.get(f"{url}/v1/statistics/health", timeout=10)
        health_response.raise_for_status()
        health_data = health_response.json()
        
        click.echo("=== Server Health ===")
        click.echo(f"Status: {health_data['status']}")
        click.echo(f"Uptime: {health_data['uptime']:.1f}s")
        click.echo(f"GPU Available: {health_data['gpu_available']}")
        
        # Model info
        model_info = health_data['model_info']
        click.echo(f"\n=== Model Info ===")
        click.echo(f"Model: {model_info['model_name']}")
        click.echo(f"Device: {model_info['device']}")
        click.echo(f"Batch Size: {model_info['batch_size']}")
        
        # Performance stats
        stats_response = requests.get(f"{url}/v1/statistics/stats", timeout=10)
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            batch_stats = stats_data['batch_processor']
            
            click.echo(f"\n=== Performance Stats ===")
            click.echo(f"Queue Size: {batch_stats['queue_size']}")
            click.echo(f"Active Jobs: {batch_stats['processing_jobs']}")
            click.echo(f"Jobs Processed: {batch_stats['jobs_processed']}")
            click.echo(f"Average Batch Size: {batch_stats['average_batch_size']:.1f}")
        
    except requests.RequestException as e:
        click.echo(f"Failed to get server status: {e}", err=True)

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

if __name__ == '__main__':
    cli()