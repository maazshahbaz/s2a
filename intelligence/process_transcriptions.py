#!/usr/bin/env python3
"""
Process customer support transcriptions using Action Pipeline
"""

import json
import os
from typing import List, Dict, Any
from action_pipeline import ActionPipelineExtractor, UnifiedPayload


class TranscriptionProcessor:
    """Process transcription JSON files with the Action Pipeline"""
    
    def __init__(self, vllm_base_url: str = "http://localhost:8000/v1"):
        self.extractor = ActionPipelineExtractor(vllm_base_url)
        self.results = []
    
    def load_transcriptions(self, json_file_path: str) -> List[Dict[str, Any]]:
        """Load transcriptions from JSON file"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if it's a single object with transcriptions array or direct array
        if isinstance(data, dict) and 'transcriptions' in data:
            return data['transcriptions']
        elif isinstance(data, dict) and 'file_path' in data:
            # Single transcription object
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Unexpected JSON format")
    
    def process_transcription(self, transcription_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single transcription"""
        # Extract the actual transcription text
        transcript = transcription_data.get('transcription', '')
        file_name = transcription_data.get('file_name', 'unknown')
        
        print(f"Processing: {file_name}")
        
        # Use the Action Pipeline to extract structured data
        result = self.extractor.extract(transcript)
        
        # Add metadata
        result['metadata'] = {
            'file_name': file_name,
            'file_path': transcription_data.get('file_path', ''),
            'duration_seconds': transcription_data.get('duration_seconds', 0),
            'word_count': transcription_data.get('word_count', 0),
            'original_model': transcription_data.get('model_used', ''),
            'processed_at': transcription_data.get('processed_at', '')
        }
        
        return result
    
    def process_all(self, json_file_path: str, max_records: int = None) -> List[Dict[str, Any]]:
        """Process all transcriptions in the file"""
        transcriptions = self.load_transcriptions(json_file_path)
        
        if max_records:
            transcriptions = transcriptions[:max_records]
        
        print(f"Processing {len(transcriptions)} transcriptions...")
        
        results = []
        for i, transcription in enumerate(transcriptions, 1):
            try:
                result = self.process_transcription(transcription)
                results.append(result)
                
                # Print progress
                if i % 10 == 0 or i == len(transcriptions):
                    print(f"Processed {i}/{len(transcriptions)} transcriptions")
                    
            except Exception as e:
                print(f"Error processing transcription {i}: {e}")
                continue
        
        self.results = results
        return results
    
    def save_results(self, output_file: str):
        """Save processed results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    def print_summary(self):
        """Print processing summary and metrics"""
        if not self.results:
            print("No results to summarize")
            return
        
        successful = sum(1 for r in self.results if r.get('success', False))
        failed = len(self.results) - successful
        
        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Total transcriptions: {len(self.results)}")
        print(f"Successful extractions: {successful}")
        print(f"Failed extractions: {failed}")
        print(f"Success rate: {successful/len(self.results)*100:.1f}%")
        
        # Print extraction metrics
        metrics = self.extractor.get_metrics()
        print(f"\n=== EXTRACTION METRICS ===")
        print(f"Average latency: {metrics['avg_latency']:.2f}s")
        print(f"JSON pass rate: {metrics['json_pass_rate']*100:.1f}%")
        print(f"Retry count: {metrics['retry_count']}")
        
        print("\n=== FIELD HIT RATES ===")
        for field, rate in metrics['field_hit_rates'].items():
            print(f"{field}: {rate*100:.1f}%")
    
    def print_sample_outputs(self, num_samples: int = 3):
        """Print sample outputs to show what to expect"""
        if not self.results:
            print("No results to show")
            return
        
        print(f"\n=== SAMPLE OUTPUTS (first {num_samples}) ===")
        
        for i, result in enumerate(self.results[:num_samples], 1):
            print(f"\n--- Sample {i} ---")
            print(f"File: {result['metadata']['file_name']}")
            print(f"Success: {result['success']}")
            
            if result['success'] and result['data']:
                data = result['data']
                print(f"Summary: {data.get('summary', 'N/A')}")
                print(f"Intent: {data.get('intent', 'N/A')}")
                print(f"Sentiment: {data.get('sentiment', 'N/A')}")
                print(f"Action Items: {len(data.get('action_items', []))}")
                
                # Show entities if any
                entities = data.get('entities', {})
                if any(entities.values()):
                    print("Entities found:")
                    for entity_type, values in entities.items():
                        if values:
                            print(f"  {entity_type}: {values}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
    
    def close(self):
        """Clean up resources"""
        self.extractor.close()


def main():
    """Example usage"""
    # File path
    json_file = "/home/sj/work_space/bytepulse-ai/Language_Model/customer_support_transcriptions.json"
    
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
        return
    
    # Initialize processor
    processor = TranscriptionProcessor()
    
    try:
        # Process first 5 transcriptions as a test
        print("Processing first 5 transcriptions as a test...")
        results = processor.process_all(json_file, max_records=5)
        
        # Show what to expect
        processor.print_summary()
        processor.print_sample_outputs()
        
        # Save results
        processor.save_results("transcription_results_sample.json")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure vLLM server is running:")
        print("python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --max-model-len 16000 --trust-remote-code")
        
    finally:
        processor.close()


if __name__ == "__main__":
    main()