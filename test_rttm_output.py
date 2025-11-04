#!/usr/bin/env python3
"""Test to see raw RTTM output from diarization model"""

import nemo.collections.asr as nemo_asr

print("Loading model...")
model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained('nvidia/diar_sortformer_4spk-v1')

print("\nRunning diarization...")
diar_output = model.diarize(audio='/tmp/test_audio.wav', batch_size=1, verbose=False)

print(f"\nOutput type: {type(diar_output)}")
print(f"Output length: {len(diar_output)}")

if diar_output and len(diar_output) > 0:
    print(f"\nFirst element type: {type(diar_output[0])}")
    print(f"First element length: {len(diar_output[0])}")
    
    # Show first 10 RTTM lines
    print("\nFirst 10 RTTM lines:")
    for i, line in enumerate(diar_output[0][:10]):
        print(f"  {i+1}: {line}")
    
    # Show last 5 RTTM lines
    print("\nLast 5 RTTM lines:")
    for i, line in enumerate(diar_output[0][-5:]):
        print(f"  {len(diar_output[0])-5+i+1}: {line}")
    
    # Count unique speakers
    speakers = set()
    for line in diar_output[0]:
        if ' SPEAKER ' in line:
            parts = line.split()
            if len(parts) >= 8:
                speaker = parts[7]
                speakers.add(speaker)
    
    print(f"\nUnique speakers in RTTM: {sorted(speakers)}")
    print(f"Number of unique speakers: {len(speakers)}")
