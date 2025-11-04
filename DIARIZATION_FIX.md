# Diarization Fix Summary

## Problem
The diarization service was returning only 1 speaker even for audio with multiple speakers.

## Root Cause
The `nvidia/diar_sortformer_4spk-v1` model was not being loaded correctly:

1. **Wrong model class**: The code tried to load it as `ClusteringDiarizer` or `NeuralDiarizer`, but this model is actually a `SortformerEncLabelModel` (subclass of `EncDecSpeakerLabelModel`).

2. **Wrong RTTM format**: The Sortformer model outputs a simplified RTTM format (`start end speaker`) instead of the standard full RTTM format, and the parser wasn't handling this.

## Solution

### Changes Made to `services/diarization_service.py`

1. **Model Loading** (lines 35-77):
   - Changed loading order to try `EncDecSpeakerLabelModel` FIRST
   - This correctly loads the `nvidia/diar_sortformer_4spk-v1` as `SortformerEncLabelModel`
   - Added mode `'sortformer'` to track this model type

2. **Diarization Inference** (lines 93-134):
   - Added specific handling for `sortformer` mode
   - Calls `model.diarize(audio=audio_path, batch_size=1, verbose=False)`
   - This returns `List[List[str]]` where inner lists contain simplified RTTM lines

3. **RTTM Parsing** (lines 220-261):
   - Added support for simplified format: `"start end speaker"`
   - Converts `speaker_0`, `speaker_1` to `SPK_1`, `SPK_2` for consistency
   - Falls back to standard RTTM parsing if needed

## Testing Results

### Before Fix
```
Unique speakers detected: 1
Speaker labels: ['SPK_1']
```

### After Fix
```
Unique speakers detected: 2
Speaker labels: ['SPK_1', 'SPK_2']
Found 134 segments
```

## Model Information

- **Model**: `nvidia/diar_sortformer_4spk-v1`
- **Class**: `SortformerEncLabelModel` (extends `EncDecSpeakerLabelModel`)
- **Max Speakers**: 4
- **Output Format**: List of simplified RTTM lines (`start end speaker`)

## Verification

Tested with:
- Audio file: `/home/sj/Desktop/data/back2/bytepulse-ai/uploads/2025-10-29/badcb4b0-f69f-4384-863e-09b09ce0582f.wav`
- Duration: ~928 seconds (15.5 minutes)
- Result: Successfully detected 2 speakers with 134 segments

## Deployment

The fix is now active in the development environment. To deploy to production:

1. Ensure the updated `services/diarization_service.py` is in the codebase
2. Rebuild the Docker image or restart the service
3. The model will automatically load correctly on first diarization request
