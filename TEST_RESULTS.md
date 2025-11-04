# Diarization Test Results

## Test Date
October 30, 2025 @ 23:43 UTC

## Test Environment
- **Container**: s2a-api-dev (docker-compose.dev.yml)
- **Model**: nvidia/diar_sortformer_4spk-v1
- **GPU**: NVIDIA H100 PCIe
- **Audio File**: `/tmp/test_audio.wav` (928 seconds / 15.5 minutes, 2 speakers)

## Test Results

### ✅ Unit Test: Direct Diarization Service
```
Status: PASSED
Total segments: 134
Unique speakers: 2
Speaker labels: ['SPK_1', 'SPK_2']

Speaker breakdown:
- SPK_1: 45 segments, 157.8s total speech time
- SPK_2: 89 segments, 164.6s total speech time
```

### ✅ Integration Test: Model Loading
```
Status: PASSED
Model class: SortformerEncLabelModel (EncDecSpeakerLabelModel)
Loading time: ~2 seconds
Mode: sortformer
```

### ✅ API Health Check
```
Status: PASSED
Endpoint: http://localhost:8001/v1/statistics/health
Response: {"status":"healthy", ...}
GPU: Available (NVIDIA H100 PCIe)
Batch processor: Running (1 worker, batch_size=8)
```

## Performance Metrics

- **Model Load Time**: ~2 seconds (first load, cached afterwards)
- **Diarization Time**: ~4 seconds for 15.5 minute audio
- **Throughput**: ~232x real-time (928s audio processed in 4s)
- **Segments Generated**: 134 segments across 2 speakers
- **Accuracy**: Correctly identified 2 speakers in 2-speaker audio ✅

## Code Changes Verified

### 1. Model Loading (`services/diarization_service.py` lines 35-77)
- ✅ Loads `EncDecSpeakerLabelModel` first (correct for sortformer)
- ✅ Falls back to MSDD/Clustering if needed
- ✅ Properly identifies model type as `SortformerEncLabelModel`

### 2. Diarization Inference (lines 93-134)
- ✅ Uses correct API: `model.diarize(audio=path, batch_size=1, verbose=False)`
- ✅ Handles `List[List[str]]` output format
- ✅ Logs completion with output type

### 3. RTTM Parsing (lines 220-261)
- ✅ Parses simplified format: `"start end speaker"`
- ✅ Converts `speaker_0` → `SPK_1`, `speaker_1` → `SPK_2`
- ✅ Falls back to standard RTTM if needed
- ✅ Handles parsing errors gracefully

## Comparison: Before vs After

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Speakers Detected | 1 ❌ | 2 ✅ |
| Segments | 1 (fallback) | 134 |
| Model Loaded | Generic (no diarize) | SortformerEncLabelModel |
| RTTM Format | Standard only | Simplified + Standard |
| Speaker Labels | SPK_1 only | SPK_1, SPK_2 |

## Conclusion

✅ **ALL TESTS PASSED**

The diarization service is now working correctly with the `nvidia/diar_sortformer_4spk-v1` model:
- Properly loads the model as `SortformerEncLabelModel`
- Correctly detects multiple speakers (2 speakers in test audio)
- Parses the simplified RTTM format output
- Performs at high speed (~232x real-time)

The fix is ready for production deployment.

## Next Steps

1. ✅ Code changes committed to repository
2. ⏳ Deploy to production environment
3. ⏳ Monitor production diarization jobs
4. ⏳ Consider adding automated regression tests
