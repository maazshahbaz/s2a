import numpy as np

from api.streaming_security import is_ip_allowed, parse_allowed_networks
from intelligent_pipeline.streaming_asr_client import AsyncStreamingASRClient
from intelligent_pipeline.streaming_diar_client import AsyncStreamingDiarClient
from intelligent_pipeline.streaming_merger import StreamingMerger


def test_streaming_merger_speaker_switch_has_valid_segment_end_times():
    merger = StreamingMerger()
    asr_result = {
        "text": "a b c d e",
        "word_timestamps": [
            {"text": "a", "start": 0.0, "end": 0.2},
            {"text": "b", "start": 0.2, "end": 0.4},
            {"text": "c", "start": 0.4, "end": 0.6},
            {"text": "d", "start": 0.6, "end": 0.8},
            {"text": "e", "start": 0.8, "end": 1.0},
        ],
    }
    diar_result = {
        "segments": [
            {"speaker": 0, "start": 0.0, "end": 0.45},
            {"speaker": 1, "start": 0.45, "end": 0.75},
            {"speaker": 0, "start": 0.75, "end": 1.0},
        ],
        "num_speakers": 2,
    }

    segments = merger.merge(asr_result, diar_result, 0.0)

    assert len(segments) == 3
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] > segments[0]["start"]
    assert segments[1]["end"] > segments[1]["start"]
    assert segments[2]["end"] > segments[2]["start"]


def test_streaming_merger_reuses_last_speaker_without_diarization():
    merger = StreamingMerger()

    first_asr = {
        "text": "hello there",
        "word_timestamps": [
            {"text": "hello", "start": 0.0, "end": 0.3},
            {"text": "there", "start": 0.3, "end": 0.6},
        ],
    }
    first_diar = {"segments": [{"speaker": 1, "start": 0.0, "end": 1.0}], "num_speakers": 2}
    merger.merge(first_asr, first_diar, 0.0)

    second_asr = {
        "text": "follow up",
        "word_timestamps": [
            {"text": "follow", "start": 0.0, "end": 0.2},
            {"text": "up", "start": 0.2, "end": 0.4},
        ],
    }
    second_diar = {"segments": [], "num_speakers": 0}
    segments = merger.merge(second_asr, second_diar, 1.0)

    assert len(segments) == 1
    assert segments[0]["speaker"] == "Speaker 2"


def test_parse_allowed_networks_and_ip_matching():
    networks = parse_allowed_networks("10.0.0.0/8, 192.168.1.10, bad-entry")

    assert len(networks) == 2
    assert is_ip_allowed("10.4.2.9", networks)
    assert is_ip_allowed("192.168.1.10", networks)
    assert not is_ip_allowed("172.16.0.1", networks)


def test_asr_decode_json_output_accepts_nested_array():
    raw = np.array([[b'{"text":"hello","word_timestamps":[]}']], dtype=object)
    result = AsyncStreamingASRClient._decode_json_output(raw)
    assert result["text"] == "hello"
    assert result["word_timestamps"] == []


def test_diar_decode_json_output_accepts_flat_array():
    raw = np.array([b'{"segments":[],"num_speakers":0}'], dtype=object)
    result = AsyncStreamingDiarClient._decode_json_output(raw)
    assert result["segments"] == []
    assert result["num_speakers"] == 0
