import subprocess
import json
from pathlib import Path
from typing import Optional

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# Example constant (you can import from your enums instead)
MIN_AUDIO_DURATION = 1.0  # seconds


def get_audio_duration(audio_path: str | Path) -> Optional[float]:
    """
    Get duration of an audio file in seconds.
    Tries librosa first, then falls back to ffprobe.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds, or None if it cannot be determined.

    Raises:
        ValueError: If the audio is shorter than MIN_AUDIO_DURATION.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    duration = None

    # --- Try librosa first ---
    if HAS_LIBROSA:
        try:
            duration = librosa.get_duration(filename=str(audio_path))
        except Exception:
            duration = None

    # --- Fallback: use ffprobe ---
    if duration is None:
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", str(audio_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration = float(data["format"]["duration"])
        except Exception:
            duration = None

    # --- Validation ---
    if duration is not None and duration < MIN_AUDIO_DURATION:
        raise ValueError(f"Audio too short: {duration:.1f}s (minimum {MIN_AUDIO_DURATION}s)")

    return duration
