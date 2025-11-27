from typing import List, Dict

class GlobalDiarizationManager:
    """
    Manages GLOBAL diarization - runs diarization on complete audio ONCE,
    then provides speaker labels for any timestamp in the audio.
    
    This solves the problem of inconsistent speaker IDs when diarizing chunks separately.
    """
    
    def __init__(self):
        self.global_segments: List[Dict] = []
        self.speaker_list: List[str] = []
        self.audio_duration: float = 0.0
        self.is_initialized: bool = False
    
    def set_global_diarization(self, diarization_result: Dict, audio_duration: float = None):
        """
        Store the global diarization result from complete audio.
        
        Args:
            diarization_result: Result from diarizing the COMPLETE audio file
            audio_duration: Total audio duration in seconds
        """
        if diarization_result.get('status') == 'error':
            print(f" Global diarization failed: {diarization_result.get('message', 'Unknown error')}")
            self.global_segments = []
            self.speaker_list = []
        else:
            self.global_segments = diarization_result.get('segments', [])
            # Sort by start time
            self.global_segments.sort(key=lambda x: x['start'])
            # Get unique speakers
            self.speaker_list = sorted(list(set(seg['speaker'] for seg in self.global_segments)))
        
        self.audio_duration = audio_duration or 0.0
        self.is_initialized = True
        
        return {
            'total_speakers': len(self.speaker_list),
            'speakers': self.speaker_list,
            'total_segments': len(self.global_segments)
        }
    
    def get_speaker_at_time(self, timestamp: float) -> str:
        """
        Get the speaker at a specific timestamp.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Speaker ID or 'unknown'
        """
        for seg in self.global_segments:
            if seg['start'] <= timestamp <= seg['end']:
                return seg['speaker']
        
        # If no exact match, find closest segment
        min_distance = float('inf')
        closest_speaker = 'unknown'
        
        for seg in self.global_segments:
            if timestamp < seg['start']:
                distance = seg['start'] - timestamp
            else:
                distance = timestamp - seg['end']
            
            if distance < min_distance:
                min_distance = distance
                closest_speaker = seg['speaker']
        
        return closest_speaker
    
    def get_segments_for_time_range(self, start_time: float, end_time: float) -> List[Dict]:
        """
        Get all speaker segments that overlap with a time range.
        
        Args:
            start_time: Start of range in seconds
            end_time: End of range in seconds
            
        Returns:
            List of segments (with times relative to the range start)
        """
        overlapping_segments = []
        
        for seg in self.global_segments:
            # Check if segment overlaps with range
            if seg['end'] > start_time and seg['start'] < end_time:
                # Clip segment to range
                clipped_start = max(seg['start'], start_time)
                clipped_end = min(seg['end'], end_time)
                
                overlapping_segments.append({
                    'speaker': seg['speaker'],
                    'start': clipped_start,  # Keep global times
                    'end': clipped_end,
                    'duration': clipped_end - clipped_start
                })
        
        return overlapping_segments
    
    def get_summary(self) -> Dict:
        """Get summary of global diarization"""
        if not self.is_initialized:
            return {'error': 'Not initialized'}
        
        # Calculate speaking time per speaker
        speaker_times = {}
        for seg in self.global_segments:
            spk = seg['speaker']
            duration = seg.get('duration', seg['end'] - seg['start'])
            speaker_times[spk] = speaker_times.get(spk, 0) + duration
        
        return {
            'total_speakers': len(self.speaker_list),
            'speakers': self.speaker_list,
            'total_segments': len(self.global_segments),
            'audio_duration': self.audio_duration,
            'speaker_times': speaker_times,
            'segments': self.global_segments
        }
