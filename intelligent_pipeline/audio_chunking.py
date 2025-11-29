import os
import webrtcvad
from pydub import AudioSegment
import asyncio

class AudioChunking:
    def __init__(self):
        self.target_chunk_ms = 5 * 60 * 1000  # 5 minutes
        self.allowed_drift_ms = 20 * 1000     # allow ±20 sec drift around 5 min
        self.silence_padding_ms = 200
        self.vad_aggressiveness = 2
        self.frame_ms = 30
        self.aggressiveness = 2
        self.temp_dir = "/tmp/s2a"
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def __frame_generator(self, audio):
        """Yield frames of audio (bytes) for VAD."""
        frame_len = int(audio.frame_rate * self.frame_ms / 1000) * 2  # 16-bit audio
        pcm = audio.raw_data

        for i in range(0, len(pcm), frame_len):
            yield pcm[i:i+frame_len]
            
    def __vad_activity(self, audio):
        """Return list of speech activity decisions per frame."""
        vad = webrtcvad.Vad(self.aggressiveness)
        frames = list(self.__frame_generator(audio))

        decisions = []
        for f in frames:
            if len(f) < (audio.frame_rate * self.frame_ms / 1000) * 2:
                decisions.append(False)
                continue
            decisions.append(vad.is_speech(f, audio.frame_rate))
        
        return decisions
    
    def __split_into_chunks_webrtcvad(self, audio_path):
        """
        Produces ~5-min chunks, cutting only at natural silence detected by WebRTC VAD.
        Returns chunks and their start times in the original audio.
        """
        audio = AudioSegment.from_file(audio_path)
        decisions = self.__vad_activity(audio)
        frame_len_ms = self.frame_ms
        
        # Create list of silence positions (frame indexes)
        silence_frames = [i for i, d in enumerate(decisions) if not d]
        silence_times = [f * frame_len_ms for f in silence_frames]  # in ms

        chunks = []
        chunk_timings = []  # Store (start_ms, end_ms) for each chunk
        chunk_start = 0
        total_ms = len(audio)

        while chunk_start < total_ms:
            target_end = chunk_start + self.target_chunk_ms

            if target_end >= total_ms:
                chunks.append(audio[chunk_start:])
                chunk_timings.append((chunk_start, total_ms))
                break

            # Search window around target (±20 seconds)
            window_start = max(0, target_end - self.allowed_drift_ms)
            window_end = min(total_ms, target_end + self.allowed_drift_ms)

            # Find silence points inside the search window
            candidates = [t for t in silence_times if window_start <= t <= window_end]

            if candidates:
                cut_point = min(candidates, key=lambda t: abs(t - target_end))
            else:
                cut_point = target_end  # fallback: no silence

            chunk_end = cut_point + self.silence_padding_ms
            chunk = audio[chunk_start:chunk_end]
            chunks.append(chunk)
            chunk_timings.append((chunk_start, min(chunk_end, total_ms)))
            chunk_start = cut_point

        return chunks, chunk_timings
    
    async def __export_chunk_async(self, chunk, filename):
        """Export a single chunk asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, chunk.export, filename, "wav")
        return filename
    
    async def __export_chunks_async(self, chunks, path):
        """Export all chunks asynchronously in parallel."""
        tasks = []
        for i, chunk in enumerate(chunks):
            filename = f"{path}_{i+1}.wav"
            tasks.append(self.__export_chunk_async(chunk, filename))
        
        final_paths = await asyncio.gather(*tasks)
        return final_paths
    
    async def create_chunks_async(self, audio_path):
        """Async version of create_chunks with timing information."""
        loop = asyncio.get_event_loop()
        
        # Run the CPU-intensive splitting in a thread pool
        audio_chunks, chunk_timings = await loop.run_in_executor(
            None, self.__split_into_chunks_webrtcvad, audio_path
        )
        
        audio_file_name = os.path.basename(audio_path).split(".")[0]
        temp_audio_path = os.path.join(self.temp_dir, audio_file_name)
        
        # Export chunks asynchronously in parallel
        file_list = await self.__export_chunks_async(audio_chunks, temp_audio_path)
        
        # Convert timings from ms to seconds
        chunk_timings_sec = [(start/1000, end/1000) for start, end in chunk_timings]
        
        return file_list, chunk_timings_sec

