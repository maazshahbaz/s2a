import asyncio
from typing import Dict, Optional
from loguru import logger
from .streaming_session import StreamingSession


class SessionManager:
    """
    Manages all active streaming sessions.
    Thread-safe registry for concurrent call sessions.
    """

    def __init__(self, max_concurrent: int = 50, chunk_duration: float = 1.0):
        self._sessions: Dict[str, StreamingSession] = {}
        self._lock = asyncio.Lock()
        self.max_concurrent = max_concurrent
        self.chunk_duration = chunk_duration

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def create_session(
        self,
        session_id: str,
        call_metadata: Optional[Dict] = None,
        callback_url: Optional[str] = None,
        input_sample_rate: int = 8000,
    ) -> StreamingSession:
        """
        Create a new streaming session.

        Raises:
            ValueError: If session_id already exists
            RuntimeError: If max concurrent sessions reached
        """
        async with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session {session_id} already exists")

            if len(self._sessions) >= self.max_concurrent:
                raise RuntimeError(
                    f"Max concurrent sessions ({self.max_concurrent}) reached"
                )

            session = StreamingSession(
                session_id=session_id,
                call_metadata=call_metadata,
                callback_url=callback_url,
                input_sample_rate=input_sample_rate,
                chunk_duration=self.chunk_duration,
            )
            self._sessions[session_id] = session

            logger.info(
                f"[SessionManager] Created session {session_id} "
                f"({len(self._sessions)}/{self.max_concurrent} active)"
            )
            return session

    async def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def end_session(self, session_id: str) -> Optional[StreamingSession]:
        """
        Remove and return a session.
        Returns None if session doesn't exist.
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                logger.info(
                    f"[SessionManager] Ended session {session_id} "
                    f"(duration={session.duration:.1f}s, "
                    f"chunks={session.chunks_processed}, "
                    f"{len(self._sessions)}/{self.max_concurrent} active)"
                )
            return session

    async def get_all_sessions(self) -> Dict[str, StreamingSession]:
        """Get all active sessions (for monitoring)."""
        return dict(self._sessions)
