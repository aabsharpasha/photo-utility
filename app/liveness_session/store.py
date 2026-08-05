"""Session/verdict store: in-memory default, Redis optional via REDIS_URL."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Session:
    id: str
    token_sha256: str
    api_key_sha256: str  # bound to the caller identity that created it
    challenge: str  # "blink" | "turn_left" | "turn_right"
    required_blinks: int
    created_at: float
    expires_at: float
    status: str = "pending"  # pending | streaming | done
    live: Optional[bool] = None
    reasons: list[str] = field(default_factory=list)
    verdict_at: Optional[float] = None

    def expired(self, now: float | None = None) -> bool:
        return (now or time.time()) > self.expires_at


class SessionStore:
    """Abstract store interface."""

    def put(self, session: Session, ttl_seconds: int) -> None:
        raise NotImplementedError

    def get(self, session_id: str) -> Optional[Session]:
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
    """Thread-safe in-memory store with per-entry TTL (single-process default)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, tuple[Session, float]] = {}

    def put(self, session: Session, ttl_seconds: int) -> None:
        with self._lock:
            self._items[session.id] = (session, time.time() + ttl_seconds)
            self._purge_locked()

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            self._purge_locked()
            entry = self._items.get(session_id)
            return entry[0] if entry else None

    def _purge_locked(self) -> None:
        now = time.time()
        stale = [k for k, (_, exp) in self._items.items() if exp < now]
        for k in stale:
            del self._items[k]


class RedisSessionStore(SessionStore):
    """Redis-backed store (used when REDIS_URL is set and redis-py is installed)."""

    _PREFIX = "liveness_session:"

    def __init__(self, url: str) -> None:
        import redis  # optional dependency; import deferred on purpose

        self._client = redis.Redis.from_url(url, decode_responses=True)

    def put(self, session: Session, ttl_seconds: int) -> None:
        self._client.setex(self._PREFIX + session.id, ttl_seconds, json.dumps(asdict(session)))

    def get(self, session_id: str) -> Optional[Session]:
        raw = self._client.get(self._PREFIX + session_id)
        return Session(**json.loads(raw)) if raw else None


_store: SessionStore | None = None
_store_lock = threading.Lock()


def get_session_store() -> SessionStore:
    """Singleton store; Redis when configured, else in-memory."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                url = get_settings().redis_url.strip()
                if url:
                    try:
                        _store = RedisSessionStore(url)
                        logger.info("Liveness session store: redis")
                    except Exception as e:
                        logger.warning("Redis unavailable (%s); falling back to in-memory store", e)
                        _store = InMemorySessionStore()
                else:
                    _store = InMemorySessionStore()
    return _store
