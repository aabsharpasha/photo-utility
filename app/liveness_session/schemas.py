"""Request/Response schemas for the session-based liveness endpoints."""

from typing import Optional

from pydantic import BaseModel, Field


class SessionCreateResponse(BaseModel):
    session_id: str
    stream_token: str = Field(
        description="Short-lived token for the WebSocket stream (browsers cannot send custom headers on WS)."
    )
    challenge: str = Field(description="blink | turn_left | turn_right")
    required_blinks: Optional[int] = Field(
        default=None, description="Only set for the blink challenge."
    )
    expires_in_seconds: int
    stream_path: str = Field(description="WebSocket path to stream frames to (append ?token=...).")


class SessionResultResponse(BaseModel):
    session_id: str
    status: str = Field(description="pending | streaming | done | expired")
    live: Optional[bool] = None
    reasons: list[str] = Field(default_factory=list)
