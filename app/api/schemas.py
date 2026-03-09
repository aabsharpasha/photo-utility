"""Request/Response schemas for OpenAPI and validation."""

from pydantic import BaseModel, Field


class LivenessRequest(BaseModel):
    """Input for liveness check: single image as base64."""

    image_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded image (JPEG/PNG). Optional data URI prefix supported.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."}
        }
    }


class MotionLivenessRequest(BaseModel):
    """Input for motion-based liveness: multiple frames (base64)."""

    frames: list[str] = Field(
        ...,
        min_length=2,
        description="At least 2 base64-encoded images (JPEG/PNG) captured in sequence.",
    )


class LivenessResponse(BaseModel):
    """Liveness check result."""

    live: bool = Field(..., description="True if image is classified as live face")
    confidence: float = Field(..., ge=0, le=1, description="Liveness confidence 0–1")
    details: dict = Field(default_factory=dict, description="Diagnostic details")
    errors: list[str] = Field(default_factory=list, description="Errors if any")

    model_config = {"json_schema_extra": {"example": {"live": True, "confidence": 0.85, "details": {}, "errors": []}}}


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="ok | degraded")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Current environment")
