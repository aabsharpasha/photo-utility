"""Request/Response schemas for OpenAPI and validation."""

from typing import Literal, Optional

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
        description="Multiple base64-encoded images (JPEG/PNG) in sequence; minimum count is enforced server-side (default 3).",
    )
    challenge: Literal["none", "left-right", "right-left", "side-to-side"] = Field(
        default="side-to-side",
        description=(
            "Optional challenge-response pattern for replay resistance. "
            "'left-right' means frame2 should shift left from frame1 and frame3 shift right from frame2; "
            "'right-left' is the opposite."
        ),
    )


class LivenessResponse(BaseModel):
    """Liveness check result."""

    live: bool = Field(..., description="True if image is classified as live face")
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "0–1. Single-image: score for that frame. Motion: blended per-frame confidence "
            "(details.aggregate_confidence); may be high while live is false if an auxiliary gate failed."
        ),
    )
    details: dict = Field(default_factory=dict, description="Diagnostic details")
    errors: list[str] = Field(default_factory=list, description="Errors if any")

    model_config = {"json_schema_extra": {"example": {"live": True, "confidence": 0.85, "details": {}, "errors": []}}}


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="ok | degraded")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Current environment")


class RekognitionImage(BaseModel):
    """AWS Rekognition-style Image wrapper (we support Bytes only)."""

    Bytes: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded image bytes. Optional data URI prefix supported.",
    )


class CompareFacesRequest(BaseModel):
    """
    Face match request following AWS Rekognition CompareFaces.
    """

    SourceImage: RekognitionImage = Field(..., description="Source image containing the face to search for.")
    TargetImage: RekognitionImage = Field(..., description="Target image that may contain matching faces.")
    SimilarityThreshold: Optional[float] = Field(
        default=45.0,
        ge=0,
        le=100,
        description="Face match threshold in percent (0-100). Defaults to 45 in case of aadhar photo match.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "SourceImage": {
                    "Bytes": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                },
                "TargetImage": {
                    "Bytes": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                },
                "SimilarityThreshold": 45,
            }
        }
    }


class BoundingBox(BaseModel):
    Left: float = Field(..., ge=0, le=1)
    Top: float = Field(..., ge=0, le=1)
    Width: float = Field(..., ge=0, le=1)
    Height: float = Field(..., ge=0, le=1)


class FaceDetail(BaseModel):
    # Use a plain dict here to avoid Optional/forward-ref issues with nested models in older Pydantic setups.
    BoundingBox: dict = Field(default_factory=dict)
    Confidence: float = Field(..., ge=0, le=100)


class CompareFaceMatch(BaseModel):
    Similarity: float = Field(..., ge=0, le=100)
    Face: FaceDetail


class CompareFacesResponse(BaseModel):
    """
    Response payload aligned with AWS Rekognition CompareFaces (subset).
    """

    Match: bool = Field(
        default=False,
        description="True when at least one target face matches source above threshold.",
    )
    SourceImageFace: Optional[FaceDetail] = None
    FaceMatches: list[CompareFaceMatch] = Field(
        default_factory=list,
        description="Faces in target image that match the source face.",
    )
    UnmatchedFaces: list[FaceDetail] = Field(
        default_factory=list,
        description="Faces in target image that did not meet similarity threshold.",
    )

