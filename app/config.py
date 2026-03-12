"""Application configuration from environment."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service
    app_name: str = Field(default="Liveness API", description="Application name")
    environment: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    log_level: str = Field(default="INFO", description="Log level")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Server
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port")
    workers: int = Field(default=1, ge=1, description="Uvicorn workers (1 recommended for model)")

    # InsightFace
    insightface_model: str = Field(default="buffalo_l", description="Model pack name")
    insightface_root: str = Field(
        default="",
        description="Model root dir (default: INSIGHTFACE_HOME or ~/.insightface). Set to /app/.insightface in Docker.",
    )
    insightface_det_size_w: int = Field(default=640, ge=320, description="Detection width")
    insightface_det_size_h: int = Field(default=640, ge=320, description="Detection height")
    insightface_ctx_id: int = Field(default=0, description="GPU device id (-1 for CPU)")

    @property
    def insightface_det_size(self) -> tuple[int, int]:
        return (self.insightface_det_size_w, self.insightface_det_size_h)

    # Liveness thresholds
    laplacian_min: float = Field(
        default=25.0,
        ge=0,
        description="Min Laplacian variance for sharpness (50–80=strict, 25–35=webcam-friendly)",
    )
    face_area_min_ratio: float = Field(
        default=0.02,
        ge=0,
        le=1,
        description="Min face area as ratio of image",
    )
    liveness_confidence_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Min confidence to report live=True",
    )
    face_match_default_similarity_threshold: float = Field(
        default=65.0,
        ge=0,
        le=100,
        description="Default face-match similarity threshold in percent (0-100) when request omits SimilarityThreshold.",
    )
    min_det_score: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Min InsightFace detection score to count a face",
    )

    # Anti-spoof ONNX (MiniFAS-style, 128x128 RGB). Default False: real mobile selfies often fail otherwise.
    antispoof_enabled: bool = Field(default=False, description="Use anti-spoof ONNX model")
    antispoof_model_path: str = Field(
        default="/app/models/antispoof.onnx",
        description="Path to ONNX model file (or URL not used at runtime)",
    )
    antispoof_input_size: int = Field(default=128, ge=64, description="Model input size (H=W)")
    antispoof_real_threshold: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Min score for 'real' class to pass (0.6=reject spoofs; lower to 0.35 if real selfies fail)",
    )
    antispoof_weight: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Weight of anti-spoof in final confidence (0.5 = 50%% antispoof, 50%% heuristics)",
    )
    # Many MiniFAS/SuriAI ONNX models output [real, spoof]. 0 = first class is real (default), 1 = second class is real.
    antispoof_real_index: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Output index for 'real' class: 0 or 1. If real selfies fail with low real_score, try 1.",
    )
    # Padding around face for anti-spoof crop. Larger = more context (screen/paper edges) to detect presentation attacks.
    antispoof_crop_padding_ratio: float = Field(
        default=0.5,
        ge=0,
        le=2,
        description="Padding as ratio of face size (0.5 = 50%% each side). Use 0.5–1.0 to better detect photo/screen held to camera.",
    )
    # SuriAI uses logit_diff = real_logit - spoof_logit; require this >= value to pass (stricter). Set to -999 to disable.
    antispoof_min_logit_diff: float = Field(
        default=0.0,
        ge=-10,
        le=10,
        description="Min (real_logit - spoof_logit) to pass. 0 = balanced, 1+ = stricter (reject more spoofs).",
    )
    # Run anti-spoof on a second, large context crop (face + scene); use min(face_score, context_score). Catches photo/screen held to camera.
    antispoof_dual_crop: bool = Field(
        default=True,
        description="If True, also run on a large-context crop and use the minimum real_score (stricter).",
    )
    antispoof_context_padding_ratio: float = Field(
        default=2.0,
        ge=0.5,
        le=5,
        description="Padding for context crop (2.0 = 2x face size each side). Only used if antispoof_dual_crop=True.",
    )

    # Limits
    max_image_size_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Max request body / image size (bytes)",
    )
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        description="Max requests per minute per client (rate limit)",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
