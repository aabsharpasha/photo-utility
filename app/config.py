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
    use_gpu: bool = Field(
        default=False,
        description="If True, try CUDA for ONNX Runtime (anti-spoof + InsightFace) when available.",
    )
    workers: int = Field(default=1, ge=1, description="Uvicorn workers (1 recommended for model)")
    api_key_query_value: str = Field(
        default="my-secret-api-key",
        description="If set, require this API key in query parameter `api_key` for protected endpoints.",
    )

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
    face_match_low_quality_similarity_threshold_delta: float = Field(
        default=10.0,
        ge=0,
        le=100,
        description="Threshold reduction applied when source image quality is low (Aadhaar/photo scan case).",
    )
    face_match_low_face_area_ratio_min: float = Field(
        default=0.015,
        ge=0,
        le=1,
        description="Source face area ratio below this is treated as low quality for face-match.",
    )
    face_match_low_sharpness_laplacian_min: float = Field(
        default=18.0,
        ge=0,
        description="Source Laplacian variance below this is treated as low quality for face-match.",
    )
    face_match_low_det_score_min: float = Field(
        default=0.45,
        ge=0,
        le=1,
        description="Source detection score below this is treated as low quality for face-match.",
    )
    min_det_score: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Min InsightFace detection score to count a face",
    )

    # Anti-spoof ONNX (MiniFAS-style, 128x128 RGB).
    antispoof_enabled: bool = Field(
        default=True,
        description="Use anti-spoof ONNX model (baked-in default: enabled).",
    )
    antispoof_model_path: str = Field(
        default="/app/models/antispoof.onnx",
        description="Path to ONNX model file (or URL not used at runtime)",
    )
    antispoof_input_size: int = Field(default=128, ge=64, description="Model input size (H=W)")
    antispoof_real_threshold: float = Field(
        default=0.54,
        ge=0,
        le=1,
        description="Asymmetric: min face-crop 'real' for single-frame liveness (context uses antispoof_context_real_threshold).",
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
    # Second crop: large context (face + scene). How face + context combine: see antispoof_dual_crop_strategy.
    antispoof_dual_crop: bool = Field(
        default=True,
        description="If True, also score a large scene crop next to the face crop.",
    )
    antispoof_dual_crop_strategy: Literal["min", "asymmetric"] = Field(
        default="asymmetric",
        description=(
            "min: require min(face,context) >= threshold (harsh on real users). "
            "asymmetric: face >= antispoof_real_threshold AND context >= antispoof_context_real_threshold."
        ),
    )
    antispoof_context_real_threshold: float = Field(
        default=0.38,
        ge=0.0,
        le=1.0,
        description="Asymmetric mode: min context-crop 'real' score (below face bar; replay often fails here).",
    )
    antispoof_context_padding_ratio: float = Field(
        default=2.0,
        ge=0.5,
        le=5,
        description="Padding for context crop (2.0 = 2x face size each side). Only used if antispoof_dual_crop=True.",
    )

    # Motion-only liveness (/v1/liveness-motion): stronger checks for video / screen replay.
    motion_require_context_antispoof: bool = Field(
        default=True,
        description="If True, run large-context anti-spoof each motion frame (uses antispoof_dual_crop_strategy).",
    )
    motion_antispoof_real_threshold: float = Field(
        default=0.52,
        ge=0.0,
        le=1.0,
        description="Asymmetric: min face-crop 'real' score per motion frame (context has its own threshold).",
    )
    motion_antispoof_context_real_threshold: float = Field(
        default=0.36,
        ge=0.0,
        le=1.0,
        description="Asymmetric motion only: min context-crop 'real' score (override via check()).",
    )
    motion_antispoof_min_logit_diff: float = Field(
        default=0.05,
        ge=-10.0,
        le=10.0,
        description=(
            "With asymmetric strategy, applied to face crop only. With min strategy, applied to both crops. "
            "0 disables logit gate."
        ),
    )
    motion_frame_liveness_confidence_threshold: float = Field(
        default=0.41,
        ge=0.0,
        le=1.0,
        description="Min blended confidence per motion frame for that frame to count as live.",
    )
    motion_require_all_frames_live: bool = Field(
        default=False,
        description=(
            "If True, every motion frame must pass per-frame liveness (strictest). "
            "If False, require motion_min_frames_with_live_face (default 2/3) to reduce real-user false rejects."
        ),
    )
    motion_min_frames_with_live_face: int = Field(
        default=2,
        ge=1,
        le=12,
        description="When motion_require_all_frames_live is False, min frames that pass per-frame liveness.",
    )
    motion_moire_gate_enabled: bool = Field(
        default=True,
        description=(
            "If True, fail motion when moiré score exceeds motion_moire_max_score (helps vs filming a screen). "
            "Disable or raise max if real scenes false-fail."
        ),
    )
    motion_moire_max_score: float = Field(
        default=0.76,
        ge=0.0,
        le=1.0,
        description=(
            "Fail when max per-frame moiré >= this. Balance: real webcams vs filming a screen. Raise if false rejects."
        ),
    )
    motion_min_frames: int = Field(
        default=3,
        ge=2,
        le=12,
        description="Minimum frames for motion liveness (3+ mimics multi-frame video-style checks).",
    )
    motion_min_normalized_shift: float = Field(
        default=0.01,
        ge=0.005,
        le=0.2,
        description="Min normalized face-center movement between frames (see motion_require_shift_all_consecutive_pairs).",
    )
    motion_require_shift_all_consecutive_pairs: bool = Field(
        default=False,
        description=(
            "If True, every consecutive pair must exceed the shift minimum. If False, only the "
            "largest pairwise shift must exceed it (more forgiving when one interval is nearly still)."
        ),
    )
    motion_face_consistency_enabled: bool = Field(
        default=True,
        description="If True, require embedding similarity across each consecutive pair (same identity as AWS-style continuity).",
    )
    motion_min_consecutive_face_similarity: float = Field(
        default=38.0,
        ge=0.0,
        le=100.0,
        description="Min face similarity (0–100, CompareFaces-style) between consecutive frames.",
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
