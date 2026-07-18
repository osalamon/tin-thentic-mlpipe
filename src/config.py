from dataclasses import dataclass, field
from pathlib import Path
from typing import Set


@dataclass
class IngestConfig:
    """Configuration for file discovery and metadata ingestion."""
    
    STAGING_DIR: Path = Path(".")
    START_YEAR: int = 2024
    ALLOWED_EXTENSIONS: Set[str] = field(default_factory=lambda: {"jpg", "jpeg", "png", "heic", "webp"})
    MIN_FILE_SIZE_BYTES: int = 1500
    OLD_PROFILES_DIR: Path = Path("./old_profiles")
    SIMILARITY_THRESHOLD: float = 0.85
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    ANCHOR_FACE_PATH: Path = Path("./anchor_face.jpg")
    FACE_DISTANCE_TOLERANCE: float = 0.55
