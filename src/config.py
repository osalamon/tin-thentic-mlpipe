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
