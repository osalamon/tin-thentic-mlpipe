import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional

from PIL import Image
from tqdm import tqdm

from src.config import IngestConfig

logger = logging.getLogger(__name__)


def get_exif_datetime(filepath: Path) -> Optional[datetime]:
    """Extract DateTimeOriginal from EXIF tags using PIL.
    
    Returns None if EXIF parsing fails or the tag is missing.
    """
    try:
        img = Image.open(filepath)
        exif = img._getexif()
        if exif is None:
            return None
        
        # EXIF tag 36867 is DateTimeOriginal
        date_str = exif.get(36867)
        if date_str is None:
            return None
        
        # Format: "YYYY:MM:DD HH:MM:SS"
        return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def get_system_creation_time(filepath: Path) -> Optional[datetime]:
    """Fallback: get file creation time from the filesystem."""
    try:
        stat = os.stat(filepath)
        # st_birthtime on macOS, st_ctime on Linux (creation time)
        ctime = getattr(stat, "st_birthtime", stat.st_ctime)
        return datetime.fromtimestamp(ctime)
    except Exception:
        return None


def discover_files(config: IngestConfig) -> Iterator[Dict[str, object]]:
    """Scan STAGING_DIR recursively and yield metadata for matching image files.
    
    Yields:
        dict with keys: "path" (str), "year" (int), "size" (int)
    """
    staging_dir = config.STAGING_DIR
    allowed_exts = config.ALLOWED_EXTENSIONS
    min_size = config.MIN_FILE_SIZE_BYTES
    start_year = config.START_YEAR
    
    for root, _, files in os.walk(staging_dir):
        for filename in files:
            filepath = Path(root) / filename
            
            # Check extension
            ext = filepath.suffix.lower().lstrip(".")
            if ext not in allowed_exts:
                continue
            
            # Check file size
            try:
                size = filepath.stat().st_size
            except OSError:
                continue
            
            if size < min_size:
                continue
            
            # Try EXIF first, then fallback to system creation time
            dt = get_exif_datetime(filepath)
            if dt is None:
                dt = get_system_creation_time(filepath)
            
            if dt is None:
                logger.debug("Could not determine date for: %s", filepath)
                continue
            
            if dt.year >= start_year:
                yield {
                    "path": str(filepath),
                    "year": dt.year,
                    "size": size,
                }


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Parse test directory from command line or use current directory
    test_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    
    config = IngestConfig(STAGING_DIR=test_dir)
    
    # First pass: count total files (for progress bar context)
    allowed_exts = config.ALLOWED_EXTENSIONS
    total_files = 0
    for root, _, files in os.walk(test_dir):
        for f in files:
            ext = Path(f).suffix.lower().lstrip(".")
            if ext in allowed_exts:
                total_files += 1
    
    logger.info("Scanning %d files in %s...", total_files, test_dir)
    
    matching_files = 0
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for metadata in discover_files(config):
            matching_files += 1
            pbar.update(1)
            pbar.set_postfix(matching=matching_files)
    
    logger.info("Total files found: %d", total_files)
    logger.info("Files matching date filter (year >= %d): %d", config.START_YEAR, matching_files)
