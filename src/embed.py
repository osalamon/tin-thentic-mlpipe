import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.config import IngestConfig

logger = logging.getLogger(__name__)


class CLIPExclusionEngine:
    """Pre-computes embeddings of old profile images and checks new images for duplicates."""

    def __init__(self, config: IngestConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        self.model: CLIPModel = CLIPModel.from_pretrained(config.CLIP_MODEL).to(self.device)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        self.model.eval()

        self._cached_embeddings: Optional[torch.Tensor] = None
        self._build_cache()

    def _load_image_safe(self, image_path: Path) -> Optional[Image.Image]:
        """Load an image with robust error handling for truncated/corrupt files."""
        try:
            img = Image.open(image_path)
            img = img.convert("RGB")
            return img
        except Exception:
            logger.warning("Skipping unreadable image: %s", image_path)
            return None

    def _get_embedding(self, image: Image.Image) -> Optional[torch.Tensor]:
        """Generate a normalized CLIP embedding for a single PIL image."""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            # get_image_features returns BaseModelOutputWithPooling; extract the tensor
            embedding = outputs.pooler_output
            # Normalize to unit length for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu()
        except Exception as e:
            logger.warning("Failed to generate embedding for image: %s", e)
            return None

    def _build_cache(self) -> None:
        """Pre-compute and cache normalized embeddings for all images in OLD_PROFILES_DIR."""
        old_dir = self.config.OLD_PROFILES_DIR
        if not old_dir.exists():
            logger.warning("OLD_PROFILES_DIR does not exist: %s. Cache will be empty.", old_dir)
            self._cached_embeddings = torch.empty((0, self.model.config.projection_dim))
            return

        image_paths: List[Path] = []
        for ext in self.config.ALLOWED_EXTENSIONS:
            image_paths.extend(old_dir.glob(f"*.{ext}"))
            image_paths.extend(old_dir.glob(f"*.{ext.upper()}"))

        if not image_paths:
            logger.warning("No images found in OLD_PROFILES_DIR. Cache will be empty.")
            self._cached_embeddings = torch.empty((0, self.model.config.projection_dim))
            return

        embeddings_list: List[torch.Tensor] = []
        for img_path in image_paths:
            img = self._load_image_safe(img_path)
            if img is None:
                continue
            emb = self._get_embedding(img)
            if emb is not None:
                embeddings_list.append(emb)

        if embeddings_list:
            self._cached_embeddings = torch.cat(embeddings_list, dim=0)
        else:
            self._cached_embeddings = torch.empty((0, self.model.config.projection_dim))

        logger.info("Cached %d embeddings from old profiles.", self._cached_embeddings.shape[0])

        # Free GPU memory after building cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def is_duplicate(self, image_path: Path) -> bool:
        """Check if an image is too similar to any old profile image.

        Returns True if cosine similarity with any cached embedding exceeds SIMILARITY_THRESHOLD.
        """
        if self._cached_embeddings is None or self._cached_embeddings.shape[0] == 0:
            return False

        img = self._load_image_safe(image_path)
        if img is None:
            logger.warning("Cannot check duplicate for unreadable image: %s", image_path)
            return False

        emb = self._get_embedding(img)
        if emb is None:
            return False

        # Cosine similarity: dot product of normalized vectors
        similarities = torch.mm(emb, self._cached_embeddings.T).squeeze(0)
        max_sim = similarities.max().item()

        # Free GPU memory after processing
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return max_sim >= self.config.SIMILARITY_THRESHOLD


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = IngestConfig()
    engine = CLIPExclusionEngine(config)

    # Print cache stats
    if engine._cached_embeddings is not None:
        num_cached = engine._cached_embeddings.shape[0]
        emb_dim = engine._cached_embeddings.shape[1] if num_cached > 0 else 0
        logger.info("Cache stats: %d embeddings, dimension=%d", num_cached, emb_dim)
    else:
        logger.info("Cache is empty.")

    # Test a sample image if provided as argument, otherwise pick first from old_profiles
    if len(sys.argv) > 1:
        test_image = Path(sys.argv[1])
    else:
        old_dir = config.OLD_PROFILES_DIR
        if old_dir.exists():
            candidates = list(old_dir.iterdir())
            test_image = candidates[0] if candidates else None
        else:
            test_image = None

    if test_image and test_image.exists():
        logger.info("Testing duplicate check on: %s", test_image)
        is_dup = engine.is_duplicate(test_image)
        logger.info("Is duplicate: %s", is_dup)
    else:
        logger.warning("No test image available.")
