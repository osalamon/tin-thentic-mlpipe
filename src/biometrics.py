import logging
from pathlib import Path
from typing import Optional

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from src.config import IngestConfig

logger = logging.getLogger(__name__)

# Maximum dimension (width or height) to scale images down to before face detection.
# This prevents memory issues and speeds up inference on large photos.
MAX_IMAGE_DIM = 1200


class BiometricValidator:
    """Validates whether a target image contains the user's face using FaceNet."""

    def __init__(self, config: IngestConfig) -> None:
        self._tolerance: float = config.FACE_DISTANCE_TOLERANCE
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Using device: %s", self._device)

        # Initialize face detection and recognition models
        self._mtcnn: MTCNN = MTCNN(
            keep_all=True,
            device=self._device,
        )
        self._resnet: InceptionResnetV1 = InceptionResnetV1(
            pretrained="vggface2",
        ).eval().to(self._device)

        # Pre-compute the anchor face embedding
        self._anchor_embedding: torch.Tensor = self._compute_anchor_embedding(
            config.ANCHOR_FACE_PATH
        )

    def _compute_anchor_embedding(self, anchor_path: Path) -> torch.Tensor:
        """Load the anchor image, detect the face, and compute its embedding."""
        logger.info("Loading anchor face from %s", anchor_path)
        try:
            img = Image.open(anchor_path).convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Failed to load anchor image: {anchor_path}"
            ) from exc

        img = self._scale_down_pil(img)

        with torch.no_grad():
            face_tensors = self._mtcnn(img)
            if face_tensors is None:
                raise ValueError(
                    f"No face found in anchor image: {anchor_path}"
                )
            # If multiple faces detected, use the first one
            if face_tensors.ndim == 4 and face_tensors.shape[0] > 1:
                logger.warning(
                    "Multiple faces (%d) found in anchor image; using the first one.",
                    face_tensors.shape[0],
                )
                face_tensors = face_tensors[0:1]

            embedding = self._resnet(face_tensors.to(self._device))

        return embedding.squeeze(0)  # Return as 1D tensor

    def _scale_down_pil(self, img: Image.Image) -> Image.Image:
        """Scale image down if its largest dimension exceeds MAX_IMAGE_DIM."""
        w, h = img.size
        if max(w, h) <= MAX_IMAGE_DIM:
            return img
        scale = MAX_IMAGE_DIM / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return img.resize((new_w, new_h), Image.LANCZOS)

    def is_user_face_present(self, image_path: Path) -> bool:
        """
        Check if the user's face is present in the target image.

        Returns True if at least one detected face matches the anchor
        within the configured Euclidean distance tolerance.
        Returns False otherwise (no faces, no match, or corrupted image).
        """
        logger.debug("Scanning for user face in %s", image_path)

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            logger.warning("Failed to load image: %s", image_path, exc_info=True)
            return False

        img = self._scale_down_pil(img)

        with torch.no_grad():
            face_tensors = self._mtcnn(img)
            if face_tensors is None:
                logger.debug("No faces detected in %s", image_path)
                return False

            # Ensure we have a batch dimension
            if face_tensors.ndim == 3:
                face_tensors = face_tensors.unsqueeze(0)

            embeddings = self._resnet(face_tensors.to(self._device))

            # Compute Euclidean distances between each detected face and the anchor
            distances = torch.nn.functional.pairwise_distance(
                embeddings, self._anchor_embedding.unsqueeze(0).expand_as(embeddings)
            )

            matching = distances <= self._tolerance
            if matching.any():
                logger.debug(
                    "Match found in %s (min distance: %.4f)",
                    image_path,
                    distances.min().item(),
                )
                return True

        logger.debug("No face matched the anchor in %s", image_path)
        return False
