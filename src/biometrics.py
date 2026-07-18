import logging
from pathlib import Path
from typing import Optional, Tuple

import face_recognition
import numpy as np
from PIL import Image

from src.config import IngestConfig

logger = logging.getLogger(__name__)

# Maximum dimension (width or height) to scale images down to before face detection.
# This prevents memory issues and speeds up CPU inference on large photos.
MAX_IMAGE_DIM = 1200


class BiometricValidator:
    """Validates whether a target image contains the user's face."""

    def __init__(self, config: IngestConfig) -> None:
        self._tolerance: float = config.FACE_DISTANCE_TOLERANCE
        self._anchor_encoding: np.ndarray = self._load_anchor_encoding(
            config.ANCHOR_FACE_PATH
        )

    def _load_anchor_encoding(self, anchor_path: Path) -> np.ndarray:
        """Load and encode the anchor face. Raises ValueError if no face found."""
        logger.info("Loading anchor face from %s", anchor_path)
        image = face_recognition.load_image_file(str(anchor_path))
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            raise ValueError(
                f"No face found in anchor image: {anchor_path}"
            )
        if len(encodings) > 1:
            logger.warning(
                "Multiple faces (%d) found in anchor image; using the first one.",
                len(encodings),
            )
        return encodings[0]

    def _scale_down(self, image: np.ndarray) -> np.ndarray:
        """Scale image down if its largest dimension exceeds MAX_IMAGE_DIM."""
        h, w = image.shape[:2]
        if max(h, w) <= MAX_IMAGE_DIM:
            return image
        scale = MAX_IMAGE_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        return np.array(pil_img)

    def _largest_face_area(self, locations) -> Optional[Tuple[int, int, int, int]]:
        """Return the face location with the largest bounding box area."""
        if not locations:
            return None
        return max(locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))

    def is_user_face_present(self, image_path: Path) -> bool:
        """
        Check if the user's face is present in the target image.

        Returns True if:
          - Exactly one face matches the anchor within tolerance, OR
          - Multiple faces detected, but the largest face matches the anchor.
        Returns False otherwise (no faces, no match, or multiple matches
        where the largest is not the user).
        """
        logger.debug("Scanning for user face in %s", image_path)
        image = face_recognition.load_image_file(str(image_path))
        image = self._scale_down(image)

        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            logger.debug("No faces detected in %s", image_path)
            return False

        face_encodings = face_recognition.face_encodings(image, face_locations)
        if not face_encodings:
            logger.debug("Faces located but no encodings extracted in %s", image_path)
            return False

        distances = face_recognition.face_distance(face_encodings, self._anchor_encoding)
        matching_indices = [
            i for i, d in enumerate(distances) if d <= self._tolerance
        ]

        if not matching_indices:
            logger.debug("No face matched the anchor in %s", image_path)
            return False

        if len(matching_indices) == 1:
            logger.debug("Exactly one matching face found in %s", image_path)
            return True

        # Multiple matches: accept only if the largest face is among the matches.
        largest_location = self._largest_face_area(face_locations)
        if largest_location is None:
            return False

        largest_index = face_locations.index(largest_location)
        if largest_index in matching_indices:
            logger.debug(
                "Multiple matches in %s; largest face matches anchor.", image_path
            )
            return True

        logger.debug(
            "Multiple matches in %s but largest face does not match anchor.",
            image_path,
        )
        return False
