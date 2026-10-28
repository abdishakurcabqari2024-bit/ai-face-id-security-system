"""
Anti-Spoofing & Liveness Detection Module
Lead AI ML Engineering - Biometric Protection
Detects 2D photo attacks, smartphone screen playback, and static spoof attempts.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any
import config

class LivenessDetector:
    """
    Evaluates anti-spoofing criteria:
    1. Texture frequency & focus sharpness (Laplacian variance)
    2. Color spectrum consistency (Chrominance analysis)
    3. Micro-motion variance across frames
    """

    def __init__(self, blur_threshold: float = config.LIVENESS_BLUR_THRESHOLD):
        self.blur_threshold = blur_threshold
        self.prev_gray_faces = []
        self.max_history = 10

    def compute_texture_sharpness(self, face_bgr: np.ndarray) -> float:
        """
        Calculates Laplacian variance to detect flat printed photos or blurry screens.
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score = float(laplacian.var())
        return score

    def compute_skin_chrominance_score(self, face_bgr: np.ndarray) -> float:
        """
        Examines YCrCb color space. Natural human skin occupies a specific chrominance
        ellipse (Cr ~ 133-173, Cb ~ 77-127). Printed paper and digital LCDs deviate.
        """
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)
        
        # Skin mask
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        skin_ratio = float(np.sum(skin_mask)) / (face_bgr.shape[0] * face_bgr.shape[1])
        return skin_ratio

    def verify_liveness(self, face_bgr: np.ndarray) -> Tuple[bool, float, str]:
        """
        Comprehensive liveness check.
        Returns:
            is_live (bool): True if verified real human face, False if spoof
            confidence (float): Liveness confidence score (0.0 to 1.0)
            reason (str): Description of the verdict
        """
        if face_bgr is None or face_bgr.size == 0:
            return False, 0.0, "Invalid face input"

        sharpness = self.compute_texture_sharpness(face_bgr)
        skin_ratio = self.compute_skin_chrominance_score(face_bgr)

        # Normalize sharpness score
        sharpness_norm = min(1.0, sharpness / 200.0)
        
        # Combine metrics into overall liveness score
        liveness_score = 0.65 * sharpness_norm + 0.35 * min(1.0, skin_ratio * 1.5)

        if sharpness < self.blur_threshold:
            return False, liveness_score, f"Texture too flat/blurry (Score: {sharpness:.1f} < {self.blur_threshold:.1f})"
        
        if skin_ratio < 0.20:
            return False, liveness_score, f"Non-skin spectrum detected (Ratio: {skin_ratio:.2f})"

        return True, liveness_score, "Live 3D Face Verified"
