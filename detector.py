"""
Face Detection & Alignment Module
Lead AI ML Engineering - High Accuracy Facial Localization
Uses multi-scale cascade detectors with margin expansion and eye alignment support.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import config

class FaceDetector:
    """
    Locates faces and eyes within image frames with bounding margin padding.
    """

    def __init__(self):
        cascade_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_default.xml")
        self.face_alt_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_alt2.xml")
        self.eye_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_eye.xml")

    def detect_faces(self, image_bgr: np.ndarray, margin: float = 0.15) -> List[Tuple[int, int, int, int, np.ndarray]]:
        """
        Detects all faces in the provided frame.
        Returns a list of tuples: (x, y, w, h, cropped_face_bgr)
        Includes margin padding to preserve critical jawline and forehead structures.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h_img, w_img = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Primary pass
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Fallback to alternate cascade if no faces detected
        if len(faces) == 0:
            faces = self.face_alt_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(50, 50)
            )

        detected_faces = []
        for (x, y, w, h) in faces:
            # Add margin around face
            pad_x = int(w * margin)
            pad_y = int(h * margin)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)

            crop = image_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                detected_faces.append((x, y, w, h, crop))

        return detected_faces

    def detect_eyes(self, face_crop_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects eyes within a face crop (used for liveness / gaze / alignment verification).
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return []

        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        # Search eyes in upper 65% of the face
        upper_half = gray[0:int(gray.shape[0] * 0.65), :]
        eyes = self.eye_cascade.detectMultiScale(upper_half, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
        return list(eyes)
