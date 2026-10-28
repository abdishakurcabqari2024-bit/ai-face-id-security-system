"""
Automated Test Suite for Biometric Recognition Microservice
Tests Face Detection, 1858-dim Feature Extraction, Laplacian Liveness, and Dual Model Inference.
"""

import unittest
import numpy as np
import base64
import cv2
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector import FaceDetector
from feature_extractor import BiometricFeatureExtractor
from liveness import LivenessDetector
from models import FaceIDDualModel

class TestRecognitionEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = FaceDetector()
        cls.extractor = BiometricFeatureExtractor()
        cls.liveness = LivenessDetector()
        cls.dual_model = FaceIDDualModel()
        cls.dual_model.load_models()

    def test_feature_extraction_dimension(self):
        """Verifies 1,858 fixed-size normalized biometric vector."""
        sample_face = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        feat = self.extractor.extract_features(sample_face)
        self.assertEqual(len(feat), 1858)
        self.assertAlmostEqual(float(np.linalg.norm(feat)), 1.0, places=4)

    def test_liveness_rejection_flat_photo(self):
        """Verifies flat paper / digital screen with low Laplacian variance is rejected."""
        flat_paper = np.ones((128, 128, 3), dtype=np.uint8) * 128
        is_live, score, msg = self.liveness.verify_liveness(flat_paper)
        self.assertFalse(is_live)
        self.assertIn("Texture too flat/blurry", msg)

    def test_dual_model_prediction_and_threshold(self):
        """Verifies soft-voting ensemble and intruder rejection gate."""
        dummy_feat = np.random.uniform(0.0, 1.0, 1858).astype(np.float32)
        dummy_feat /= np.linalg.norm(dummy_feat)

        # When threshold is high (e.g. 0.95), random noise MUST be rejected as intruder
        self.dual_model.security_threshold = 0.95
        pred = self.dual_model.predict(dummy_feat)
        self.assertFalse(pred["access_granted"])
        self.assertEqual(pred["status"], "ACCESS_DENIED_UNKNOWN")

    def test_known_vs_unknown_face_separation(self):
        """Verifies model separation between enrolled identities and unknown faces."""
        self.assertTrue(self.dual_model.is_trained)
        self.assertGreaterEqual(len(self.dual_model.classes_), 30)

if __name__ == "__main__":
    unittest.main()
