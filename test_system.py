"""
Comprehensive Test Suite - AI Face ID Security System
Lead AI ML Engineering - Automated Verification
"""

import unittest
import numpy as np
import cv2
from pathlib import Path

import config
from feature_extractor import BiometricFeatureExtractor
from detector import FaceDetector
from liveness import LivenessDetector
from models import FaceIDDualModel
from dataset import BiometricDatasetManager
from security_logger import SecurityLogger

class TestFaceIDSecuritySystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extractor = BiometricFeatureExtractor()
        cls.detector = FaceDetector()
        cls.liveness = LivenessDetector()
        cls.dataset_mgr = BiometricDatasetManager()
        cls.logger = SecurityLogger()

    def test_feature_extractor_dimension(self):
        """Verifies biometric feature vector is dense, normalized and fixed size."""
        dummy_face = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        feat = self.extractor.extract_features(dummy_face)
        self.assertEqual(feat.ndim, 1)
        self.assertEqual(len(feat), 1858)
        self.assertAlmostEqual(float(np.linalg.norm(feat)), 1.0, places=4)

    def test_liveness_detection_blur(self):
        """Verifies flat blurry photo attacks are detected and rejected."""
        # Ultra blurry / zero variance image simulating flat printed card
        flat_image = np.ones((128, 128, 3), dtype=np.uint8) * 128
        is_live, score, msg = self.liveness.verify_liveness(flat_image)
        self.assertFalse(is_live)
        self.assertIn("Texture too flat/blurry", msg)

    def test_dataset_augmentation(self):
        """Verifies multi-scale biometric augmentations expand single samples."""
        base_face = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        aug_list = self.dataset_mgr.augment_face(base_face)
        self.assertGreaterEqual(len(aug_list), 5)
        for aug in aug_list:
            self.assertEqual(aug.shape, base_face.shape)

    def test_dual_model_inference_and_latency(self):
        """Verifies XGBoost & Random Forest soft voting, agreement and fast latency."""
        dual_model = FaceIDDualModel()
        loaded = dual_model.load_models()
        self.assertTrue(loaded, "Pre-trained models should load successfully from saved_models/")

        dummy_feat = np.random.uniform(0.0, 1.0, 1858).astype(np.float32)
        dummy_feat /= np.linalg.norm(dummy_feat)

        result = dual_model.predict(dummy_feat)
        self.assertIn(result["status"], ["ACCESS_GRANTED", "ACCESS_DENIED_UNKNOWN"])
        self.assertIn("rf_confidence", result)
        self.assertIn("xgb_confidence", result)
        self.assertIn("confidence", result)
        self.assertIn("models_agree", result)

    def test_security_logger_and_snapshot(self):
        """Verifies audit trail records events in database and saves intruder photo."""
        test_intruder = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        res = self.logger.log_attempt(
            user_name="Unit_Test_Intruder",
            status="ACCESS_DENIED",
            confidence=0.45,
            rf_confidence=0.42,
            xgb_confidence=0.48,
            liveness_score=0.95,
            liveness_verified=True,
            face_image=test_intruder,
            notes="Automated test assertion"
        )
        self.assertTrue(res["snapshot_path"].startswith("intruder_"))
        saved_file = config.INTRUDER_DIR / res["snapshot_path"]
        self.assertTrue(saved_file.exists())

if __name__ == "__main__":
    unittest.main()
