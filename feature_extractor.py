"""
Feature Extractor Module
Lead AI ML Engineering - Hybrid Feature Pipeline
Combines HOG (Histogram of Oriented Gradients), Multi-Block LBP (Local Binary Patterns),
and Spatial Statistical Invariants for Tree-Based Models (XGBoost & Random Forest).
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import config

class BiometricFeatureExtractor:
    """
    Extracts dense, rotation/illumination invariant biometric feature vectors
    specifically engineered for Tree-Based Classifiers (Random Forest & XGBoost).
    """

    def __init__(self, target_size: Tuple[int, int] = (config.FACE_WIDTH, config.FACE_HEIGHT)):
        self.target_size = target_size
        
        # Native OpenCV HOG Descriptor (High-performance C++ implementation)
        # winSize=(128,128), blockSize=(32,32), blockStride=(16,16), cellSize=(16,16), nbins=8
        self.hog = cv2.HOGDescriptor(
            _winSize=self.target_size,
            _blockSize=(32, 32),
            _blockStride=(16, 16),
            _cellSize=(16, 16),
            _nbins=8
        )
        self.feature_dim = None

    def preprocess_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocesses face crop:
        1. Resize to target dimension
        2. Convert to Grayscale
        3. Equalize Histogram (CLAHE) to neutralize harsh lighting and shadows
        """
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face image provided to preprocessor.")

        resized = cv2.resize(face_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized.copy()

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Bilateral filter to smooth noise while keeping facial edges sharp
        filtered = cv2.bilateralFilter(equalized, d=5, sigmaColor=50, sigmaSpace=50)
        return filtered

    def _extract_hog(self, gray_face: np.ndarray) -> np.ndarray:
        """
        Extracts HOG descriptor representing structural edge orientations (eyes, nose, jawline).
        """
        hog_features = self.hog.compute(gray_face)
        if hog_features is None:
            return np.zeros((392,), dtype=np.float32)
        return hog_features.flatten()

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """
        Calculates 8-neighbor Local Binary Pattern for micro-texture representation.
        Vectorized with NumPy for real-time speed.
        """
        h, w = gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        center = gray[1:-1, 1:-1]
        
        # 8 neighbors
        lbp |= ((gray[0:-2, 0:-2] >= center) << 7).astype(np.uint8)
        lbp |= ((gray[0:-2, 1:-1] >= center) << 6).astype(np.uint8)
        lbp |= ((gray[0:-2, 2:  ] >= center) << 5).astype(np.uint8)
        lbp |= ((gray[1:-1, 2:  ] >= center) << 4).astype(np.uint8)
        lbp |= ((gray[2:  , 2:  ] >= center) << 3).astype(np.uint8)
        lbp |= ((gray[2:  , 1:-1] >= center) << 2).astype(np.uint8)
        lbp |= ((gray[2:  , 0:-2] >= center) << 1).astype(np.uint8)
        lbp |= ((gray[1:-1, 0:-2] >= center) << 0).astype(np.uint8)
        
        return lbp

    def _extract_spatial_lbp_histograms(self, gray_face: np.ndarray, grid_x: int = 4, grid_y: int = 4) -> np.ndarray:
        """
        Computes spatially-partitioned LBP histograms (4x4 cells).
        Preserves spatial locality (forehead, left eye, right eye, nose, mouth).
        """
        lbp_image = self._compute_lbp(gray_face)
        h, w = lbp_image.shape
        step_h = h // grid_y
        step_w = w // grid_x
        
        histograms = []
        for r in range(grid_y):
            for c in range(grid_x):
                cell = lbp_image[r*step_h:(r+1)*step_h, c*step_w:(c+1)*step_w]
                hist, _ = np.histogram(cell, bins=16, range=(0, 256), density=True)
                histograms.append(hist)
                
        return np.concatenate(histograms)

    def _extract_statistical_invariants(self, gray_face: np.ndarray) -> np.ndarray:
        """
        Calculates intensity statistics and spatial moments for facial geometry.
        """
        mean_val = np.mean(gray_face) / 255.0
        std_val = np.std(gray_face) / 255.0
        
        # Horizontal & vertical intensity profiles (collapsed projections)
        h_profile = cv2.resize(np.mean(gray_face, axis=1), (1, 16)).flatten() / 255.0
        v_profile = cv2.resize(np.mean(gray_face, axis=0), (1, 16)).flatten() / 255.0
        
        return np.concatenate([[mean_val, std_val], h_profile, v_profile])

    def extract_features(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Primary feature extraction pipeline.
        Concatenates HOG + Spatial LBP + Structural Projections.
        Returns a 1D normalized float32 NumPy array.
        """
        gray_processed = self.preprocess_face(face_bgr)
        
        hog_feat = self._extract_hog(gray_processed)
        lbp_feat = self._extract_spatial_lbp_histograms(gray_processed)
        stat_feat = self._extract_statistical_invariants(gray_processed)
        
        # Concatenate into unified biometric vector
        feature_vector = np.concatenate([hog_feat, lbp_feat, stat_feat]).astype(np.float32)
        
        # L2-normalization for numerical stability across lighting changes
        norm = np.linalg.norm(feature_vector)
        if norm > 1e-6:
            feature_vector = feature_vector / norm
            
        if self.feature_dim is None:
            self.feature_dim = len(feature_vector)
            
        return feature_vector

    def extract_batch(self, faces: List[np.ndarray]) -> np.ndarray:
        """
        Extracts features for a batch of face images.
        """
        features = [self.extract_features(face) for face in faces]
        return np.vstack(features)
