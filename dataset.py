"""
Dataset Management & Biometric Data Augmentation Module
Lead AI ML Engineering - High Quality Biometric Pipeline
Handles enrollment image persistence, robust facial data augmentation,
and training dataset generation for Random Forest & XGBoost.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import config
from feature_extractor import BiometricFeatureExtractor
from detector import FaceDetector

class BiometricDatasetManager:
    """
    Manages face images, user enrollment, and biometric augmentations.
    """

    def __init__(self,
                 faces_dir: Path = config.FACES_DIR,
                 feature_extractor: Optional[BiometricFeatureExtractor] = None):
        self.faces_dir = faces_dir
        self.extractor = feature_extractor or BiometricFeatureExtractor()
        self.detector = FaceDetector()

    def get_enrolled_users(self) -> List[str]:
        """
        Lists all enrolled user names from the dataset directory.
        """
        if not self.faces_dir.exists():
            return []
        users = [d.name for d in self.faces_dir.iterdir() if d.is_dir()]
        return sorted(users)

    def create_user_directory(self, user_name: str) -> Path:
        """
        Creates storage folder for a user.
        """
        clean_name = "".join([c for c in user_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        user_path = self.faces_dir / clean_name
        user_path.mkdir(parents=True, exist_ok=True)
        return user_path

    def augment_face(self, face_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Applies realistic biometric augmentations to ensure tree models
        are invariant to lighting, contrast, and minor head poses:
        1. Original
        2. Brightness Boost (+25%)
        3. Shadow / Low Light (-25%)
        4. Gamma Correction (gamma = 1.3)
        5. Minor Rotation (+6 deg)
        6. Minor Rotation (-6 deg)
        7. Horizontal Flip
        """
        augmented = [face_bgr]
        h, w = face_bgr.shape[:2]

        # 1. Brightness Boost
        bright = cv2.convertScaleAbs(face_bgr, alpha=1.25, beta=15)
        augmented.append(bright)

        # 2. Low Light / Shadow
        dim = cv2.convertScaleAbs(face_bgr, alpha=0.75, beta=-15)
        augmented.append(dim)

        # 3. Gamma Correction (Non-linear illumination shift)
        inv_gamma = 1.0 / 1.3
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_adj = cv2.LUT(face_bgr, table)
        augmented.append(gamma_adj)

        # 4. Rotations (+/- 6 degrees)
        center = (w // 2, h // 2)
        for angle in [6, -6]:
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(face_bgr, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            augmented.append(rotated)

        # 5. Horizontal Flip (Facial symmetry)
        flipped = cv2.flip(face_bgr, 1)
        augmented.append(flipped)

        return augmented

    def save_face_sample(self, user_name: str, face_bgr: np.ndarray, sample_idx: int) -> str:
        """
        Saves a single captured face crop to the user's directory.
        """
        user_dir = self.create_user_directory(user_name)
        filename = f"{user_name}_{sample_idx:04d}.jpg"
        filepath = user_dir / filename
        
        resized = cv2.resize(face_bgr, (config.FACE_WIDTH, config.FACE_HEIGHT))
        cv2.imwrite(str(filepath), resized)
        return str(filepath)

    def load_dataset(self, apply_augmentation: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Loads all faces from disk, applies augmentations, extracts biometric feature vectors.
        Returns:
            X: 2D NumPy array of shape (N_samples, N_features)
            y: List of strings (User labels)
        """
        users = self.get_enrolled_users()
        if len(users) == 0:
            return np.empty((0, 0)), []

        feature_list = []
        label_list = []

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        for user in users:
            user_dir = self.faces_dir / user
            image_files = [f for f in user_dir.iterdir() if f.suffix.lower() in valid_extensions]

            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                if apply_augmentation:
                    augmented_faces = self.augment_face(img)
                else:
                    augmented_faces = [img]

                for face in augmented_faces:
                    try:
                        feat = self.extractor.extract_features(face)
                        feature_list.append(feat)
                        label_list.append(user)
                    except Exception as e:
                        print(f"Skipping corrupt image {img_path}: {e}")

        if len(feature_list) == 0:
            return np.empty((0, 0)), []

        X = np.vstack(feature_list)
        return X, label_list

    def load_kaggle_dataset(self, split: str = "train", use_cache: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Loads the cleaned Kaggle dataset (31 classes).
        Extracts 1,858-dim biometric vectors and utilizes .npz caching for instantaneous subsequent loading.
        """
        split_dir = config.KAGGLE_DATASET_DIR / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Kaggle dataset split directory not found: {split_dir}")

        cache_path = config.CACHE_DIR / f"kaggle_{split}_features.npz"

        # Check if cached
        if use_cache and cache_path.exists():
            print(f"[CACHE] Loading pre-extracted Kaggle features from: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            X = data['X']
            y = list(data['y'])
            return X, y

        print(f"[EXTRACT] Processing images from {split_dir}...")
        feature_list = []
        label_list = []

        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        total_imgs = sum(len(list(d.glob("*.jpg"))) for d in class_dirs)
        print(f"[EXTRACT] Found {len(class_dirs)} classes, {total_imgs} images. Extracting biometric vectors...")

        processed = 0
        valid_exts = ('.jpg', '.jpeg', '.png')
        for cdir in class_dirs:
            label = cdir.name
            img_files = [f for f in cdir.iterdir() if f.suffix.lower() in valid_exts]
            for img_p in img_files:
                img = cv2.imread(str(img_p))
                if img is None:
                    continue
                try:
                    feat = self.extractor.extract_features(img)
                    feature_list.append(feat)
                    label_list.append(label)
                    processed += 1
                except Exception as e:
                    print(f"Skipping {img_p}: {e}")

        X = np.vstack(feature_list)
        y = label_list

        # Cache features to disk
        if use_cache:
            print(f"[CACHE] Saving extracted features to {cache_path} ({X.shape[0]} samples)...")
            np.savez_compressed(cache_path, X=X, y=np.array(y))

        return X, y

    def import_kaggle_profiles_to_faces(self, samples_per_user: int = 15) -> int:
        """
        Copies sample images from Kaggle dataset into data/faces/
        so that all 31 classes appear in the system GUI and local user directories.
        """
        train_dir = config.KAGGLE_DATASET_DIR / "train"
        if not train_dir.exists():
            return 0

        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        imported_count = 0

        for cdir in class_dirs:
            user_name = cdir.name
            user_target_dir = self.create_user_directory(user_name)
            img_files = sorted(list(cdir.glob("*.jpg")))[:samples_per_user]

            for s_idx, img_p in enumerate(img_files):
                dest_file = user_target_dir / f"{user_name}_{s_idx:04d}.jpg"
                if not dest_file.exists():
                    img = cv2.imread(str(img_p))
                    if img is not None:
                        resized = cv2.resize(img, (config.FACE_WIDTH, config.FACE_HEIGHT))
                        cv2.imwrite(str(dest_file), resized)
                        imported_count += 1

        return imported_count

    def generate_demo_dataset(self) -> Dict[str, int]:
        """
        Generates synthetic biometric profiles for out-of-the-box demonstration & unit testing
        (User: 'Ali_Hassan', 'Fatima_Omar', 'Ahmed_Noor').
        """
        demo_users = ["Ali_Hassan", "Fatima_Omar", "Ahmed_Noor"]
        created = {}

        for idx, user in enumerate(demo_users):
            user_dir = self.create_user_directory(user)
            # Create synthetic facial patterns with distinctive textures and features
            np.random.seed(100 + idx * 37)
            
            for s in range(15):
                # Base facial canvas
                canvas = np.ones((config.FACE_HEIGHT, config.FACE_WIDTH, 3), dtype=np.uint8) * (180 + idx * 15)
                
                # Synthetic facial oval
                cv2.ellipse(canvas, (64, 64), (40 + idx*2, 52 + s%3), 0, 0, 360, (140 - idx*10, 160, 200), -1)
                
                # Eye landmarks
                eye_y = 50 + (s % 2)
                cv2.circle(canvas, (46, eye_y), 6, (30 + idx*20, 30, 40), -1)
                cv2.circle(canvas, (82, eye_y), 6, (30 + idx*20, 30, 40), -1)
                
                # Nose landmark
                cv2.line(canvas, (64, 58), (64, 76), (90, 100, 120), 2)
                
                # Mouth landmark
                mouth_y = 92 + (idx * 2)
                cv2.ellipse(canvas, (64, mouth_y), (16, 6 + idx), 0, 0, 180, (60, 60, 140), 2)
                
                # Add natural variations
                noise = np.random.normal(0, 5, canvas.shape).astype(np.int16)
                noisy_canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                out_path = user_dir / f"{user}_{s:04d}.jpg"
                cv2.imwrite(str(out_path), noisy_canvas)
                
            created[user] = 15

        return created
