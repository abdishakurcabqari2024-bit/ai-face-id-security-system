"""
Core Runtime Engine: Real-Time AI Face ID Verification
Lead AI ML Engineering - Real-Time Video HUD & Biometric Processing
Orchestrates Face Detection, Liveness Verification, Feature Extraction,
XGBoost & Random Forest Inference, and Dynamic HUD Overlays.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

import config
from detector import FaceDetector
from feature_extractor import BiometricFeatureExtractor
from liveness import LivenessDetector
from models import FaceIDDualModel
from security_logger import SecurityLogger

class FaceIDEngine:
    """
    Main computational engine processing live camera feeds or static image inputs.
    """

    def __init__(self,
                 detector: Optional[FaceDetector] = None,
                 extractor: Optional[BiometricFeatureExtractor] = None,
                 liveness: Optional[LivenessDetector] = None,
                 model: Optional[FaceIDDualModel] = None,
                 logger: Optional[SecurityLogger] = None):

        self.detector = detector or FaceDetector()
        self.extractor = extractor or BiometricFeatureExtractor()
        self.liveness = liveness or LivenessDetector()
        self.model = model or FaceIDDualModel()
        self.logger = logger or SecurityLogger()

        # Cooldown timer to prevent repetitive logging spam (seconds)
        self.log_cooldown = 4.0
        self.last_log_time = 0.0
        self.last_logged_identity = ""

        # Performance monitoring
        self.fps_tracker = []
        self.current_fps = 0.0

        # Try auto-loading trained models
        self.model.load_models()

    def process_frame(self, frame: np.ndarray, log_events: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Processes a single BGR frame:
        1. Detects all faces
        2. Checks liveness / anti-spoofing
        3. Extracts biometric features
        4. Predicts with Random Forest & XGBoost Ensemble
        5. Renders Futuristic Security HUD
        6. Logs security events with cooldown
        """
        t_start = time.perf_counter()
        annotated_frame = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        faces = self.detector.detect_faces(frame)
        detection_results = []
        current_time = time.time()

        for (x, y, w, h, face_crop) in faces:
            # 1. Anti-Spoofing & Liveness Check
            is_live, live_score, live_msg = self.liveness.verify_liveness(face_crop)

            result_entry = {
                "box": (x, y, w, h),
                "is_live": is_live,
                "liveness_score": live_score,
                "liveness_msg": live_msg,
                "status": "PROCESSING",
                "identity": "Unknown",
                "confidence": 0.0,
                "rf_conf": 0.0,
                "xgb_conf": 0.0,
                "access_granted": False
            }

            if config.REQUIRE_LIVENESS and not is_live:
                # Spoof / Attack Detected!
                result_entry["status"] = "SPOOF_ATTACK"
                result_entry["identity"] = "PHOTO / SPOOF"
                self._draw_hud(annotated_frame, x, y, w, h, result_entry, color=(0, 140, 255)) # Orange

                if log_events and (current_time - self.last_log_time > self.log_cooldown):
                    self.logger.log_attempt(
                        user_name="SPOOF_ATTACK",
                        status="SPOOF_ATTACK_DENIED",
                        confidence=0.0,
                        rf_confidence=0.0,
                        xgb_confidence=0.0,
                        liveness_score=live_score,
                        liveness_verified=False,
                        face_image=face_crop,
                        notes=live_msg
                    )
                    self.last_log_time = current_time

            else:
                # 2. Extract Biometric Features
                feat = self.extractor.extract_features(face_crop)

                # 3. Predict with Dual Model (XGBoost + Random Forest)
                pred = self.model.predict(feat)

                result_entry.update({
                    "status": pred["status"],
                    "identity": pred["identity"],
                    "access_granted": pred["access_granted"],
                    "confidence": pred["confidence"],
                    "rf_conf": pred["rf_confidence"],
                    "xgb_conf": pred["xgb_confidence"],
                    "rf_identity": pred["rf_identity"],
                    "xgb_identity": pred["xgb_identity"],
                    "models_agree": pred["models_agree"]
                })

                # Determine HUD color
                if pred["access_granted"]:
                    color = (0, 255, 0) # Green for ACCESS GRANTED
                else:
                    color = (0, 0, 255) # Red for ACCESS DENIED / INTRUDER

                self._draw_hud(annotated_frame, x, y, w, h, result_entry, color=color)

                # 4. Security Audit Logging
                if log_events and self.model.is_trained:
                    cooldown_passed = (current_time - self.last_log_time > self.log_cooldown)
                    identity_changed = (pred["identity"] != self.last_logged_identity)

                    if cooldown_passed or identity_changed:
                        self.logger.log_attempt(
                            user_name=pred.get("candidate_identity", "UNKNOWN"),
                            status="ACCESS_GRANTED" if pred["access_granted"] else "ACCESS_DENIED",
                            confidence=pred["confidence"],
                            rf_confidence=pred["rf_confidence"],
                            xgb_confidence=pred["xgb_confidence"],
                            liveness_score=live_score,
                            liveness_verified=True,
                            face_image=face_crop if not pred["access_granted"] else None,
                            notes=f"Models Agree: {pred['models_agree']}"
                        )
                        self.last_log_time = current_time
                        self.last_logged_identity = pred["identity"]

            detection_results.append(result_entry)

        # FPS calculation
        t_delta = time.perf_counter() - t_start
        fps = 1.0 / max(t_delta, 1e-4)
        self.fps_tracker.append(fps)
        if len(self.fps_tracker) > 15:
            self.fps_tracker.pop(0)
        self.current_fps = float(np.mean(self.fps_tracker))

        # Draw Global HUD Header
        self._draw_header(annotated_frame, len(faces))

        return annotated_frame, detection_results

    def _draw_hud(self, frame: np.ndarray, x: int, y: int, w: int, h: int, result: Dict[str, Any], color: Tuple[int, int, int]):
        """
        Draws high-tech bounding box corners, identity tag, and dual-model confidence meters.
        """
        corner_len = int(min(w, h) * 0.25)
        thickness = 2

        # 4 Corner brackets (Futuristic Scanner Appearance)
        # Top-left
        cv2.line(frame, (x, y), (x + corner_len, y), color, thickness + 1)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thickness + 1)
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness + 1)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness + 1)
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness + 1)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness + 1)
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness + 1)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness + 1)

        # Main bounding box (semi-transparent)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        # Header Badge
        status_label = "ACCESS GRANTED" if result.get("access_granted") else ("SPOOF REJECTED" if "SPOOF" in result.get("status", "") else "ACCESS DENIED")
        identity = result.get("identity", "Unknown")
        conf = result.get("confidence", 0.0) * 100

        badge_text = f"[{status_label}] {identity}"
        sub_text = f"Ensemble: {conf:.1f}% | XGB: {result.get('xgb_conf', 0)*100:.0f}% | RF: {result.get('rf_conf', 0)*100:.0f}%"

        # Background badge for text
        cv2.rectangle(frame, (x, max(0, y - 48)), (x + max(w, 240), y), (20, 20, 25), -1)
        cv2.rectangle(frame, (x, max(0, y - 48)), (x + max(w, 240), y), color, 1)

        # Text strings
        cv2.putText(frame, badge_text, (x + 6, max(16, y - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
        cv2.putText(frame, sub_text, (x + 6, max(32, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

    def _draw_header(self, frame: np.ndarray, face_count: int):
        """
        Draws top status bar with FPS and model readiness.
        """
        # Top banner background
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (15, 18, 25), -1)
        cv2.line(frame, (0, 36), (frame.shape[1], 36), (60, 80, 100), 1)

        status_text = "AI FACE ID SECURITY SYSTEM | XGBoost & Random Forest Engine"
        model_state = "MODELS: TRAINED & ACTIVE" if self.model.is_trained else "MODELS: UNTRAINED (ENROLL USERS)"
        meta_text = f"FPS: {self.current_fps:.1f} | Faces: {face_count} | {model_state}"

        cv2.putText(frame, status_text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 225, 255), 1)
        cv2.putText(frame, meta_text, (frame.shape[1] - 460, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
