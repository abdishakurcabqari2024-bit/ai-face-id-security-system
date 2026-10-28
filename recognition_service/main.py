"""
Standalone Biometric Recognition Microservice
Lead AI ML Engineering - FastAPI Service
Hosts OpenCV Face Detection, Multi-Scale HOG+LBP Feature Extraction,
Laplacian Anti-Spoofing, and XGBoost + Random Forest Dual-Model Inference.
Deployable on Render, Railway, Fly.io, or Hugging Face Spaces.
"""

import os
import sys
import base64
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root directory is in sys.path to access models and core algorithms
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from detector import FaceDetector
from liveness import LivenessDetector
from feature_extractor import BiometricFeatureExtractor
from models import FaceIDDualModel

app = FastAPI(
    title="AI Face ID Biometric Recognition Engine",
    description="Microservice hosting XGBoost & Random Forest Face Recognition for Enterprise Security",
    version="2.0.0"
)

# Enable CORS for Next.js frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Biometric Engines
detector = FaceDetector()
liveness = LivenessDetector()
extractor = BiometricFeatureExtractor()
dual_model = FaceIDDualModel()

# Attempt loading pre-trained models
dual_model.load_models()

RECOGNITION_SECRET_KEY = os.getenv("RECOGNITION_SECRET_KEY", "dev_secret_biometric_key_12345")

def verify_token(authorization: Optional[str] = Header(None)):
    """Verifies bearer token if set in production."""
    if os.getenv("REQUIRE_AUTH", "false").lower() == "true":
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        token = authorization.split(" ")[1]
        if token != RECOGNITION_SECRET_KEY:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid biometric service secret key")
    return True

# --- Request / Response Schemas ---
class VerifyPayload(BaseModel):
    image: str = Field(..., description="Base64 encoded JPEG or PNG frame")
    threshold: Optional[float] = Field(0.75, description="Security confidence threshold (0.0 to 1.0)")

class EnrollPayload(BaseModel):
    user_name: str = Field(..., description="Unique identity/user label")
    images: List[str] = Field(..., description="Array of base64 face samples")

def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decodes dataURL or raw base64 string into BGR OpenCV image."""
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image buffer.")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

# --- Endpoints ---
@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "service": "AI Face ID Biometric Recognition Engine",
        "models_trained": dual_model.is_trained,
        "classes_count": len(dual_model.classes_),
        "classes": dual_model.classes_[:10], # preview first 10
        "security_threshold": dual_model.security_threshold,
        "feature_dimensions": 1858
    }

@app.post("/api/v1/verify")
def verify_face(payload: VerifyPayload, authorized: bool = Depends(verify_token)):
    t0 = time.perf_counter()
    frame = decode_base64_image(payload.image)

    # 1. Face Detection
    faces = detector.detect_faces(frame)
    if not faces:
        # Check if the submitted image was already a cropped face
        h, w = frame.shape[:2]
        if 80 <= h <= 600 and 80 <= w <= 600:
            face_crop = frame
            x, y = 0, 0
        else:
            return {
                "status": "NO_FACE_DETECTED",
                "identity": "No Face Detected",
                "access_granted": False,
                "confidence": 0.0,
                "rf_confidence": 0.0,
                "xgb_confidence": 0.0,
                "rf_identity": "None",
                "xgb_identity": "None",
                "models_agree": False,
                "liveness": {"is_live": False, "score": 0.0, "message": "No face found in camera frame"},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)
            }
    else:
        x, y, w, h, face_crop = faces[0]

    # 2. Anti-Spoofing & Liveness Detection
    is_live, live_score, live_msg = liveness.verify_liveness(face_crop)
    if config.REQUIRE_LIVENESS and not is_live:
        return {
            "status": "SPOOF_ATTACK",
            "identity": "PHOTO / SPOOF ATTACK",
            "access_granted": False,
            "confidence": 0.0,
            "rf_confidence": 0.0,
            "xgb_confidence": 0.0,
            "rf_identity": "None",
            "xgb_identity": "None",
            "models_agree": False,
            "liveness": {"is_live": False, "score": round(live_score, 3), "message": live_msg},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)
        }

    # 3. Extract 1,858-dim Biometric Feature Vector
    feat = extractor.extract_features(face_crop)

    # 4. Run XGBoost + Random Forest Ensemble
    orig_thresh = dual_model.security_threshold
    if payload.threshold is not None:
        dual_model.security_threshold = payload.threshold

    pred = dual_model.predict(feat)
    dual_model.security_threshold = orig_thresh

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "status": pred["status"],
        "identity": pred["identity"],
        "access_granted": pred["access_granted"],
        "confidence": round(pred["confidence"], 4),
        "rf_confidence": round(pred["rf_confidence"], 4),
        "xgb_confidence": round(pred["xgb_confidence"], 4),
        "rf_identity": pred["rf_identity"],
        "xgb_identity": pred["xgb_identity"],
        "models_agree": pred["models_agree"],
        "liveness": {
            "is_live": True,
            "score": round(live_score, 3),
            "message": live_msg
        },
        "latency_ms": latency_ms
    }

@app.post("/api/v1/enroll")
def enroll_user(payload: EnrollPayload, authorized: bool = Depends(verify_token)):
    user_name = payload.user_name.strip()
    if not user_name:
        raise HTTPException(status_code=400, detail="Invalid user_name")

    saved_count = 0
    for idx, img_b64 in enumerate(payload.images):
        try:
            frame = decode_base64_image(img_b64)
            faces = detector.detect_faces(frame)
            crop = faces[0][4] if faces else frame
            user_dir = config.FACES_DIR / user_name
            user_dir.mkdir(parents=True, exist_ok=True)
            out_file = user_dir / f"{user_name}_web_{int(time.time())}_{idx}.jpg"
            cv2.imwrite(str(out_file), cv2.resize(crop, (config.FACE_WIDTH, config.FACE_HEIGHT)))
            saved_count += 1
        except Exception as e:
            print(f"Error enrolling frame {idx}: {e}")

    return {
        "success": True,
        "message": f"Saved {saved_count} face samples for {user_name}.",
        "samples_enrolled": saved_count
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
