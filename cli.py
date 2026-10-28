"""
Command-Line Interface (CLI) - AI Face ID Security System
Lead AI ML Engineering - Headless Automation & Verification
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import config
from dataset import BiometricDatasetManager
from models import FaceIDDualModel
from security_logger import SecurityLogger

def cmd_demo():
    print("=" * 60)
    print("[+] GENERATING DEMO PROFILES & SYNTHETIC DATASET")
    print("=" * 60)
    dm = BiometricDatasetManager()
    created = dm.generate_demo_dataset()
    for user, count in created.items():
        print(f" [+] Profile '{user}': {count} facial samples created.")
    
    print("\nTraining Dual Model (XGBoost + Random Forest)...")
    X, y = dm.load_dataset(apply_augmentation=True)
    print(f" [+] Augmented Dataset: {X.shape[0]} samples, {X.shape[1]} features.")
    
    dual_model = FaceIDDualModel()
    metrics = dual_model.train(X, y)
    
    print("\n--- PERFORMANCE EVALUATION BENCHMARK ---")
    print(f"Random Forest Accuracy: {metrics['random_forest']['accuracy']*100:.2f}% | Latency: {metrics['random_forest']['inference_latency_ms']} ms")
    print(f"XGBoost Accuracy:       {metrics['xgboost']['accuracy']*100:.2f}% | Latency: {metrics['xgboost']['inference_latency_ms']} ms")
    print(f"Ensemble Accuracy:      {metrics['ensemble']['accuracy']*100:.2f}% | Latency: {metrics['ensemble']['inference_latency_ms']} ms")
    print("\n[OK] Models successfully trained and saved to 'saved_models/'!")

def cmd_train():
    print("=" * 60)
    print("[*] TRAINING DUAL MODEL PIPELINE (XGBoost + Random Forest)")
    print("=" * 60)
    dm = BiometricDatasetManager()
    users = dm.get_enrolled_users()
    if len(users) < 2:
        print(f"[-] Error: Found {len(users)} user(s). Need at least 2 distinct user profiles to train.")
        return

    print(f"[+] Found {len(users)} user profiles: {users}")
    print("[+] Extracting features with HOG, LBP, and spatial moments...")
    X, y = dm.load_dataset(apply_augmentation=True)
    print(f"[+] Loaded {len(y)} samples. Fitting models...")

    dual_model = FaceIDDualModel()
    metrics = dual_model.train(X, y)

    print("\n--- RESULTS ---")
    print(f"RF F1-Score:       {metrics['random_forest']['f1_score']:.4f}")
    print(f"XGBoost F1-Score:  {metrics['xgboost']['f1_score']:.4f}")
    print(f"Ensemble Accuracy: {metrics['ensemble']['accuracy']*100:.2f}%")
    print("[OK] Model weights updated successfully.")

def cmd_list():
    dm = BiometricDatasetManager()
    users = dm.get_enrolled_users()
    print("=" * 60)
    print(f"[PROFILES] ENROLLED BIOMETRIC PROFILES (Total: {len(users)})")
    print("=" * 60)
    for u in users:
        u_dir = config.FACES_DIR / u
        c = len(list(u_dir.glob("*.jpg")))
        print(f"  * {u:<25} ({c} images)")

def cmd_logs():
    logger = SecurityLogger()
    logs = logger.get_recent_logs(limit=20)
    print("=" * 75)
    print("[SECURITY] RECENT ACCESS AUDIT LOGS")
    print("=" * 75)
    header = f"{'ID':<4} | {'Timestamp':<19} | {'User':<16} | {'Status':<16} | {'Conf':<6}"
    print(header)
    print("-" * 75)
    for r in logs:
        print(f"{r['id']:<4} | {r['timestamp']:<19} | {r['user_name']:<16} | {r['status']:<16} | {r['confidence']*100:.1f}%")

def cmd_train_kaggle():
    import train_kaggle
    train_kaggle.run_kaggle_pipeline()

def main():
    parser = argparse.ArgumentParser(description="AI Face ID Security System CLI")
    parser.add_argument("command", choices=["demo", "train", "train-kaggle", "list", "logs"], help="Command to execute")
    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo()
    elif args.command == "train":
        cmd_train()
    elif args.command == "train-kaggle":
        cmd_train_kaggle()
    elif args.command == "list":
        cmd_list()
    elif args.command == "logs":
        cmd_logs()

if __name__ == "__main__":
    main()
