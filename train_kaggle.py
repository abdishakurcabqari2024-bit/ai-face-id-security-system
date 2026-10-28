"""
Kaggle Dataset Training & Benchmarking Pipeline (31 Classes, 2,557 Images)
Lead AI ML Engineering - Full Industry-Standard Evaluation
Fits Random Forest & XGBoost on 31 Classes, Benchmarks Test Split, Generates Metric Visualizations.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

import config
from dataset import BiometricDatasetManager
from models import FaceIDDualModel

def plot_benchmarks(metrics: dict, save_path: str):
    """
    Generates comparison bar chart of Random Forest vs XGBoost vs Ensemble.
    """
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    rf_vals = [
        metrics['random_forest']['accuracy'] * 100,
        metrics['random_forest']['precision'] * 100,
        metrics['random_forest']['recall'] * 100,
        metrics['random_forest']['f1_score'] * 100
    ]
    xgb_vals = [
        metrics['xgboost']['accuracy'] * 100,
        metrics['xgboost']['precision'] * 100,
        metrics['xgboost']['recall'] * 100,
        metrics['xgboost']['f1_score'] * 100
    ]
    ens_vals = [
        metrics['ensemble']['accuracy'] * 100,
        metrics['ensemble']['precision'] * 100,
        metrics['ensemble']['recall'] * 100,
        metrics['ensemble']['f1_score'] * 100
    ]

    x = np.arange(len(labels))
    width = 0.25

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, rf_vals, width, label='Random Forest (200 Trees)', color='#38bdf8')
    rects2 = ax.bar(x, xgb_vals, width, label='XGBoost (Gradient Boosted)', color='#22c55e')
    rects3 = ax.bar(x + width, ens_vals, width, label='Soft-Voting Ensemble (Hybrid)', color='#f59e0b')

    ax.set_ylabel('Score Percentage (%)', fontsize=12)
    ax.set_title('AI Face ID Security System: Kaggle 31-Class Benchmark', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim([70, 105])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PLOT] Benchmark chart saved to: {save_path}")

def plot_confusion_matrix(cm, classes, save_path: str):
    """
    Renders high-resolution confusion matrix heatmap.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax, cbar=True)
    ax.set_title('Kaggle 31-Class Confusion Matrix (Soft-Voting Ensemble)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Identity', fontsize=11)
    ax.set_ylabel('True Identity', fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PLOT] Confusion matrix saved to: {save_path}")

def run_kaggle_pipeline():
    print("=" * 75)
    print("🚀 TRAINING AI FACE ID SYSTEM ON CLEANED KAGGLE DATASET (31 CLASSES)")
    print("=" * 75)

    dm = BiometricDatasetManager()

    # Step 1: Import profiles to faces/ for UI visibility
    print("[1/5] Importing 31 identity profiles to local directory...")
    imported = dm.import_kaggle_profiles_to_faces(samples_per_user=15)
    print(f" [+] Profile images imported/synchronized: {imported}")

    # Step 2: Load Train and Test Sets
    print("\n[2/5] Loading training set (2,045 images)...")
    t0 = time.time()
    X_train, y_train = dm.load_kaggle_dataset(split="train", use_cache=True)
    print(f" [+] Train features ready: shape={X_train.shape}, labels={len(y_train)} (Elapsed: {time.time()-t0:.2f}s)")

    print("\n[3/5] Loading test set (256 images)...")
    t0 = time.time()
    X_test, y_test = dm.load_kaggle_dataset(split="test", use_cache=True)
    print(f" [+] Test features ready: shape={X_test.shape}, labels={len(y_test)} (Elapsed: {time.time()-t0:.2f}s)")

    # Step 3: Label Encoding
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    classes = list(label_encoder.classes_)
    num_classes = len(classes)
    print(f" [+] Number of unique classes: {num_classes}")

    # Step 4: Fit Models
    dual_model = FaceIDDualModel()
    dual_model.classes_ = classes
    dual_model.label_encoder = label_encoder
    dual_model.build_models(num_classes)

    print("\n[4/5] Training Random Forest Classifier (200 trees, parallel)...")
    rf_start = time.perf_counter()
    dual_model.rf_model.fit(X_train, y_train_enc)
    rf_train_time = time.perf_counter() - rf_start
    print(f" [+] Random Forest fitted in {rf_train_time:.2f} seconds.")

    print("\nFitting XGBoost Classifier (Gradient Boosting with L1/L2 Regularization)...")
    xgb_start = time.perf_counter()
    dual_model.xgb_model.fit(X_train, y_train_enc)
    xgb_train_time = time.perf_counter() - xgb_start
    print(f" [+] XGBoost fitted in {xgb_train_time:.2f} seconds.")

    # Step 5: Test Set Predictions & Probabilities
    print("\n[5/5] Evaluating on Out-Of-Sample Test Set (256 images)...")
    rf_proba = dual_model.rf_model.predict_proba(X_test)
    xgb_proba = dual_model.xgb_model.predict_proba(X_test)

    # Soft-Voting Probability Fusion
    ens_proba = (dual_model.weight_xgb * xgb_proba) + (dual_model.weight_rf * rf_proba)

    rf_preds = np.argmax(rf_proba, axis=1)
    xgb_preds = np.argmax(xgb_proba, axis=1)
    ens_preds = np.argmax(ens_proba, axis=1)

    # Latency Benchmarks
    sample = X_test[0:1]
    t_rf = time.perf_counter()
    for _ in range(50):
        _ = dual_model.rf_model.predict_proba(sample)
    rf_lat = ((time.perf_counter() - t_rf) / 50) * 1000

    t_xgb = time.perf_counter()
    for _ in range(50):
        _ = dual_model.xgb_model.predict_proba(sample)
    xgb_lat = ((time.perf_counter() - t_xgb) / 50) * 1000

    # Top-3 Accuracy for Ensemble
    top3_correct = 0
    for i in range(len(y_test_enc)):
        top3_indices = np.argsort(ens_proba[i])[-3:]
        if y_test_enc[i] in top3_indices:
            top3_correct += 1
    top3_acc = top3_correct / len(y_test_enc)

    metrics = {
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        "num_classes": num_classes,
        "classes": classes,
        "random_forest": {
            "accuracy": float(accuracy_score(y_test_enc, rf_preds)),
            "precision": float(precision_score(y_test_enc, rf_preds, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test_enc, rf_preds, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_test_enc, rf_preds, average='weighted', zero_division=0)),
            "train_time_sec": round(rf_train_time, 2),
            "inference_latency_ms": round(rf_lat, 2)
        },
        "xgboost": {
            "accuracy": float(accuracy_score(y_test_enc, xgb_preds)),
            "precision": float(precision_score(y_test_enc, xgb_preds, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test_enc, xgb_preds, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_test_enc, xgb_preds, average='weighted', zero_division=0)),
            "train_time_sec": round(xgb_train_time, 2),
            "inference_latency_ms": round(xgb_lat, 2)
        },
        "ensemble": {
            "accuracy": float(accuracy_score(y_test_enc, ens_preds)),
            "top3_accuracy": float(top3_acc),
            "precision": float(precision_score(y_test_enc, ens_preds, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test_enc, ens_preds, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_test_enc, ens_preds, average='weighted', zero_division=0)),
            "inference_latency_ms": round(rf_lat + xgb_lat, 2)
        }
    }

    # Print Summary Table
    print("\n" + "=" * 75)
    print("📊 KAGGLE 31-CLASS BENCHMARK REPORT")
    print("=" * 75)
    print(f"{'Metric':<25} | {'Random Forest':<15} | {'XGBoost':<15} | {'Soft-Voting Ensemble':<20}")
    print("-" * 75)
    print(f"{'Accuracy (Top-1)':<25} | {metrics['random_forest']['accuracy']*100:>13.2f}% | {metrics['xgboost']['accuracy']*100:>13.2f}% | {metrics['ensemble']['accuracy']*100:>18.2f}%")
    print(f"{'Accuracy (Top-3)':<25} | {'--':>14} | {'--':>14} | {top3_acc*100:>18.2f}%")
    print(f"{'Precision (Weighted)':<25} | {metrics['random_forest']['precision']:>14.4f} | {metrics['xgboost']['precision']:>14.4f} | {metrics['ensemble']['precision']:>19.4f}")
    print(f"{'Recall (Weighted)':<25} | {metrics['random_forest']['recall']:>14.4f} | {metrics['xgboost']['recall']:>14.4f} | {metrics['ensemble']['recall']:>19.4f}")
    print(f"{'F1-Score':<25} | {metrics['random_forest']['f1_score']:>14.4f} | {metrics['xgboost']['f1_score']:>14.4f} | {metrics['ensemble']['f1_score']:>19.4f}")
    print(f"{'Inference Latency':<25} | {metrics['random_forest']['inference_latency_ms']:>11.2f} ms | {metrics['xgboost']['inference_latency_ms']:>11.2f} ms | {metrics['ensemble']['inference_latency_ms']:>16.2f} ms")
    print(f"{'Training Duration':<25} | {metrics['random_forest']['train_time_sec']:>12.2f} s | {metrics['xgboost']['train_time_sec']:>12.2f} s | {'--':>19}")
    print("=" * 75)

    # Save Models
    dual_model.is_trained = True
    dual_model.last_metrics = metrics
    dual_model.save_models()
    print("\n[OK] Trained 31-class models successfully saved to 'saved_models/'!")

    # Generate Visualizations
    cm = confusion_matrix(y_test_enc, ens_preds)
    plot_benchmarks(metrics, str(config.LOGS_DIR / "kaggle_evaluation_chart.png"))
    plot_confusion_matrix(cm, classes, str(config.LOGS_DIR / "kaggle_confusion_matrix.png"))

    return metrics

if __name__ == "__main__":
    run_kaggle_pipeline()
