"""
Security Audit Logger Module
Lead AI ML Engineering - Forensic & Compliance Logging
Maintains SQLite database and CSV logs of all access attempts, intruder captures,
and biometric verification confidence scores.
"""

import os
import csv
import sqlite3
import datetime
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

import config

class SecurityLogger:
    """
    Handles audit trails, forensic snapshot storage, and security metrics logging.
    """

    def __init__(self, db_path: Path = config.DB_PATH, csv_path: Path = config.CSV_LOG_PATH):
        self.db_path = db_path
        self.csv_path = csv_path
        self.intruder_dir = config.INTRUDER_DIR
        self._init_db()
        self._init_csv()

    def _init_db(self):
        """
        Creates SQLite table for security audit if not exists.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_name TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                rf_confidence REAL NOT NULL,
                xgb_confidence REAL NOT NULL,
                liveness_score REAL NOT NULL,
                liveness_verified INTEGER NOT NULL,
                snapshot_path TEXT,
                notes TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _init_csv(self):
        """
        Initializes CSV audit log with headers if not present.
        """
        if not self.csv_path.exists():
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "User_Name", "Status", "Ensemble_Confidence",
                    "RF_Confidence", "XGB_Confidence", "Liveness_Score",
                    "Liveness_Verified", "Snapshot_Path", "Notes"
                ])

    def log_attempt(self,
                    user_name: str,
                    status: str,
                    confidence: float,
                    rf_confidence: float,
                    xgb_confidence: float,
                    liveness_score: float,
                    liveness_verified: bool,
                    face_image: Optional[np.ndarray] = None,
                    notes: str = "") -> Dict[str, Any]:
        """
        Logs an access event to SQLite and CSV.
        If access was DENIED or SPOOF detected, automatically saves intruder photo!
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshot_filename = ""

        # Capture forensic snapshot if unauthorized or suspicious
        if face_image is not None and ("DENIED" in status or "SPOOF" in status):
            clean_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snapshot_filename = f"intruder_{clean_ts}.jpg"
            full_path = self.intruder_dir / snapshot_filename
            cv2.imwrite(str(full_path), face_image)

        # Insert into SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO access_logs (
                timestamp, user_name, status, confidence,
                rf_confidence, xgb_confidence, liveness_score,
                liveness_verified, snapshot_path, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, user_name, status, round(confidence, 4),
            round(rf_confidence, 4), round(xgb_confidence, 4),
            round(liveness_score, 4), 1 if liveness_verified else 0,
            snapshot_filename, notes
        ))
        conn.commit()
        conn.close()

        # Append to CSV
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, user_name, status, f"{confidence:.4f}",
                f"{rf_confidence:.4f}", f"{xgb_confidence:.4f}",
                f"{liveness_score:.4f}", "YES" if liveness_verified else "NO",
                snapshot_filename, notes
            ])

        return {
            "timestamp": timestamp,
            "user_name": user_name,
            "status": status,
            "snapshot_path": snapshot_filename
        }

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent access log entries.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM access_logs ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns security statistics: Total attempts, Granted, Denied, Spoofs.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM access_logs")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE status LIKE '%GRANTED%'")
        granted = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE status LIKE '%DENIED%'")
        denied = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE status LIKE '%SPOOF%'")
        spoofs = cursor.fetchone()[0]

        conn.close()
        return {
            "total_attempts": total,
            "access_granted": granted,
            "access_denied": denied,
            "spoof_attempts": spoofs
        }
