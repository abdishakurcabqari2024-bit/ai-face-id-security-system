"""
Production GUI Dashboard - AI Face ID Security System
Lead AI ML Engineering - Modern Responsive Desktop Application
Features:
- Live Webcam Face Scanner with Futuristic HUD Overlays
- Real-time Ensemble Decision Metrics (XGBoost vs Random Forest)
- User Enrollment with Automated Guided Capture & Augmentation
- Model Training & Evaluation Analytics Dashboard
- Forensic Security Logs & Intruder Snapshot Viewer
- Dynamic Security Threshold & Anti-Spoofing Configuration
"""

import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from typing import Optional

import config
from face_id_engine import FaceIDEngine
from dataset import BiometricDatasetManager
from models import FaceIDDualModel
from security_logger import SecurityLogger

class FaceIDSecurityApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Face ID Biometric Security System | XGBoost & Random Forest")
        self.root.geometry("1240x820")
        self.root.minsize(1080, 720)
        self.root.configure(bg="#0f141c")

        # Core Engines
        self.dataset_manager = BiometricDatasetManager()
        self.security_logger = SecurityLogger()
        self.engine = FaceIDEngine(logger=self.security_logger)

        # Video Streaming State
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_scanning = False
        self.is_enrolling = False
        self.enroll_target_user = ""
        self.enroll_samples_target = 30
        self.enroll_samples_count = 0
        self.last_capture_time = 0.0

        # UI Styling Setup
        self._setup_styles()
        self._build_header()
        self._build_notebook()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Status Update Loop
        self._update_status_bar()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Notebook tabs styling
        style.configure("TNotebook", background="#0f141c", borderwidth=0)
        style.configure("TNotebook.Tab", background="#182230", foreground="#94a3b8",
                        padding=[18, 9], font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", "#0284c7")],
                  foreground=[("selected", "#ffffff")])

        # Treeview styling for logs
        style.configure("Treeview",
                        background="#131c27",
                        foreground="#e2e8f0",
                        fieldbackground="#131c27",
                        rowheight=26,
                        font=("Segoe UI", 9))
        style.configure("Treeview.Heading",
                        background="#1e293b",
                        foreground="#38bdf8",
                        font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#0369a1")])

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg="#131d2a", height=70, padx=20, pady=10)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        # Logo and Title
        title_box = tk.Frame(header_frame, bg="#131d2a")
        title_box.pack(side=tk.LEFT)

        title_lbl = tk.Label(
            title_box,
            text="🛡️ AI FACE ID BIOMETRIC SECURITY SYSTEM",
            font=("Segoe UI", 15, "bold"),
            fg="#38bdf8",
            bg="#131d2a"
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            title_box,
            text="Dual ML Architecture: Extreme Gradient Boosting (XGBoost) + Random Forest Classifier",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#131d2a"
        )
        subtitle_lbl.pack(anchor="w")

        # Top Right Status Badges
        status_box = tk.Frame(header_frame, bg="#131d2a")
        status_box.pack(side=tk.RIGHT)

        self.model_status_badge = tk.Label(
            status_box,
            text="● MODEL: " + ("ACTIVE" if self.engine.model.is_trained else "UNTRAINED"),
            font=("Segoe UI", 9, "bold"),
            fg="#22c55e" if self.engine.model.is_trained else "#ef4444",
            bg="#1e293b",
            padx=12,
            pady=4,
            relief=tk.FLAT
        )
        self.model_status_badge.pack(side=tk.RIGHT, padx=6)

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        # Tab 1: Live Face ID Scanner
        self.tab_scanner = tk.Frame(self.notebook, bg="#0f141c")
        self.notebook.add(self.tab_scanner, text="  👁️ Live Scanner  ")
        self._build_scanner_tab()

        # Tab 2: User Enrollment
        self.tab_enrollment = tk.Frame(self.notebook, bg="#0f141c")
        self.notebook.add(self.tab_enrollment, text="  👤 User Enrollment  ")
        self._build_enrollment_tab()

        # Tab 3: Model Training & Evaluation Analytics
        self.tab_analytics = tk.Frame(self.notebook, bg="#0f141c")
        self.notebook.add(self.tab_analytics, text="  🧠 Dual ML Training  ")
        self._build_analytics_tab()

        # Tab 4: Security Audit Trail & Intruder Logs
        self.tab_logs = tk.Frame(self.notebook, bg="#0f141c")
        self.notebook.add(self.tab_logs, text="  📋 Security Audit Logs  ")
        self._build_logs_tab()

        # Tab 5: Settings & Thresholds
        self.tab_settings = tk.Frame(self.notebook, bg="#0f141c")
        self.notebook.add(self.tab_settings, text="  ⚙️ Security Settings  ")
        self._build_settings_tab()

    # -------------------------------------------------------------
    # TAB 1: LIVE FACE ID SCANNER
    # -------------------------------------------------------------
    def _build_scanner_tab(self):
        container = tk.Frame(self.tab_scanner, bg="#0f141c")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Column: Video Feed
        video_col = tk.Frame(container, bg="#131c27", relief=tk.RIDGE, bd=1)
        video_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Canvas for Camera Display
        self.video_canvas = tk.Canvas(video_col, bg="#0a0f16", highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Control Bar below video
        ctrl_bar = tk.Frame(video_col, bg="#131c27", pady=8, padx=10)
        ctrl_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_toggle_scan = tk.Button(
            ctrl_bar,
            text="▶ Start Live Scanner",
            font=("Segoe UI", 10, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=6,
            command=self._toggle_scanner
        )
        self.btn_toggle_scan.pack(side=tk.LEFT, padx=6)

        cam_lbl = tk.Label(ctrl_bar, text="Camera Index:", fg="#94a3b8", bg="#131c27", font=("Segoe UI", 9))
        cam_lbl.pack(side=tk.LEFT, padx=(16, 4))
        self.cam_index_spin = tk.Spinbox(ctrl_bar, from_=0, to=5, width=4, bg="#1e293b", fg="#ffffff", relief=tk.FLAT)
        self.cam_index_spin.pack(side=tk.LEFT)

        # Right Column: Decision Telemetry & Verification HUD
        telemetry_col = tk.Frame(container, bg="#131c27", width=380, relief=tk.RIDGE, bd=1, padx=16, pady=16)
        telemetry_col.pack(side=tk.RIGHT, fill=tk.Y)
        telemetry_col.pack_propagate(False)

        tk.Label(telemetry_col, text="VERIFICATION TELEMETRY", font=("Segoe UI", 11, "bold"), fg="#38bdf8", bg="#131c27").pack(anchor="w")

        # Security Status Card
        self.status_card = tk.Label(
            telemetry_col,
            text="STANDBY\nReady to Scan",
            font=("Segoe UI", 14, "bold"),
            fg="#94a3b8",
            bg="#1e293b",
            height=3,
            relief=tk.FLAT
        )
        self.status_card.pack(fill=tk.X, pady=(12, 16))

        # Identity Card
        id_frame = tk.Frame(telemetry_col, bg="#172230", padx=12, pady=10, relief=tk.FLAT)
        id_frame.pack(fill=tk.X, pady=6)
        tk.Label(id_frame, text="Recognized Identity:", fg="#94a3b8", bg="#172230", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.lbl_tele_identity = tk.Label(id_frame, text="-- None --", fg="#38bdf8", bg="#172230", font=("Segoe UI", 13, "bold"))
        self.lbl_tele_identity.pack(anchor="w", pady=(2, 0))

        # Ensemble Score
        ens_frame = tk.Frame(telemetry_col, bg="#172230", padx=12, pady=10, relief=tk.FLAT)
        ens_frame.pack(fill=tk.X, pady=6)
        tk.Label(ens_frame, text="Ensemble Confidence:", fg="#94a3b8", bg="#172230", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.lbl_tele_ens_conf = tk.Label(ens_frame, text="0.0%", fg="#ffffff", bg="#172230", font=("Segoe UI", 12, "bold"))
        self.lbl_tele_ens_conf.pack(anchor="w")
        self.prog_ens = ttk.Progressbar(ens_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.prog_ens.pack(fill=tk.X, pady=(4, 0))

        # XGBoost vs Random Forest Breakdown
        breakdown_frame = tk.Frame(telemetry_col, bg="#172230", padx=12, pady=10, relief=tk.FLAT)
        breakdown_frame.pack(fill=tk.X, pady=6)
        tk.Label(breakdown_frame, text="Dual Model Distribution:", fg="#94a3b8", bg="#172230", font=("Segoe UI", 8, "bold")).pack(anchor="w")

        # XGBoost
        self.lbl_tele_xgb = tk.Label(breakdown_frame, text="XGBoost: 0.0%", fg="#38bdf8", bg="#172230", font=("Segoe UI", 9))
        self.lbl_tele_xgb.pack(anchor="w", pady=(4, 1))
        self.prog_xgb = ttk.Progressbar(breakdown_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.prog_xgb.pack(fill=tk.X)

        # Random Forest
        self.lbl_tele_rf = tk.Label(breakdown_frame, text="Random Forest: 0.0%", fg="#38bdf8", bg="#172230", font=("Segoe UI", 9))
        self.lbl_tele_rf.pack(anchor="w", pady=(6, 1))
        self.prog_rf = ttk.Progressbar(breakdown_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.prog_rf.pack(fill=tk.X)

        # Liveness & Model Agreement
        meta_frame = tk.Frame(telemetry_col, bg="#172230", padx=12, pady=10, relief=tk.FLAT)
        meta_frame.pack(fill=tk.X, pady=6)
        self.lbl_tele_liveness = tk.Label(meta_frame, text="Anti-Spoofing: Verified", fg="#22c55e", bg="#172230", font=("Segoe UI", 9))
        self.lbl_tele_liveness.pack(anchor="w")
        self.lbl_tele_agreement = tk.Label(meta_frame, text="Models Agreement: Yes", fg="#94a3b8", bg="#172230", font=("Segoe UI", 9))
        self.lbl_tele_agreement.pack(anchor="w", pady=(4, 0))

    def _toggle_scanner(self):
        if not self.is_scanning:
            try:
                cam_idx = int(self.cam_index_spin.get())
            except ValueError:
                cam_idx = 0
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Try standard backend if CAP_DSHOW fails
                self.cap = cv2.VideoCapture(cam_idx)

            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", f"Unable to open camera index {cam_idx}. Check connection.")
                return

            self.is_scanning = True
            self.btn_toggle_scan.config(text="⏹ Stop Scanner", bg="#ef4444")
            self._scan_stream_loop()
        else:
            self.is_scanning = False
            self.btn_toggle_scan.config(text="▶ Start Live Scanner", bg="#0284c7")
            if self.cap:
                self.cap.release()
                self.cap = None
            self.status_card.config(text="SCANNER STOPPED", fg="#94a3b8", bg="#1e293b")
            self.video_canvas.delete("all")

    def _scan_stream_loop(self):
        if not self.is_scanning or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            # Process frame through FaceIDEngine
            annotated_frame, results = self.engine.process_frame(frame, log_events=True)

            # Update Telemetry UI
            if results:
                res = results[0]
                status = res.get("status", "")
                identity = res.get("identity", "Unknown")
                conf = res.get("confidence", 0.0) * 100
                xgb_conf = res.get("xgb_conf", 0.0) * 100
                rf_conf = res.get("rf_conf", 0.0) * 100
                is_granted = res.get("access_granted", False)
                models_agree = res.get("models_agree", False)
                is_live = res.get("is_live", True)

                if is_granted:
                    self.status_card.config(text=f"ACCESS GRANTED\n{identity}", fg="#22c55e", bg="#064e3b")
                elif "SPOOF" in status:
                    self.status_card.config(text="SPOOF ATTACK DETECTED\nPhoto / Screen Rejected", fg="#f97316", bg="#431407")
                else:
                    self.status_card.config(text=f"ACCESS DENIED\nUnauthorized / Unknown", fg="#ef4444", bg="#450a0a")

                self.lbl_tele_identity.config(text=identity)
                self.lbl_tele_ens_conf.config(text=f"{conf:.1f}%")
                self.prog_ens['value'] = conf

                self.lbl_tele_xgb.config(text=f"XGBoost: {xgb_conf:.1f}%")
                self.prog_xgb['value'] = xgb_conf

                self.lbl_tele_rf.config(text=f"Random Forest: {rf_conf:.1f}%")
                self.prog_rf['value'] = rf_conf

                self.lbl_tele_liveness.config(
                    text=f"Anti-Spoofing: {'Verified 3D Face' if is_live else 'REJECTED (Spoof)'}",
                    fg="#22c55e" if is_live else "#ef4444"
                )
                self.lbl_tele_agreement.config(
                    text=f"Models Agreement: {'MATCH (RF == XGB)' if models_agree else 'DIVERGENT'}",
                    fg="#22c55e" if models_agree else "#f59e0b"
                )
            else:
                self.status_card.config(text="SEARCHING...\nAlign Face to Camera", fg="#94a3b8", bg="#1e293b")
                self.lbl_tele_identity.config(text="-- None --")
                self.lbl_tele_ens_conf.config(text="0.0%")
                self.prog_ens['value'] = 0

            # Render to Canvas
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            canvas_w = self.video_canvas.winfo_width()
            canvas_h = self.video_canvas.winfo_height()

            if canvas_w > 10 and canvas_h > 10:
                # Maintain aspect ratio
                h, w = rgb_frame.shape[:2]
                scale = min(canvas_w / w, canvas_h / h)
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(rgb_frame, (nw, nh))

                img = Image.fromarray(resized)
                self.tk_img = ImageTk.PhotoImage(image=img)
                self.video_canvas.delete("all")
                self.video_canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=self.tk_img)

        # Loop after 25ms (~40 FPS target)
        self.root.after(25, self._scan_stream_loop)

    # -------------------------------------------------------------
    # TAB 2: USER ENROLLMENT
    # -------------------------------------------------------------
    def _build_enrollment_tab(self):
        container = tk.Frame(self.tab_enrollment, bg="#0f141c")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left Column: Enrollment Form & Capture Preview
        left_box = tk.Frame(container, bg="#131c27", relief=tk.RIDGE, bd=1, padx=20, pady=20)
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(left_box, text="ENROLL NEW USER", font=("Segoe UI", 12, "bold"), fg="#38bdf8", bg="#131c27").pack(anchor="w")

        # Form fields
        form_grid = tk.Frame(left_box, bg="#131c27", pady=10)
        form_grid.pack(fill=tk.X)

        tk.Label(form_grid, text="Full Name / User ID:", fg="#cbd5e1", bg="#131c27", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", pady=6)
        self.ent_user_name = tk.Entry(form_grid, font=("Segoe UI", 10), bg="#1e293b", fg="#ffffff", insertbackground="#ffffff", width=28)
        self.ent_user_name.grid(row=0, column=1, sticky="w", padx=10, pady=6)

        tk.Label(form_grid, text="Target Face Samples:", fg="#cbd5e1", bg="#131c27", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=6)
        self.spin_samples = tk.Spinbox(form_grid, from_=10, to=100, increment=5, font=("Segoe UI", 10), bg="#1e293b", fg="#ffffff", width=8)
        self.spin_samples.delete(0, "end")
        self.spin_samples.insert(0, "30")
        self.spin_samples.grid(row=1, column=1, sticky="w", padx=10, pady=6)

        # Instructions Banner
        inst_text = "💡 Enrollment Guidance:\n- Position face in center of camera frame.\n- Slowly tilt head slightly (left, right, up, down).\n- Maintain natural expressions (smiling, neutral) to capture intra-class variance."
        tk.Label(left_box, text=inst_text, justify=tk.LEFT, fg="#94a3b8", bg="#1e293b", font=("Segoe UI", 9), padx=12, pady=8).pack(fill=tk.X, pady=10)

        # Enrollment Canvas preview
        self.enroll_canvas = tk.Canvas(left_box, bg="#0a0f16", height=280, highlightthickness=0)
        self.enroll_canvas.pack(fill=tk.BOTH, expand=True, pady=6)

        # Progress bar
        self.lbl_enroll_progress = tk.Label(left_box, text="Samples Captured: 0 / 30", font=("Segoe UI", 10, "bold"), fg="#38bdf8", bg="#131c27")
        self.lbl_enroll_progress.pack(pady=4)
        self.prog_enroll = ttk.Progressbar(left_box, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.prog_enroll.pack(fill=tk.X, pady=4)

        # Action Buttons
        btn_box = tk.Frame(left_box, bg="#131c27", pady=8)
        btn_box.pack(fill=tk.X)

        self.btn_start_enroll = tk.Button(
            btn_box,
            text="📸 Start Live Webcam Capture",
            font=("Segoe UI", 10, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=14,
            pady=6,
            command=self._start_enrollment
        )
        self.btn_start_enroll.pack(side=tk.LEFT, padx=(0, 10))

        # Right Column: Currently Registered Users List
        right_box = tk.Frame(container, bg="#131c27", width=360, relief=tk.RIDGE, bd=1, padx=16, pady=20)
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_box.pack_propagate(False)

        tk.Label(right_box, text="ENROLLED BIOMETRIC PROFILES", font=("Segoe UI", 11, "bold"), fg="#38bdf8", bg="#131c27").pack(anchor="w")

        self.lst_users = tk.Listbox(right_box, bg="#1a2432", fg="#f8fafc", font=("Segoe UI", 10), selectbackground="#0284c7", borderwidth=0)
        self.lst_users.pack(fill=tk.BOTH, expand=True, pady=10)

        refresh_btn = tk.Button(
            right_box,
            text="🔄 Refresh Profiles",
            font=("Segoe UI", 9),
            bg="#334155",
            fg="#ffffff",
            relief=tk.FLAT,
            pady=4,
            command=self._refresh_users_list
        )
        refresh_btn.pack(fill=tk.X, pady=4)

        import_kaggle_btn = tk.Button(
            right_box,
            text="📥 Sync 31 Kaggle Profiles",
            font=("Segoe UI", 9),
            bg="#0369a1",
            fg="#ffffff",
            relief=tk.FLAT,
            pady=4,
            command=self._import_kaggle_profiles
        )
        import_kaggle_btn.pack(fill=tk.X, pady=4)

        del_btn = tk.Button(
            right_box,
            text="🗑️ Delete Selected Profile",
            font=("Segoe UI", 9),
            bg="#991b1b",
            fg="#ffffff",
            relief=tk.FLAT,
            pady=4,
            command=self._delete_selected_user
        )
        del_btn.pack(fill=tk.X, pady=4)

        self._refresh_users_list()

    def _start_enrollment(self):
        user_name = self.ent_user_name.get().strip()
        if not user_name:
            messagebox.showwarning("Validation Error", "Please enter a valid user name or ID.")
            return

        try:
            self.enroll_samples_target = int(self.spin_samples.get())
        except ValueError:
            self.enroll_samples_target = 30

        self.enroll_target_user = user_name
        self.enroll_samples_count = 0
        self.prog_enroll['value'] = 0
        self.is_enrolling = True

        # Open webcam if not already open
        if self.cap is None or not self.cap.isOpened():
            try:
                cam_idx = int(self.cam_index_spin.get())
            except ValueError:
                cam_idx = 0
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(cam_idx)

        self.btn_start_enroll.config(state=tk.DISABLED, text="Capturing...")
        self._enroll_loop()

    def _enroll_loop(self):
        if not self.is_enrolling:
            return

        ret, frame = self.cap.read() if self.cap else (False, None)
        if ret:
            faces = self.engine.detector.detect_faces(frame)
            current_t = time.time()

            # Render preview on enrollment canvas
            preview_frame = frame.copy()

            if faces:
                x, y, w, h, crop = faces[0]
                cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(preview_frame, f"Capturing: {self.enroll_samples_count}/{self.enroll_samples_target}",
                            (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Capture sample every 180ms to ensure frame diversity
                if current_t - self.last_capture_time > 0.18:
                    self.dataset_manager.save_face_sample(self.enroll_target_user, crop, self.enroll_samples_count)
                    self.enroll_samples_count += 1
                    self.last_capture_time = current_t

                    pct = (self.enroll_samples_count / self.enroll_samples_target) * 100
                    self.prog_enroll['value'] = pct
                    self.lbl_enroll_progress.config(
                        text=f"Samples Captured: {self.enroll_samples_count} / {self.enroll_samples_target}"
                    )

            # Display on Canvas
            rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            cw = self.enroll_canvas.winfo_width()
            ch = self.enroll_canvas.winfo_height()
            if cw > 10 and ch > 10:
                h_img, w_img = rgb.shape[:2]
                scale = min(cw / w_img, ch / h_img)
                resized = cv2.resize(rgb, (int(w_img * scale), int(h_img * scale)))
                img = Image.fromarray(resized)
                self.enroll_tk_img = ImageTk.PhotoImage(image=img)
                self.enroll_canvas.delete("all")
                self.enroll_canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self.enroll_tk_img)

            # Check completion
            if self.enroll_samples_count >= self.enroll_samples_target:
                self.is_enrolling = False
                self.btn_start_enroll.config(state=tk.NORMAL, text="📸 Start Live Webcam Capture")
                if not self.is_scanning and self.cap:
                    self.cap.release()
                    self.cap = None

                messagebox.showinfo("Enrollment Success",
                                    f"Successfully enrolled {self.enroll_target_user} with {self.enroll_samples_count} samples!\n"
                                    "Now navigate to 'Dual ML Training' to train XGBoost & Random Forest.")
                self._refresh_users_list()
                return

        self.root.after(30, self._enroll_loop)

    def _refresh_users_list(self):
        self.lst_users.delete(0, tk.END)
        users = self.dataset_manager.get_enrolled_users()
        for u in users:
            u_dir = config.FACES_DIR / u
            count = len(list(u_dir.glob("*.jpg")))
            self.lst_users.insert(tk.END, f"{u} ({count} images)")

    def _delete_selected_user(self):
        selected = self.lst_users.curselection()
        if not selected:
            messagebox.showwarning("Selection", "Select a user from the list to delete.")
            return

        item_text = self.lst_users.get(selected[0])
        user_name = item_text.split(" ")[0]

        if messagebox.askyesno("Confirm Deletion", f"Permanently delete all facial records for '{user_name}'?"):
            u_dir = config.FACES_DIR / user_name
            if u_dir.exists():
                for f in u_dir.glob("*"):
                    f.unlink()
                u_dir.rmdir()
            self._refresh_users_list()
            messagebox.showinfo("Deleted", f"User '{user_name}' has been deleted.")

    def _import_kaggle_profiles(self):
        count = self.dataset_manager.import_kaggle_profiles_to_faces(samples_per_user=15)
        self._refresh_users_list()
        messagebox.showinfo("Import Success", f"Successfully synchronized {count} facial samples from Kaggle dataset (31 Classes)!")

    # -------------------------------------------------------------
    # TAB 3: DUAL ML MODEL TRAINING & BENCHMARKS
    # -------------------------------------------------------------
    def _build_analytics_tab(self):
        container = tk.Frame(self.tab_analytics, bg="#0f141c")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Top Control Card
        top_card = tk.Frame(container, bg="#131c27", relief=tk.RIDGE, bd=1, padx=20, pady=16)
        top_card.pack(fill=tk.X, pady=(0, 14))

        tk.Label(top_card, text="DUAL MACHINE LEARNING ENGINE: XGBOOST & RANDOM FOREST",
                 font=("Segoe UI", 12, "bold"), fg="#38bdf8", bg="#131c27").pack(anchor="w")
        tk.Label(top_card,
                 text="Trains Bagged Decision Trees (Random Forest) and Extreme Gradient Boosting (XGBoost) with probabilistic soft-voting.",
                 font=("Segoe UI", 9), fg="#94a3b8", bg="#131c27").pack(anchor="w", pady=(2, 10))

        btn_row = tk.Frame(top_card, bg="#131c27")
        btn_row.pack(fill=tk.X)

        self.btn_train = tk.Button(
            btn_row,
            text="🚀 Train Local Models",
            font=("Segoe UI", 10, "bold"),
            bg="#22c55e",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=8,
            command=self._run_training_thread
        )
        self.btn_train.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_train_kaggle = tk.Button(
            btn_row,
            text="⚡ Retrain Kaggle 31-Classes",
            font=("Segoe UI", 10, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=8,
            command=self._run_kaggle_training_thread
        )
        self.btn_train_kaggle.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_train_status = tk.Label(btn_row, text="Status: Ready", font=("Segoe UI", 10), fg="#38bdf8", bg="#131c27")
        self.lbl_train_status.pack(side=tk.LEFT, padx=10)

        # Comparison Metrics Table
        metrics_card = tk.Frame(container, bg="#131c27", relief=tk.RIDGE, bd=1, padx=20, pady=16)
        metrics_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(metrics_card, text="MODEL BENCHMARK & COMPARISON MATRIX", font=("Segoe UI", 11, "bold"), fg="#38bdf8", bg="#131c27").pack(anchor="w")

        # Table with columns: Metric | Random Forest | XGBoost | Soft-Voting Ensemble
        cols = ("Metric", "Random Forest", "XGBoost", "Ensemble (Hybrid)")
        self.tree_metrics = ttk.Treeview(metrics_card, columns=cols, show='headings', height=7)
        for c in cols:
            self.tree_metrics.heading(c, text=c)
            self.tree_metrics.column(c, width=180, anchor=tk.CENTER)
        self.tree_metrics.pack(fill=tk.X, pady=12)

        # Console / Detailed Classification Log
        tk.Label(metrics_card, text="TRAINING PIPELINE LOGS & TELEMETRY", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#131c27").pack(anchor="w")
        self.txt_train_log = tk.Text(metrics_card, height=10, bg="#0a0f16", fg="#e2e8f0", font=("Consolas", 9), relief=tk.FLAT)
        self.txt_train_log.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        # Populate with existing saved metrics if available
        if self.engine.model.last_metrics:
            self._display_metrics(self.engine.model.last_metrics)

    def _run_training_thread(self):
        self.btn_train.config(state=tk.DISABLED)
        self.lbl_train_status.config(text="Status: Loading & Augmenting Dataset...", fg="#f59e0b")
        self.txt_train_log.delete("1.0", tk.END)
        self.txt_train_log.insert(tk.END, "[INFO] Starting Biometric Pipeline...\n")

        thread = threading.Thread(target=self._train_worker, daemon=True)
        thread.start()

    def _run_kaggle_training_thread(self):
        self.btn_train_kaggle.config(state=tk.DISABLED)
        self.lbl_train_status.config(text="Status: Training Kaggle 31-Classes...", fg="#f59e0b")
        self.txt_train_log.delete("1.0", tk.END)
        self.txt_train_log.insert(tk.END, "[INFO] Starting Kaggle 31-Class Pipeline...\n")

        thread = threading.Thread(target=self._kaggle_worker, daemon=True)
        thread.start()

    def _kaggle_worker(self):
        try:
            import train_kaggle
            self._log_txt("[STEP 1] Loading pre-extracted Kaggle features (Train: 2045, Test: 256)...")
            metrics = train_kaggle.run_kaggle_pipeline()
            self.root.after(0, lambda: self._display_metrics(metrics))
            self.root.after(0, lambda: self.lbl_train_status.config(text="Status: Kaggle Models Active!", fg="#22c55e"))
            self.root.after(0, lambda: self.model_status_badge.config(text="● MODEL: ACTIVE (31 Classes)", fg="#22c55e"))
            self.root.after(0, lambda: messagebox.showinfo("Success", "Trained on Kaggle dataset successfully!"))
        except Exception as e:
            self._log_txt(f"[ERROR] Kaggle training failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.btn_train_kaggle.config(state=tk.NORMAL))

    def _train_worker(self):
        try:
            self._log_txt("[STEP 1] Scanning biometric user profiles from disk...")
            users = self.dataset_manager.get_enrolled_users()
            if len(users) < 2:
                self._log_txt(f"[ERROR] Found {len(users)} user(s). Need at least 2 distinct user profiles to train multi-class classifiers.")
                self.root.after(0, lambda: messagebox.showwarning("Dataset Incomplete", "Please enroll at least 2 distinct users before training."))
                self.root.after(0, lambda: self.btn_train.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.lbl_train_status.config(text="Status: Insufficient Data", fg="#ef4444"))
                return

            self._log_txt(f"[STEP 2] Loading faces & applying multi-scale augmentations for {len(users)} users: {users}")
            X, y = self.dataset_manager.load_dataset(apply_augmentation=True)
            self._log_txt(f"[INFO] Dataset ready: Total Samples = {X.shape[0]}, Feature Vector Length = {X.shape[1]}")

            self._log_txt("[STEP 3] Fitting Random Forest (200 trees, parallel bagging)...")
            self._log_txt("[STEP 4] Fitting XGBoost (Gradient Boosting with L1/L2 shrinkage)...")

            metrics = self.engine.model.train(X, y)

            self._log_txt("[SUCCESS] Both models trained & serialized successfully!")
            self._log_txt(f"[BENCHMARK] Random Forest Latency: {metrics['random_forest']['inference_latency_ms']} ms/face")
            self._log_txt(f"[BENCHMARK] XGBoost Latency: {metrics['xgboost']['inference_latency_ms']} ms/face")

            self.root.after(0, lambda: self._display_metrics(metrics))
            self.root.after(0, lambda: self.lbl_train_status.config(text="Status: Training Complete & Models Active!", fg="#22c55e"))
            self.root.after(0, lambda: self.model_status_badge.config(text="● MODEL: ACTIVE", fg="#22c55e"))
            self.root.after(0, lambda: messagebox.showinfo("Training Success", "XGBoost and Random Forest models trained successfully!"))

        except Exception as e:
            self._log_txt(f"[EXCEPTION] Training failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            self.root.after(0, lambda: self.lbl_train_status.config(text="Status: Training Failed", fg="#ef4444"))
        finally:
            self.root.after(0, lambda: self.btn_train.config(state=tk.NORMAL))

    def _log_txt(self, msg: str):
        self.root.after(0, lambda: self.txt_train_log.insert(tk.END, msg + "\n"))
        self.root.after(0, lambda: self.txt_train_log.see(tk.END))

    def _display_metrics(self, m: dict):
        for item in self.tree_metrics.get_children():
            self.tree_metrics.delete(item)

        rf = m.get("random_forest", {})
        xgb = m.get("xgboost", {})
        ens = m.get("ensemble", {})

        rows = [
            ("Accuracy", f"{rf.get('accuracy', 0)*100:.1f}%", f"{xgb.get('accuracy', 0)*100:.1f}%", f"{ens.get('accuracy', 0)*100:.1f}%"),
            ("Precision (Weighted)", f"{rf.get('precision', 0)*100:.1f}%", f"{xgb.get('precision', 0)*100:.1f}%", f"{ens.get('precision', 0)*100:.1f}%"),
            ("Recall (Weighted)", f"{rf.get('recall', 0)*100:.1f}%", f"{xgb.get('recall', 0)*100:.1f}%", f"{ens.get('recall', 0)*100:.1f}%"),
            ("F1-Score", f"{rf.get('f1_score', 0)*100:.1f}%", f"{xgb.get('f1_score', 0)*100:.1f}%", f"{ens.get('f1_score', 0)*100:.1f}%"),
            ("Inference Latency", f"{rf.get('inference_latency_ms', 0):.2f} ms", f"{xgb.get('inference_latency_ms', 0):.2f} ms", f"{ens.get('inference_latency_ms', 0):.2f} ms"),
            ("Training Execution Time", f"{rf.get('train_time_sec', 0):.2f} s", f"{xgb.get('train_time_sec', 0):.2f} s", "--")
        ]
        for r in rows:
            self.tree_metrics.insert('', tk.END, values=r)

    # -------------------------------------------------------------
    # TAB 4: SECURITY AUDIT LOGS & INTRUDER GALLERY
    # -------------------------------------------------------------
    def _build_logs_tab(self):
        container = tk.Frame(self.tab_logs, bg="#0f141c")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header controls
        top_row = tk.Frame(container, bg="#131c27", padx=16, pady=10, relief=tk.RIDGE, bd=1)
        top_row.pack(fill=tk.X, pady=(0, 10))

        tk.Label(top_row, text="SECURITY ACCESS AUDIT TRAIL", font=("Segoe UI", 11, "bold"), fg="#38bdf8", bg="#131c27").pack(side=tk.LEFT)

        btn_refresh_logs = tk.Button(top_row, text="🔄 Refresh Logs", font=("Segoe UI", 9), bg="#0284c7", fg="#ffffff", relief=tk.FLAT, padx=12, command=self._refresh_audit_logs)
        btn_refresh_logs.pack(side=tk.RIGHT, padx=6)

        # Main Table & Intruder Preview split
        split_frame = tk.Frame(container, bg="#0f141c")
        split_frame.pack(fill=tk.BOTH, expand=True)

        # Logs Treeview
        log_col = tk.Frame(split_frame, bg="#131c27", relief=tk.RIDGE, bd=1)
        log_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        cols = ("ID", "Timestamp", "User / Subject", "Status", "Confidence", "RF Score", "XGB Score", "Liveness", "Snapshot")
        self.tree_logs = ttk.Treeview(log_col, columns=cols, show='headings', height=14)
        for c in cols:
            self.tree_logs.heading(c, text=c)
            self.tree_logs.column(c, width=95, anchor=tk.CENTER)
        self.tree_logs.column("Timestamp", width=140)
        self.tree_logs.column("User / Subject", width=120)
        self.tree_logs.pack(fill=tk.BOTH, expand=True)

        self.tree_logs.bind("<<TreeviewSelect>>", self._on_log_select)

        # Intruder Snapshot Preview Frame
        preview_col = tk.Frame(split_frame, bg="#131c27", width=320, relief=tk.RIDGE, bd=1, padx=14, pady=14)
        preview_col.pack(side=tk.RIGHT, fill=tk.BOTH)
        preview_col.pack_propagate(False)

        tk.Label(preview_col, text="FORENSIC SNAPSHOT PREVIEW", font=("Segoe UI", 10, "bold"), fg="#ef4444", bg="#131c27").pack(anchor="w")

        self.intruder_canvas = tk.Canvas(preview_col, bg="#0a0f16", height=240, highlightthickness=0)
        self.intruder_canvas.pack(fill=tk.X, pady=10)

        self.lbl_snapshot_info = tk.Label(preview_col, text="Select an unauthorized attempt to view captured intruder image.",
                                          font=("Segoe UI", 9), fg="#94a3b8", bg="#131c27", wraplength=280, justify=tk.LEFT)
        self.lbl_snapshot_info.pack(fill=tk.X)

        self._refresh_audit_logs()

    def _refresh_audit_logs(self):
        for item in self.tree_logs.get_children():
            self.tree_logs.delete(item)

        rows = self.security_logger.get_recent_logs(limit=100)
        for r in rows:
            conf_str = f"{r['confidence']*100:.1f}%"
            rf_str = f"{r['rf_confidence']*100:.1f}%"
            xgb_str = f"{r['xgb_confidence']*100:.1f}%"
            live_str = "Verified" if r['liveness_verified'] else "Spoof"
            snap_str = r['snapshot_path'] if r['snapshot_path'] else "--"

            self.tree_logs.insert('', tk.END, values=(
                r['id'], r['timestamp'], r['user_name'], r['status'],
                conf_str, rf_str, xgb_str, live_str, snap_str
            ))

    def _on_log_select(self, event):
        selected = self.tree_logs.selection()
        if not selected:
            return

        values = self.tree_logs.item(selected[0], "values")
        snap_filename = values[8]

        if snap_filename and snap_filename != "--":
            snap_path = config.INTRUDER_DIR / snap_filename
            if snap_path.exists():
                img = cv2.imread(str(snap_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cw = self.intruder_canvas.winfo_width()
                    ch = self.intruder_canvas.winfo_height()
                    if cw > 10 and ch > 10:
                        h_img, w_img = img_rgb.shape[:2]
                        scale = min(cw / w_img, ch / h_img)
                        resized = cv2.resize(img_rgb, (int(w_img * scale), int(h_img * scale)))
                        pil_img = Image.fromarray(resized)
                        self.snap_tk = ImageTk.PhotoImage(image=pil_img)
                        self.intruder_canvas.delete("all")
                        self.intruder_canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self.snap_tk)
                        self.lbl_snapshot_info.config(text=f"Captured Intruder: {values[2]}\nTimestamp: {values[1]}")
                        return

        self.intruder_canvas.delete("all")
        self.lbl_snapshot_info.config(text="No forensic photo attached to this event.")

    # -------------------------------------------------------------
    # TAB 5: SYSTEM SETTINGS & THRESHOLDS
    # -------------------------------------------------------------
    def _build_settings_tab(self):
        container = tk.Frame(self.tab_settings, bg="#0f141c")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        card = tk.Frame(container, bg="#131c27", relief=tk.RIDGE, bd=1, padx=24, pady=20)
        card.pack(fill=tk.X)

        tk.Label(card, text="BIOMETRIC SECURITY POLICIES & ENSEMBLE HYPERPARAMETERS",
                 font=("Segoe UI", 12, "bold"), fg="#38bdf8", bg="#131c27").pack(anchor="w", pady=(0, 16))

        # Security Threshold Slider
        tk.Label(card, text="Verification Security Confidence Threshold (%):", fg="#cbd5e1", bg="#131c27", font=("Segoe UI", 10)).pack(anchor="w")
        tk.Label(card, text="Matches with ensemble confidence below this threshold are rejected as UNKNOWN / INTRUDER.",
                 fg="#94a3b8", bg="#131c27", font=("Segoe UI", 8)).pack(anchor="w")

        thresh_frame = tk.Frame(card, bg="#131c27")
        thresh_frame.pack(fill=tk.X, pady=(4, 16))
        self.scale_thresh = tk.Scale(thresh_frame, from_=10, to=95, orient=tk.HORIZONTAL, bg="#1e293b", fg="#ffffff", highlightthickness=0, length=400)
        self.scale_thresh.set(int(self.engine.model.security_threshold * 100))
        self.scale_thresh.pack(side=tk.LEFT)

        # Ensemble Weights
        tk.Label(card, text="Ensemble Probability Fusion Weights:", fg="#cbd5e1", bg="#131c27", font=("Segoe UI", 10)).pack(anchor="w")
        tk.Label(card, text="Adjust contribution of XGBoost vs Random Forest in final soft-voting probability.",
                 fg="#94a3b8", bg="#131c27", font=("Segoe UI", 8)).pack(anchor="w")

        weights_frame = tk.Frame(card, bg="#131c27")
        weights_frame.pack(fill=tk.X, pady=(4, 16))

        tk.Label(weights_frame, text="XGBoost Weight (%):", fg="#38bdf8", bg="#131c27", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 6))
        self.scale_xgb = tk.Scale(weights_frame, from_=10, to=90, orient=tk.HORIZONTAL, bg="#1e293b", fg="#ffffff", highlightthickness=0, length=180)
        self.scale_xgb.set(int(self.engine.model.weight_xgb * 100))
        self.scale_xgb.pack(side=tk.LEFT, padx=(0, 20))

        # Anti-Spoofing Toggle
        self.var_liveness = tk.BooleanVar(value=config.REQUIRE_LIVENESS)
        chk_live = tk.Checkbutton(card, text="Enable Anti-Spoofing & Liveness Detection (Rejects 2D Photos & Screens)",
                                  variable=self.var_liveness, fg="#38bdf8", bg="#131c27", selectcolor="#0f141c", font=("Segoe UI", 10))
        chk_live.pack(anchor="w", pady=(8, 16))

        # Save Settings Button
        btn_save = tk.Button(
            card,
            text="💾 Save & Apply Security Settings",
            font=("Segoe UI", 10, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=18,
            pady=8,
            command=self._save_settings
        )
        btn_save.pack(anchor="w")

    def _save_settings(self):
        new_thresh = self.scale_thresh.get() / 100.0
        new_xgb_w = self.scale_xgb.get() / 100.0
        new_rf_w = 1.0 - new_xgb_w

        self.engine.model.security_threshold = new_thresh
        self.engine.model.weight_xgb = new_xgb_w
        self.engine.model.weight_rf = new_rf_w
        config.REQUIRE_LIVENESS = self.var_liveness.get()

        if self.engine.model.is_trained:
            self.engine.model.save_models()

        messagebox.showinfo("Settings Applied",
                            f"Security settings updated successfully!\n"
                            f"- Threshold: {new_thresh*100:.0f}%\n"
                            f"- XGBoost Weight: {new_xgb_w*100:.0f}%\n"
                            f"- Random Forest Weight: {new_rf_w*100:.0f}%\n"
                            f"- Liveness Verification: {'Enabled' if config.REQUIRE_LIVENESS else 'Disabled'}")

    def _update_status_bar(self):
        # Periodically refresh model badge in case training completed
        if self.engine.model.is_trained:
            self.model_status_badge.config(text="● MODEL: ACTIVE", fg="#22c55e")
        else:
            self.model_status_badge.config(text="● MODEL: UNTRAINED", fg="#ef4444")
        self.root.after(2000, self._update_status_bar)

    def _on_close(self):
        self.is_scanning = False
        self.is_enrolling = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()
