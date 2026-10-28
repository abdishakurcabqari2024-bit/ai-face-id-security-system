"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Shield,
  ShieldCheck,
  ShieldAlert,
  Camera,
  CameraOff,
  UserCheck,
  UserX,
  Users,
  FileText,
  Sliders,
  RefreshCw,
  Trash2,
  AlertTriangle,
  Activity,
  Cpu,
  Lock,
  Eye,
  CheckCircle2,
  XCircle
} from "lucide-react";
import { VerificationResult, UserRecord, AccessLogEntry, SecurityTelemetry } from "@/lib/types";

export default function SecurityDashboard() {
  // Navigation
  const [activeTab, setActiveTab] = useState<"scanner" | "enroll" | "users" | "logs" | "settings">("scanner");

  // Camera & Video
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [cameraPermission, setCameraPermission] = useState<boolean>(false);
  const [userConsent, setUserConsent] = useState<boolean>(true);
  const [autoScan, setAutoScan] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  // Security Threshold
  const [securityThreshold, setSecurityThreshold] = useState<number>(0.75);
  const [adminKey, setAdminKey] = useState<string>("admin_face_id_secret_key");

  // Telemetry & Verification Output
  const [latestResult, setLatestResult] = useState<VerificationResult | null>(null);
  const [telemetry, setTelemetry] = useState<SecurityTelemetry>({
    totalScans: 0,
    grantedCount: 0,
    deniedCount: 0,
    spoofCount: 0,
    activeUsers: 8,
    averageConfidence: 0.0,
    serviceOnline: false,
    modelsLoaded: true
  });

  // User Directory & Logs State
  const [users, setUsers] = useState<UserRecord[]>([]);
  const [logs, setLogs] = useState<AccessLogEntry[]>([]);
  const [logFilter, setLogFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");

  // Enrollment State
  const [enrollName, setEnrollName] = useState<string>("");
  const [enrollRole, setEnrollRole] = useState<string>("Employee");
  const [enrollDept, setEnrollDept] = useState<string>("Operations");
  const [enrollConsent, setEnrollConsent] = useState<boolean>(true);
  const [enrollSamples, setEnrollSamples] = useState<string[]>([]);
  const [enrollLoading, setEnrollLoading] = useState<boolean>(false);
  const [enrollMessage, setEnrollMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // 1. Fetch System Telemetry, Users & Logs
  const fetchTelemetry = useCallback(async () => {
    try {
      const res = await fetch("/api/stats");
      if (res.ok) {
        const data = await res.json();
        setTelemetry(data);
      }
    } catch (e) {
      console.warn("Failed fetching telemetry:", e);
    }
  }, []);

  const fetchUsers = useCallback(async () => {
    try {
      const res = await fetch("/api/users");
      if (res.ok) {
        const data = await res.json();
        setUsers(data.users || []);
      }
    } catch (e) {
      console.warn("Failed fetching users:", e);
    }
  }, []);

  const fetchLogs = useCallback(async () => {
    try {
      const url = logFilter === "ALL" ? "/api/logs?limit=50" : `/api/logs?limit=50&filter=${logFilter}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setLogs(data.logs || []);
      }
    } catch (e) {
      console.warn("Failed fetching logs:", e);
    }
  }, [logFilter]);

  useEffect(() => {
    fetchTelemetry();
    fetchUsers();
    fetchLogs();
    const interval = setInterval(fetchTelemetry, 6000);
    return () => clearInterval(interval);
  }, [fetchTelemetry, fetchUsers, fetchLogs]);

  // 2. Camera Stream Lifecycle
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setCameraActive(true);
      setCameraPermission(true);
    } catch (err: any) {
      console.error("Camera access error:", err);
      alert(`Camera Permission Error: ${err.message}. Please allow webcam access in your browser.`);
      setCameraActive(false);
      setCameraPermission(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setAutoScan(false);
  };

  // 3. Frame Capture & Face Verification
  const captureFrameBase64 = (): string | null => {
    if (!videoRef.current || !cameraActive) return null;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  };

  const executeVerification = async () => {
    if (isProcessing) return;
    if (!userConsent) {
      alert("Please check the biometric privacy consent box before verification.");
      return;
    }

    const frameBase64 = captureFrameBase64();
    if (!frameBase64) {
      alert("No active camera frame available. Please enable camera first.");
      return;
    }

    setIsProcessing(true);
    try {
      const res = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: frameBase64,
          consent: userConsent,
          securityThreshold: securityThreshold
        })
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || `HTTP ${res.status}`);
      }

      const result: VerificationResult = await res.json();
      setLatestResult(result);
      fetchTelemetry();
      fetchLogs();
    } catch (err: any) {
      console.error("Verification failed:", err);
      setLatestResult({
        status: "SERVICE_UNAVAILABLE",
        identity: "Verification Error",
        accessGranted: false,
        confidence: 0.0,
        rfConfidence: 0.0,
        xgbConfidence: 0.0,
        rfIdentity: "None",
        xgbIdentity: "None",
        modelsAgree: false,
        liveness: { isLive: false, score: 0.0, message: err.message },
        latencyMs: 0,
        timestamp: new Date().toISOString()
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Auto-Surveillance Loop
  useEffect(() => {
    let interval: any = null;
    if (autoScan && cameraActive) {
      interval = setInterval(() => {
        if (!isProcessing) executeVerification();
      }, 3000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoScan, cameraActive, isProcessing, userConsent, securityThreshold]);

  // 4. Enrollment Capture
  const captureEnrollmentSample = () => {
    const frame = captureFrameBase64();
    if (!frame) {
      alert("Enable the camera first to capture sample frames.");
      return;
    }
    if (enrollSamples.length >= 8) {
      alert("Maximum 8 samples captured for this profile.");
      return;
    }
    setEnrollSamples(prev => [...prev, frame]);
  };

  const submitEnrollment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!enrollName.trim()) {
      setEnrollMessage({ type: "error", text: "Please enter a valid full name." });
      return;
    }
    if (!enrollConsent) {
      setEnrollMessage({ type: "error", text: "Consent agreement is required." });
      return;
    }
    if (enrollSamples.length === 0) {
      setEnrollMessage({ type: "error", text: "Capture at least 1 face frame before enrolling." });
      return;
    }

    setEnrollLoading(true);
    setEnrollMessage(null);

    try {
      const res = await fetch("/api/enroll", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-admin-key": adminKey
        },
        body: JSON.stringify({
          name: enrollName.trim(),
          role: enrollRole,
          department: enrollDept,
          consent: enrollConsent,
          images: enrollSamples
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Failed to enroll user");
      }

      setEnrollMessage({ type: "success", text: `Success: ${data.message}` });
      setEnrollName("");
      setEnrollSamples([]);
      fetchUsers();
      fetchTelemetry();
    } catch (err: any) {
      setEnrollMessage({ type: "error", text: err.message });
    } finally {
      setEnrollLoading(false);
    }
  };

  // 5. Delete / Revoke User
  const handleDeleteUser = async (userId: string, userName: string) => {
    if (!confirm(`Are you sure you want to delete biometric profile for '${userName}'? This action is permanent.`)) {
      return;
    }

    try {
      const res = await fetch(`/api/users/${userId}`, {
        method: "DELETE",
        headers: { "x-admin-key": adminKey }
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Failed to delete user");
      }
      alert(`User '${userName}' has been deleted.`);
      fetchUsers();
      fetchTelemetry();
      fetchLogs();
    } catch (err: any) {
      alert(`Error deleting user: ${err.message}`);
    }
  };

  // Render Telemetry Helpers
  const filteredUsers = users.filter(u =>
    u.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    u.department.toLowerCase().includes(searchQuery.toLowerCase()) ||
    u.role.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-cyber-dark text-slate-100 flex flex-col">
      {/* Top Header Bar */}
      <header className="border-b border-cyber-border bg-cyber-card/80 backdrop-blur px-6 py-4 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-cyber-cyan/10 border border-cyber-cyan/30 text-cyber-cyan">
            <Shield className="w-6 h-6 animate-pulse-slow" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-wider text-slate-100 flex items-center gap-2">
              AI FACE ID SECURITY SYSTEM
              <span className="text-xs px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/30 font-mono">
                XGBoost + Random Forest v2.0
              </span>
            </h1>
            <p className="text-xs text-slate-400">Enterprise Biometric Access Control & Anti-Spoofing Architecture</p>
          </div>
        </div>

        {/* Global Status Badges */}
        <div className="flex items-center gap-3 text-xs font-mono">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-900 border border-slate-800">
            <span className={`w-2 h-2 rounded-full ${cameraActive ? "bg-emerald-400 animate-ping" : "bg-slate-500"}`} />
            <span className="text-slate-300">CAMERA: {cameraActive ? "ACTIVE" : "STANDBY"}</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-900 border border-slate-800">
            <Cpu className="w-3.5 h-3.5 text-cyber-cyan" />
            <span className="text-slate-300">
              RECOGNITION ENGINE:{" "}
              <span className={telemetry.serviceOnline ? "text-emerald-400" : "text-amber-400 font-bold"}>
                {telemetry.serviceOnline ? "ONLINE (31 CLASSES)" : "OFFLINE / LOCAL"}
              </span>
            </span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-900 border border-slate-800">
            <Sliders className="w-3.5 h-3.5 text-cyber-amber" />
            <span className="text-slate-300">THRESHOLD: {(securityThreshold * 100).toFixed(0)}%</span>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="border-b border-cyber-border bg-slate-950 px-6 flex gap-2 overflow-x-auto">
        <button
          onClick={() => setActiveTab("scanner")}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-semibold border-b-2 transition-colors ${
            activeTab === "scanner"
              ? "border-cyber-cyan text-cyber-cyan bg-cyber-cyan/5"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <Camera className="w-4 h-4" />
          Live Security Scanner
        </button>
        <button
          onClick={() => setActiveTab("enroll")}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-semibold border-b-2 transition-colors ${
            activeTab === "enroll"
              ? "border-cyber-cyan text-cyber-cyan bg-cyber-cyan/5"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <UserCheck className="w-4 h-4" />
          Enroll Biometrics
        </button>
        <button
          onClick={() => setActiveTab("users")}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-semibold border-b-2 transition-colors ${
            activeTab === "users"
              ? "border-cyber-cyan text-cyber-cyan bg-cyber-cyan/5"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <Users className="w-4 h-4" />
          Enrolled Profiles ({users.length})
        </button>
        <button
          onClick={() => setActiveTab("logs")}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-semibold border-b-2 transition-colors ${
            activeTab === "logs"
              ? "border-cyber-cyan text-cyber-cyan bg-cyber-cyan/5"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <FileText className="w-4 h-4" />
          Security Audit Logs
        </button>
        <button
          onClick={() => setActiveTab("settings")}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-semibold border-b-2 transition-colors ${
            activeTab === "settings"
              ? "border-cyber-cyan text-cyber-cyan bg-cyber-cyan/5"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <Sliders className="w-4 h-4" />
          Policies & Thresholds
        </button>
      </nav>

      {/* Main Tab Content */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full">
        {/* ======================================================== */}
        {/* TAB 1: LIVE FACE ID SECURITY SCANNER */}
        {/* ======================================================== */}
        {activeTab === "scanner" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left 7 Columns: Webcam Feed with Futuristic Reticle */}
            <div className="lg:col-span-7 flex flex-col gap-4">
              <div className="relative aspect-video bg-black rounded-xl border border-cyber-border overflow-hidden shadow-2xl flex items-center justify-center">
                {/* Video Stream */}
                <video
                  ref={videoRef}
                  playsInline
                  muted
                  className={`w-full h-full object-cover transform -scale-x-100 ${!cameraActive && "hidden"}`}
                />

                {/* Cyber Targeting Reticle Brackets */}
                {cameraActive && (
                  <>
                    <div className="reticle-bracket bracket-tl" />
                    <div className="reticle-bracket bracket-tr" />
                    <div className="reticle-bracket bracket-bl" />
                    <div className="reticle-bracket bracket-br" />
                    <div className="absolute inset-0 scanline-overlay" />

                    {/* HUD Central Reticle */}
                    <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                      <div className="w-48 h-60 border border-cyan-500/30 rounded-2xl flex flex-col justify-between p-2">
                        <span className="text-[10px] font-mono text-cyan-400 self-start">FACE_ALIGNMENT_TARGET</span>
                        <div className="self-center w-6 h-6 border-t border-b border-cyan-500/40" />
                        <span className="text-[10px] font-mono text-cyan-400 self-end">LIVENESS_CHECK_ACTIVE</span>
                      </div>
                    </div>
                  </>
                )}

                {/* Standby Message if Camera Off */}
                {!cameraActive && (
                  <div className="text-center p-8 flex flex-col items-center gap-3">
                    <div className="w-16 h-16 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-500">
                      <CameraOff className="w-8 h-8" />
                    </div>
                    <h3 className="text-base font-semibold text-slate-300">Biometric Camera Stream Inactive</h3>
                    <p className="text-xs text-slate-500 max-w-sm">
                      Click Initialize Camera below to request browser webcam permission and begin verification.
                    </p>
                    <button
                      onClick={startCamera}
                      className="mt-2 px-5 py-2.5 rounded-lg bg-cyber-blue hover:bg-sky-600 text-white text-xs font-bold tracking-wider uppercase transition shadow-lg shadow-sky-900/40"
                    >
                      Initialize Camera Stream
                    </button>
                  </div>
                )}
              </div>

              {/* Camera Controls & Consent Banner */}
              <div className="p-4 rounded-xl bg-cyber-card border border-cyber-border flex flex-col sm:flex-row items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  {cameraActive ? (
                    <button
                      onClick={stopCamera}
                      className="px-4 py-2 rounded-lg bg-rose-600/80 hover:bg-rose-600 text-white text-xs font-semibold flex items-center gap-2 transition"
                    >
                      <CameraOff className="w-4 h-4" /> Stop Camera
                    </button>
                  ) : (
                    <button
                      onClick={startCamera}
                      className="px-4 py-2 rounded-lg bg-cyber-blue hover:bg-sky-600 text-white text-xs font-semibold flex items-center gap-2 transition"
                    >
                      <Camera className="w-4 h-4" /> Start Camera
                    </button>
                  )}

                  <button
                    disabled={!cameraActive || isProcessing}
                    onClick={executeVerification}
                    className="px-5 py-2 rounded-lg bg-cyber-emerald hover:bg-emerald-600 text-slate-950 font-bold text-xs flex items-center gap-2 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-emerald-950/50"
                  >
                    <Eye className="w-4 h-4" />
                    {isProcessing ? "Analyzing Biometrics..." : "Verify Identity"}
                  </button>
                </div>

                {/* Auto Scan surveillance toggle */}
                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={autoScan}
                    disabled={!cameraActive}
                    onChange={e => setAutoScan(e.target.checked)}
                    className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0"
                  />
                  <span>Surveillance Mode (Auto 3s)</span>
                </label>
              </div>

              {/* Privacy Consent Agreement Checkbox */}
              <div className="px-4 py-3 rounded-lg bg-slate-900/60 border border-slate-800 flex items-start gap-3">
                <input
                  type="checkbox"
                  id="consentCheckbox"
                  checked={userConsent}
                  onChange={e => setUserConsent(e.target.checked)}
                  className="mt-1 rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0"
                />
                <label htmlFor="consentCheckbox" className="text-xs text-slate-400 leading-relaxed cursor-pointer">
                  <span className="font-semibold text-slate-300">Biometric Privacy Agreement:</span> I acknowledge and
                  consent to facial recognition analysis for the purpose of identity authentication and security access
                  control in accordance with enterprise data protection regulations.
                </label>
              </div>
            </div>

            {/* Right 5 Columns: Decision Telemetry & Verification Card */}
            <div className="lg:col-span-5 flex flex-col gap-4">
              <div className="p-5 rounded-xl bg-cyber-card border border-cyber-border flex flex-col gap-5 shadow-xl">
                <div className="flex items-center justify-between border-b border-cyber-border pb-3">
                  <span className="text-xs font-mono font-bold tracking-wider text-cyan-400 flex items-center gap-2">
                    <Activity className="w-4 h-4" /> REAL-TIME TELEMETRY
                  </span>
                  <span className="text-[11px] font-mono text-slate-500">
                    LATENCY: {latestResult ? `${latestResult.latencyMs} ms` : "0 ms"}
                  </span>
                </div>

                {/* Big Status Banner */}
                {latestResult ? (
                  <div
                    className={`p-5 rounded-xl border flex flex-col items-center justify-center text-center gap-2 ${
                      latestResult.status === "ACCESS_GRANTED"
                        ? "bg-emerald-950/40 border-emerald-500/40 text-emerald-400 glow-emerald"
                        : latestResult.status === "SPOOF_ATTACK"
                        ? "bg-amber-950/40 border-amber-500/40 text-amber-400"
                        : "bg-rose-950/40 border-rose-500/40 text-rose-400 glow-rose"
                    }`}
                  >
                    {latestResult.status === "ACCESS_GRANTED" && <ShieldCheck className="w-12 h-12" />}
                    {latestResult.status === "SPOOF_ATTACK" && <AlertTriangle className="w-12 h-12" />}
                    {latestResult.status !== "ACCESS_GRANTED" && latestResult.status !== "SPOOF_ATTACK" && (
                      <ShieldAlert className="w-12 h-12" />
                    )}

                    <span className="text-xl font-black tracking-widest uppercase">
                      {latestResult.status === "ACCESS_GRANTED"
                        ? "ACCESS GRANTED"
                        : latestResult.status === "SPOOF_ATTACK"
                        ? "SPOOF / PHOTO REJECTED"
                        : "ACCESS DENIED / INTRUDER"}
                    </span>
                    <span className="text-sm font-bold text-slate-200">{latestResult.identity}</span>
                    <span className="text-xs font-mono text-slate-400">
                      Ensemble Match: {(latestResult.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ) : (
                  <div className="p-8 rounded-xl bg-slate-900/50 border border-slate-800 text-center text-slate-500 flex flex-col items-center gap-2">
                    <Lock className="w-8 h-8 opacity-40" />
                    <span className="text-xs font-semibold">GATE STANDBY</span>
                    <span className="text-[11px]">Align face and trigger verification to evaluate access.</span>
                  </div>
                )}

                {/* Dual ML Model Probability Meters */}
                <div className="space-y-4 pt-2">
                  <div className="text-xs font-semibold text-slate-300 flex justify-between">
                    <span>XGBoost Gradient Boosting</span>
                    <span className="text-cyan-400 font-mono">
                      {latestResult ? `${(latestResult.xgbConfidence * 100).toFixed(1)}%` : "0.0%"}
                    </span>
                  </div>
                  <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                    <div
                      className="bg-cyan-500 h-2 transition-all duration-500"
                      style={{ width: `${latestResult ? latestResult.xgbConfidence * 100 : 0}%` }}
                    />
                  </div>

                  <div className="text-xs font-semibold text-slate-300 flex justify-between">
                    <span>Random Forest (200 Trees)</span>
                    <span className="text-emerald-400 font-mono">
                      {latestResult ? `${(latestResult.rfConfidence * 100).toFixed(1)}%` : "0.0%"}
                    </span>
                  </div>
                  <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                    <div
                      className="bg-emerald-500 h-2 transition-all duration-500"
                      style={{ width: `${latestResult ? latestResult.rfConfidence * 100 : 0}%` }}
                    />
                  </div>
                </div>

                {/* Sub-metrics Grid */}
                <div className="grid grid-cols-2 gap-3 pt-3 border-t border-cyber-border text-xs">
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-[10px] font-mono text-slate-500 block">LIVENESS / ANTI-SPOOF</span>
                    <span
                      className={`font-semibold ${
                        latestResult?.liveness.isLive ? "text-emerald-400" : "text-amber-400"
                      }`}
                    >
                      {latestResult?.liveness.isLive ? "3D Human Verified" : "Unverified"}
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-[10px] font-mono text-slate-500 block">MODELS AGREEMENT</span>
                    <span
                      className={`font-semibold ${
                        latestResult?.modelsAgree ? "text-emerald-400" : "text-amber-400"
                      }`}
                    >
                      {latestResult?.modelsAgree ? "MATCH (RF == XGB)" : "DIVERGENT"}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 2: BIOMETRIC USER ENROLLMENT */}
        {/* ======================================================== */}
        {activeTab === "enroll" && (
          <div className="max-w-3xl mx-auto bg-cyber-card border border-cyber-border rounded-xl p-6 shadow-xl">
            <h2 className="text-base font-bold text-slate-100 mb-1 flex items-center gap-2">
              <UserCheck className="w-5 h-5 text-cyan-400" />
              Enroll New Biometric Identity
            </h2>
            <p className="text-xs text-slate-400 mb-6">
              Captures facial invariant features (HOG + LBP) and registers the identity into the database and model
              catalogue.
            </p>

            {enrollMessage && (
              <div
                className={`p-4 rounded-lg mb-6 text-xs flex items-center gap-3 ${
                  enrollMessage.type === "success"
                    ? "bg-emerald-950/60 border border-emerald-500/40 text-emerald-300"
                    : "bg-rose-950/60 border border-rose-500/40 text-rose-300"
                }`}
              >
                {enrollMessage.type === "success" ? <CheckCircle2 className="w-5 h-5" /> : <XCircle className="w-5 h-5" />}
                {enrollMessage.text}
              </div>
            )}

            <form onSubmit={submitEnrollment} className="space-y-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">Full Legal Name / ID</label>
                  <input
                    type="text"
                    required
                    value={enrollName}
                    onChange={e => setEnrollName(e.target.value)}
                    placeholder="e.g. Zahra_Ali"
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">Security Role</label>
                  <select
                    value={enrollRole}
                    onChange={e => setEnrollRole(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="Security_Director">Security Director</option>
                    <option value="Lead_Engineer">Lead Engineer</option>
                    <option value="Security_Officer">Security Officer</option>
                    <option value="Employee">Employee</option>
                    <option value="VIP_Guest">VIP Guest</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Department</label>
                <input
                  type="text"
                  value={enrollDept}
                  onChange={e => setEnrollDept(e.target.value)}
                  placeholder="e.g. Critical Infrastructure"
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
                />
              </div>

              {/* Sample Capture Box */}
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-200">
                    Facial Snapshots Captured: ({enrollSamples.length}/8)
                  </span>
                  <div className="flex gap-2">
                    {!cameraActive && (
                      <button
                        type="button"
                        onClick={startCamera}
                        className="px-3 py-1.5 rounded bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold"
                      >
                        Start Camera
                      </button>
                    )}
                    <button
                      type="button"
                      disabled={!cameraActive}
                      onClick={captureEnrollmentSample}
                      className="px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold disabled:opacity-40"
                    >
                      📸 Snap Sample Frame
                    </button>
                  </div>
                </div>

                {/* Samples Thumbnails */}
                <div className="flex gap-3 overflow-x-auto py-2 min-h-[90px] border border-dashed border-slate-800 rounded-lg p-2 items-center">
                  {enrollSamples.length === 0 ? (
                    <span className="text-xs text-slate-600 italic mx-auto">
                      No snapshots captured yet. Turn on camera and snap 1-5 face angles.
                    </span>
                  ) : (
                    enrollSamples.map((s, idx) => (
                      <div key={idx} className="relative group flex-shrink-0 w-20 h-20 rounded border border-cyan-500/40 overflow-hidden">
                        <img src={s} alt="sample" className="w-full h-full object-cover" />
                        <button
                          type="button"
                          onClick={() => setEnrollSamples(prev => prev.filter((_, i) => i !== idx))}
                          className="absolute inset-0 bg-rose-900/80 text-white text-[10px] font-bold opacity-0 group-hover:opacity-100 flex items-center justify-center transition"
                        >
                          Remove
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Consent check */}
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="enrollConsent"
                  checked={enrollConsent}
                  onChange={e => setEnrollConsent(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0"
                />
                <label htmlFor="enrollConsent" className="text-xs text-slate-300 cursor-pointer">
                  I confirm that the individual has given written/digital biometric consent to be enrolled.
                </label>
              </div>

              <button
                type="submit"
                disabled={enrollLoading}
                className="w-full py-3 rounded-lg bg-cyber-blue hover:bg-sky-600 text-white font-bold text-xs uppercase tracking-wider transition disabled:opacity-50"
              >
                {enrollLoading ? "Registering Biometric Profile..." : "Complete Enrollment"}
              </button>
            </form>
          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 3: REGISTERED USERS DIRECTORY */}
        {/* ======================================================== */}
        {activeTab === "users" && (
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
              <div>
                <h2 className="text-base font-bold text-slate-100">Enrolled Biometric Profiles</h2>
                <p className="text-xs text-slate-400">
                  Active identities registered in PostgreSQL database and evaluated by XGBoost & Random Forest.
                </p>
              </div>
              <input
                type="text"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                placeholder="Search profiles or role..."
                className="px-3 py-1.5 bg-slate-900 border border-slate-700 rounded-lg text-xs text-slate-100 focus:outline-none focus:border-cyan-500 w-64"
              />
            </div>

            <div className="bg-cyber-card border border-cyber-border rounded-xl overflow-hidden shadow-xl">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-950 text-slate-400 font-mono text-[11px] uppercase border-b border-cyber-border">
                    <tr>
                      <th className="px-6 py-3">Identity Name</th>
                      <th className="px-6 py-3">Security Role</th>
                      <th className="px-6 py-3">Department</th>
                      <th className="px-6 py-3">Consent Status</th>
                      <th className="px-6 py-3">Enrolled At</th>
                      <th className="px-6 py-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {filteredUsers.map(user => (
                      <tr key={user.id} className="hover:bg-slate-900/40 transition">
                        <td className="px-6 py-4 font-semibold text-slate-100 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-emerald-400" />
                          {user.name}
                        </td>
                        <td className="px-6 py-4">
                          <span className="px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700 text-[10px] font-mono text-cyan-400">
                            {user.role}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-slate-400">{user.department}</td>
                        <td className="px-6 py-4">
                          <span className="text-emerald-400 font-mono text-[11px] flex items-center gap-1">
                            <CheckCircle2 className="w-3.5 h-3.5" /> VERIFIED
                          </span>
                        </td>
                        <td className="px-6 py-4 text-slate-500 font-mono text-[11px]">
                          {new Date(user.createdAt).toLocaleDateString()}
                        </td>
                        <td className="px-6 py-4 text-right">
                          <button
                            onClick={() => handleDeleteUser(user.id, user.name)}
                            className="p-1.5 rounded hover:bg-rose-950/60 text-slate-400 hover:text-rose-400 transition"
                            title="Revoke / Delete Profile"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 4: SECURITY AUDIT LOGS */}
        {/* ======================================================== */}
        {activeTab === "logs" && (
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
              <div>
                <h2 className="text-base font-bold text-slate-100">Security Access Audit Trail</h2>
                <p className="text-xs text-slate-400">
                  Forensic audit trail of all verification decisions, confidence levels, and intruder alerts.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <select
                  value={logFilter}
                  onChange={e => setLogFilter(e.target.value)}
                  className="px-3 py-1.5 bg-slate-900 border border-slate-700 rounded-lg text-xs text-slate-100 focus:outline-none"
                >
                  <option value="ALL">All Events</option>
                  <option value="ACCESS_GRANTED">Access Granted</option>
                  <option value="ACCESS_DENIED">Access Denied</option>
                  <option value="SPOOF_ATTACK">Spoof Attacks</option>
                </select>
                <button
                  onClick={fetchLogs}
                  className="p-1.5 rounded bg-slate-900 border border-slate-700 hover:bg-slate-800 text-slate-300"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="bg-cyber-card border border-cyber-border rounded-xl overflow-hidden shadow-xl">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-950 text-slate-400 font-mono text-[11px] uppercase border-b border-cyber-border">
                    <tr>
                      <th className="px-6 py-3">Timestamp</th>
                      <th className="px-6 py-3">Subject</th>
                      <th className="px-6 py-3">Verdict</th>
                      <th className="px-6 py-3">Ensemble Score</th>
                      <th className="px-6 py-3">XGB / RF Breakdown</th>
                      <th className="px-6 py-3">Liveness</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 font-mono">
                    {logs.map(log => (
                      <tr key={log.id} className="hover:bg-slate-900/40 transition">
                        <td className="px-6 py-3 text-slate-500 text-[11px]">
                          {new Date(log.timestamp).toLocaleTimeString()}
                        </td>
                        <td className="px-6 py-3 font-semibold text-slate-200">{log.userName}</td>
                        <td className="px-6 py-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              log.status === "ACCESS_GRANTED"
                                ? "bg-emerald-950 text-emerald-400 border border-emerald-500/40"
                                : log.status === "SPOOF_ATTACK"
                                ? "bg-amber-950 text-amber-400 border border-amber-500/40"
                                : "bg-rose-950 text-rose-400 border border-rose-500/40"
                            }`}
                          >
                            {log.status}
                          </span>
                        </td>
                        <td className="px-6 py-3 font-bold text-slate-100">
                          {(log.confidence * 100).toFixed(1)}%
                        </td>
                        <td className="px-6 py-3 text-[11px] text-slate-400">
                          XGB: {(log.xgbConfidence * 100).toFixed(0)}% | RF: {(log.rfConfidence * 100).toFixed(0)}%
                        </td>
                        <td className="px-6 py-3 text-[11px]">
                          <span className={log.livenessVerified ? "text-emerald-400" : "text-amber-400"}>
                            {log.livenessVerified ? "Verified" : "Failed"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 5: SYSTEM POLICIES & THRESHOLDS */}
        {/* ======================================================== */}
        {activeTab === "settings" && (
          <div className="max-w-3xl mx-auto bg-cyber-card border border-cyber-border rounded-xl p-6 shadow-xl space-y-6">
            <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
              <Sliders className="w-5 h-5 text-cyan-400" />
              Security Policies & Threshold Configuration
            </h2>

            {/* Threshold Slider */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-slate-300">Biometric Verification Confidence Threshold</span>
                <span className="font-mono text-cyan-400 text-sm font-bold">{(securityThreshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.95"
                step="0.05"
                value={securityThreshold}
                onChange={e => setSecurityThreshold(parseFloat(e.target.value))}
                className="w-full accent-cyan-400"
              />
              <p className="text-[11px] text-slate-400">
                Matches with ensemble soft-voting confidence below this threshold are immediately rejected as{" "}
                <span className="text-rose-400 font-semibold">ACCESS DENIED / INTRUDER</span>.
              </p>
            </div>

            {/* Admin Key */}
            <div className="space-y-2 pt-4 border-t border-cyber-border">
              <label className="block text-xs font-semibold text-slate-300">
                Administrator API Key (Required for mutating actions)
              </label>
              <input
                type="password"
                value={adminKey}
                onChange={e => setAdminKey(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-100 font-mono"
              />
            </div>

            {/* Architectural Info Box */}
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2 text-xs text-slate-400">
              <span className="font-bold text-slate-300 font-mono block">VERCEL DEPLOYMENT ARCHITECTURE</span>
              <p>
                <strong>Frontend:</strong> Next.js App Router on Vercel Edge/Serverless.
              </p>
              <p>
                <strong>Recognition Engine:</strong> FastAPI Python microservice running OpenCV, XGBoost, and Random
                Forest on Render / Railway with 1,858-dim feature extraction.
              </p>
              <p>
                <strong>Database:</strong> Hosted PostgreSQL (Neon / Supabase) recording users and access logs.
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
