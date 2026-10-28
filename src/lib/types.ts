// Shared TypeScript Definitions - AI Face ID Security System

export interface VerificationRequest {
  image: string; // Base64 encoded JPEG or PNG
  consent: boolean;
  securityThreshold?: number;
}

export interface VerificationResult {
  status: 'ACCESS_GRANTED' | 'ACCESS_DENIED' | 'SPOOF_ATTACK' | 'NO_FACE_DETECTED' | 'SERVICE_UNAVAILABLE';
  identity: string;
  accessGranted: boolean;
  confidence: number; // 0.0 - 1.0
  rfConfidence: number;
  xgbConfidence: number;
  rfIdentity: string;
  xgbIdentity: string;
  modelsAgree: boolean;
  liveness: {
    isLive: boolean;
    score: number;
    message: string;
  };
  latencyMs: number;
  timestamp: string;
}

export interface UserRecord {
  id: string;
  name: string;
  role: string;
  department: string;
  status: 'ACTIVE' | 'SUSPENDED' | 'REVOKED';
  consentGiven: boolean;
  consentTimestamp: string;
  createdAt: string;
  sampleCount?: number;
}

export interface EnrollmentRequest {
  name: string;
  role: string;
  department: string;
  consent: boolean;
  images: string[]; // Array of base64 face crops
}

export interface AccessLogEntry {
  id: string;
  timestamp: string;
  userName: string;
  status: string;
  confidence: number;
  rfConfidence: number;
  xgbConfidence: number;
  livenessScore: number;
  livenessVerified: boolean;
  modelsAgree: boolean;
  clientIp?: string;
  notes?: string;
}

export interface SecurityTelemetry {
  totalScans: number;
  grantedCount: number;
  deniedCount: number;
  spoofCount: number;
  activeUsers: number;
  averageConfidence: number;
  serviceOnline: boolean;
  modelsLoaded: boolean;
}
