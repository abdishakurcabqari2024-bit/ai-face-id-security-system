# 🛡️ AI Face ID Biometric Security System (Web & Vercel Edition)
### *Enterprise Decoupled Biometric Architecture: Next.js Serverless Dashboard + FastAPI Dual-Model (XGBoost + Random Forest) Engine*

[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-black?logo=next.js)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue?logo=typescript)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8?logo=tailwind-css)](https://tailwindcss.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-eb5424?logo=xgboost)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-f7931e?logo=scikit-learn)](https://scikit-learn.org/)
[![Prisma / PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon%20%7C%20Supabase-336791?logo=postgresql)](https://neon.tech/)
[![Vercel Ready](https://img.shields.io/badge/Vercel-Deployable-black?logo=vercel)](https://vercel.com/)

---

## 🏛️ Executive Summary & Architectural Decisions

### 1. The Vercel Runtime Boundary Challenge
A common question in production biometric engineering: *Can the full OpenCV + Scikit-Learn + XGBoost stack run directly inside a single Vercel Serverless Function?*

**Senior ML & Full-Stack Analysis**:
* **Vercel / AWS Lambda Bundle Limit**: Vercel Serverless Functions have an uncompressed artifact limit of **250 MB** and a Free Tier invocation timeout of **10 seconds**.
* **Biometric Stack Footprint**:
  * `opencv-python`: ~120 MB uncompressed (requires shared system libraries `libgl1`, `libglib2.0-0` absent from standard serverless container images).
  * `xgboost`: ~300 MB uncompressed (`libxgboost.so`).
  * `scikit-learn` + `scipy` + `numpy`: ~160 MB uncompressed.
  * Model Weights (`rf_model.joblib` + `xgb_model.json`): ~38 MB.
  * **Total Uncompressed Size**: **~618 MB** (over **2.5× the hard 250 MB limit**).
* **Cold Starts & Statelessness**: Cold-starting a 600 MB ML image on Lambda can take 15–20s, triggering `504 FUNCTION_INVOCATION_TIMEOUT`. Furthermore, Vercel functions cannot maintain video loops or write to persistent SQLite databases.

### 2. The Production Decoupled Architecture
To satisfy Vercel compatibility without weakening security or faking biometric recognition results:

```mermaid
flowchart TD
    subgraph Browser["User Browser / Client"]
        Cam["🎥 HTML5 getUserMedia\n(Webcam with Consent)"] --> NextUI["💻 Next.js Dark Dashboard\n(Targeting HUD & Telemetry)"]
    end

    subgraph Vercel["Vercel Edge / Serverless Platform"]
        NextUI --> RouteVerify["/api/verify\n(Rate Limiting & Proxy)"]
        NextUI --> RouteEnroll["/api/enroll\n(Admin Auth & Validation)"]
        NextUI --> RouteUsers["/api/users\n(Directory & Revocation)"]
        NextUI --> RouteLogs["/api/logs\n(Audit Trail Explorer)"]
        NextUI --> RouteStats["/api/stats\n(Real-Time KPIs)"]
    end

    subgraph DB["Hosted Database (Neon / Supabase)"]
        RouteVerify --> PG[("🐘 PostgreSQL / Prisma\n(Users, Hashes & Logs)")]
        RouteEnroll --> PG
        RouteUsers --> PG
        RouteLogs --> PG
    end

    subgraph Engine["Recognition Microservice (Render / Railway / HuggingFace)"]
        RouteVerify -- "Bearer Token + Base64 Frame" --> Micro["FastAPI Biometric Engine\n(Port 8000)"]
        RouteEnroll -- "User Face Samples" --> Micro
        
        Micro --> Det["OpenCV Haar / Cascade Face Detector"]
        Det --> Live{"Laplacian Liveness\n(Anti-Spoof Filter)"}
        Live -- "Spoof Attack" --> Deny["SPOOF REJECTED"]
        Live -- "3D Human" --> Ext["1,858-dim Vector\n(HOG + LBP + Moments)"]
        Ext --> XGB["XGBoost (55% Weight)"]
        Ext --> RF["Random Forest (45% Weight)"]
        XGB --> SoftVote["Soft-Voting Fusion"]
        RF --> SoftVote
        SoftVote --> Gate{"Confidence >= 75%?"}
        Gate -- "YES" --> Grant["ACCESS GRANTED"]
        Gate -- "NO" --> Intruder["ACCESS DENIED / INTRUDER"]
    end
```

---

## ✨ Features & Capabilities

1. **Next.js 14 Dark Cyber Dashboard**:
   * Responsive high-tech UI built with Tailwind CSS and Lucide icons.
   * Real-time targeting HUD overlay with corner reticles and scanline animations.
   * Visual Access Verdicts: `ACCESS GRANTED` (Emerald), `ACCESS DENIED` (Rose), `SPOOF ATTACK DETECTED` (Amber).
   * Dual Model Progress Meters (XGBoost vs. Random Forest).
   * Models Agreement Indicator (`MATCH (RF == XGB)` vs. `DIVERGENT`).
2. **HTML5 `getUserMedia` Webcam Pipeline**:
   * Secure, permission-based video capture.
   * Mandatory user consent checkbox complying with biometric privacy guidelines.
   * Manual Single Scan or Continuous Surveillance (3s debounced interval).
3. **Face Enrollment & Profile Management**:
   * Rapid burst capture (1–8 facial snapshots from live camera).
   * Metadata storage (Name, Role, Department, Consent Timestamp).
   * One-click profile revocation / permanent deletion.
4. **Security Audit Trail & Forensic Logging**:
   * Filterable audit table (All, Granted, Denied, Spoofs).
   * Logs client IP, timestamp, model confidence scores, and anti-spoofing diagnostics.
5. **Zero Fake Access Policy**:
   * If the recognition microservice is offline or encounters an error, the system safely denies access (`SERVICE_UNAVAILABLE`). It **never** auto-grants access on failure.
6. **Rate Limiting & Authentication**:
   * Sliding-window IP rate limiter (prevents API flooding / brute-force spoofing).
   * Admin master key (`x-admin-key`) protecting mutating endpoints (`/api/enroll`, `/api/users/[id]`).

---

## 📂 Repository Layout

```
AI Face ID Security System/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   ├── enroll/route.ts        # Admin-protected user enrollment
│   │   │   ├── health/route.ts        # Service health check & readiness
│   │   │   ├── logs/route.ts          # Forensic access log query endpoint
│   │   │   ├── stats/route.ts         # Real-time system telemetry KPIs
│   │   │   ├── users/route.ts         # Enrolled user directory
│   │   │   ├── users/[id]/route.ts    # User deletion & profile revocation
│   │   │   └── verify/route.ts        # Face verification gateway
│   │   ├── globals.css                # Dark cyber styles & scanlines
│   │   ├── layout.tsx                 # Root layout & metadata
│   │   └── page.tsx                   # Interactive Security Dashboard
│   └── lib/
│       ├── auth.ts                    # Admin authentication & password hashing
│       ├── db.ts                      # PostgreSQL adapter with in-memory fallback
│       ├── rate-limit.ts              # IP sliding-window rate limiter
│       ├── recognition-client.ts      # Typed client for Python microservice
│       └── types.ts                   # Shared TypeScript interfaces
├── recognition_service/
│   ├── main.py                        # FastAPI microservice (OpenCV, XGBoost, RF)
│   ├── requirements.txt               # Python dependencies for microservice
│   ├── Dockerfile                     # Multi-stage production container
│   ├── render.yaml                    # One-click Render Blueprint
│   ├── railway.json                   # One-click Railway deployment config
│   └── test_service.py                # Python test suite
├── prisma/
│   └── schema.prisma                  # PostgreSQL schema (User, Profile, AccessLog)
├── scripts/
│   └── test-api.ts                    # Fullstack API test suite (11 tests)
├── .env.example                       # Documented environment variables template
├── .gitignore                         # Excludes large binaries, models, datasets & keys
├── next.config.js                     # Next.js production build configuration
├── package.json                       # Next.js scripts & dependencies
├── tailwind.config.ts                 # Cyber dark theme definition
├── tsconfig.json                      # TypeScript configuration (ES2020)
└── vercel.json                        # Vercel security headers
```

---

## 🚀 Step-by-Step Deployment Instructions

### Option 1: Local Development (Windows / macOS / Linux)

#### 1. Clone & Install Dependencies
```bash
cd "AI Face ID Security System"
npm install
```

#### 2. Start the Python Biometric Recognition Microservice
In a separate terminal:
```powershell
python -m uvicorn recognition_service.main:app --host 0.0.0.0 --port 8000 --reload
```
*The microservice starts on `http://localhost:8000` with the pre-trained 31-class Kaggle models loaded.*

#### 3. Start the Next.js Dashboard
In your primary terminal:
```bash
npm run dev
```
*Open [http://localhost:3000](http://localhost:3000) in your browser.*

---

### Option 2: Deploying to Production (Vercel + Hosted Services)

#### Step 1: Set Up External PostgreSQL Database (Free Tier)
1. Go to [Neon.tech](https://neon.tech) or [Supabase.com](https://supabase.com).
2. Create a free PostgreSQL database.
3. Copy your connection string:
   ```
   postgresql://username:password@ep-sample-123.us-east-2.aws.neon.tech/face_security_db?sslmode=require
   ```

#### Step 2: Deploy the Python Recognition Microservice (Free on Render or Railway)
##### Deploying on Render (Free Tier):
1. Push your repository to **GitHub**.
2. Go to [dashboard.render.com](https://dashboard.render.com/) -> **New** -> **Web Service**.
3. Connect your GitHub repository.
4. Select **Docker** as Runtime (Render automatically detects `recognition_service/Dockerfile`).
5. Set Root Directory: `.`
6. Set Environment Variables:
   * `PORT`: `8000`
   * `RECOGNITION_SECRET_KEY`: `your_random_32_byte_secret`
   * `REQUIRE_AUTH`: `false` (or `true` if authenticating via bearer token)
7. Click **Create Web Service**. Once deployed, copy your Render URL (e.g. `https://face-id-service.onrender.com`).

##### Alternative: Deploying on Railway / Hugging Face Spaces:
* **Railway**: Click **New Project** -> **Deploy from GitHub Repo** -> Select Dockerfile.
* **Hugging Face Spaces**: Create a **Docker Space** (Free 16GB CPU). Copy `recognition_service/` files.

#### Step 3: Deploy the Next.js Web App on Vercel
1. Go to [vercel.com](https://vercel.com) and click **Add New** -> **Project**.
2. Import your GitHub repository.
3. Configure Project Settings:
   * **Framework Preset**: `Next.js`
   * **Root Directory**: `./` (leave default)
   * **Build Command**: `npm run build`
   * **Output Directory**: `.next` (leave default)
4. Add **Environment Variables** in Vercel:
   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | `postgresql://...` (from Step 1) |
   | `RECOGNITION_SERVICE_URL` | `https://face-id-service.onrender.com` (from Step 2) |
   | `RECOGNITION_SECRET_KEY` | `your_random_32_byte_secret` |
   | `ADMIN_API_KEY` | `your_secret_admin_master_key` |
   | `DEFAULT_SECURITY_THRESHOLD` | `0.75` |
   | `REQUIRE_LIVENESS` | `true` |
5. Click **Deploy**. Vercel will build and deploy the dashboard in under 60 seconds!

---

## 🧪 Testing & Quality Assurance

### Run Frontend & API Integration Tests
```bash
npm test
```
**Test Coverage**:
* `[1]` IP-based Sliding Window Rate Limiter (enforces limits and handles 429 responses).
* `[2]` Server-Side Admin Authentication (verifies `x-admin-key` validation).
* `[3]` Database Operations (user registration and profile creation).
* `[4]` Profile Revocation & Deletion (verifies removal from active directory).
* `[5]` Forensic Audit Trail (captures unauthorized intrusion records).
* `[6]` Recognition Engine Fail-Safe (verifies no fake access is granted when offline).

### Run Python Biometric Engine Unit Tests
```bash
python recognition_service/test_service.py
```
**Test Coverage**:
* `[1]` Feature Extractor Output (1,858 normalized biometric dimensions).
* `[2]` Anti-Spoofing Filter (rejects flat printed photos and screens).
* `[3]` Dual Model Prediction (XGBoost + Random Forest soft-voting threshold gate).
* `[4]` Known Identity vs. Unknown Impostor Separation.

---

## 🔒 Security & Biometric Privacy Compliance

1. **Explicit Consent Verification**:
   * The API refuses to process any facial scan if `consent !== true`, returning HTTP `403 Forbidden`.
2. **Template Protection**:
   * Raw video frames are processed in memory and never stored permanently unless an intrusion occurs for security forensics.
   * Biometric templates are cryptographically separated from public endpoints.
3. **GDPR / CCPA Right to Deletion**:
   * Administrators can permanently erase user records and associated facial sample files via `DELETE /api/users/:id`.
4. **Brute-Force & Replay Attack Defense**:
   * Integrated IP sliding-window rate limiting prevents automated face replay attacks.
