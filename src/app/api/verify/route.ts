import { NextRequest, NextResponse } from "next/server";
import { checkRateLimit } from "@/lib/rate-limit";
import { verifyFaceWithService } from "@/lib/recognition-client";
import { addAccessLog } from "@/lib/db";
import { VerificationRequest } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  // 1. IP Rate Limiting
  const clientIp = req.headers.get("x-forwarded-for") || req.ip || "127.0.0.1";
  const rateLimit = checkRateLimit(clientIp, 40, 60);

  if (!rateLimit.allowed) {
    return NextResponse.json(
      {
        error: "Rate limit exceeded. Too many verification requests.",
        retryAfter: rateLimit.resetTime
      },
      {
        status: 429,
        headers: { "Retry-After": rateLimit.resetTime.toString() }
      }
    );
  }

  try {
    const body: VerificationRequest = await req.json();

    // 2. Validate Request & Explicit Consent
    if (!body.image || typeof body.image !== "string") {
      return NextResponse.json(
        { error: "Validation Error: 'image' (Base64 string) is required." },
        { status: 400 }
      );
    }

    if (!body.consent) {
      return NextResponse.json(
        { error: "Privacy Policy: Biometric facial verification requires explicit user consent." },
        { status: 403 }
      );
    }

    const securityThreshold = typeof body.securityThreshold === "number"
      ? body.securityThreshold
      : 0.75;

    // 3. Forward to Biometric Recognition Engine
    const result = await verifyFaceWithService(body.image, securityThreshold);

    // 4. Record Forensic Access Log
    await addAccessLog({
      timestamp: new Date().toISOString(),
      userName: result.identity,
      status: result.status,
      confidence: result.confidence,
      rfConfidence: result.rfConfidence,
      xgbConfidence: result.xgbConfidence,
      livenessScore: result.liveness.score,
      livenessVerified: result.liveness.isLive,
      modelsAgree: result.modelsAgree,
      clientIp: clientIp,
      notes: result.liveness.message
    });

    return NextResponse.json(result, { status: 200 });

  } catch (err: any) {
    console.error("[API_VERIFY_ERROR]", err);
    return NextResponse.json(
      { error: "Internal Server Error during facial verification.", details: err.message },
      { status: 500 }
    );
  }
}
