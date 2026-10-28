import { NextRequest, NextResponse } from "next/server";
import { verifyAdminAccess } from "@/lib/auth";
import { createUser, addAccessLog } from "@/lib/db";
import { enrollWithService } from "@/lib/recognition-client";
import { EnrollmentRequest } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  // 1. Enforce Admin Permissions
  const auth = verifyAdminAccess(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason }, { status: 401 });
  }

  try {
    const body: EnrollmentRequest = await req.json();

    // 2. Validate Fields
    if (!body.name || typeof body.name !== "string" || body.name.trim().length < 2) {
      return NextResponse.json({ error: "Validation Error: Valid 'name' is required." }, { status: 400 });
    }

    if (!body.consent) {
      return NextResponse.json({ error: "Consent Required: Cannot enroll without recorded user consent." }, { status: 400 });
    }

    if (!body.images || !Array.isArray(body.images) || body.images.length === 0) {
      return NextResponse.json({ error: "Validation Error: At least one face image sample is required." }, { status: 400 });
    }

    const cleanName = body.name.trim();
    const role = body.role || "Employee";
    const department = body.department || "Operations";

    // 3. Save User Profile to Database
    const newUser = await createUser({
      name: cleanName,
      role: role,
      department: department,
      consent: body.consent,
      sampleCount: body.images.length
    });

    // 4. Forward Face Samples to Recognition Engine
    const enrollResult = await enrollWithService(cleanName, body.images);

    // 5. Audit Log the Enrollment Action
    await addAccessLog({
      timestamp: new Date().toISOString(),
      userName: cleanName,
      status: "USER_ENROLLED",
      confidence: 1.0,
      rfConfidence: 1.0,
      xgbConfidence: 1.0,
      livenessScore: 1.0,
      livenessVerified: true,
      modelsAgree: true,
      notes: `Enrolled by Operator. Samples: ${body.images.length}. Role: ${role}`
    });

    return NextResponse.json({
      success: true,
      user: newUser,
      message: `User '${cleanName}' enrolled successfully with ${body.images.length} biometric samples.`,
      serviceSync: enrollResult
    }, { status: 201 });

  } catch (err: any) {
    console.error("[API_ENROLL_ERROR]", err);
    return NextResponse.json({ error: "Enrollment failed", details: err.message }, { status: 500 });
  }
}
