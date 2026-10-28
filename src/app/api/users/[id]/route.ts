import { NextRequest, NextResponse } from "next/server";
import { verifyAdminAccess } from "@/lib/auth";
import { deleteUser, addAccessLog } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function DELETE(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  // 1. Verify Admin Credentials
  const auth = verifyAdminAccess(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason }, { status: 401 });
  }

  const userId = params.id;
  if (!userId) {
    return NextResponse.json({ error: "Missing user ID parameter" }, { status: 400 });
  }

  try {
    const deleted = await deleteUser(userId);
    if (!deleted) {
      return NextResponse.json({ error: "User not found or already deleted" }, { status: 404 });
    }

    // Record Deletion in Audit Log
    await addAccessLog({
      timestamp: new Date().toISOString(),
      userName: `DELETED_USER_${userId}`,
      status: "USER_DELETED",
      confidence: 1.0,
      rfConfidence: 1.0,
      xgbConfidence: 1.0,
      livenessScore: 1.0,
      livenessVerified: true,
      modelsAgree: true,
      notes: `Biometric profile and user records deleted for ID: ${userId}`
    });

    return NextResponse.json({
      success: true,
      message: `User '${userId}' has been permanently deleted.`
    }, { status: 200 });

  } catch (err: any) {
    return NextResponse.json({ error: "Failed to delete user", details: err.message }, { status: 500 });
  }
}
