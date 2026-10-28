import { NextRequest, NextResponse } from "next/server";
import { getTelemetry } from "@/lib/db";
import { checkServiceHealth } from "@/lib/recognition-client";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  try {
    const telemetry = await getTelemetry();
    const serviceHealth = await checkServiceHealth();

    return NextResponse.json({
      ...telemetry,
      serviceOnline: serviceHealth.online,
      serviceInfo: serviceHealth.info
    }, { status: 200 });
  } catch (err: any) {
    return NextResponse.json({ error: "Failed to retrieve telemetry", details: err.message }, { status: 500 });
  }
}
