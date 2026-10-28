import { NextRequest, NextResponse } from "next/server";
import { checkServiceHealth } from "@/lib/recognition-client";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const recognitionService = await checkServiceHealth();

  return NextResponse.json({
    status: "ok",
    runtime: "Vercel Serverless Node.js",
    timestamp: new Date().toISOString(),
    services: {
      nextjsApi: "HEALTHY",
      recognitionEngine: recognitionService.online ? "ONLINE" : "OFFLINE_STANDALONE_MODE",
      recognitionDetails: recognitionService.info || null
    },
    version: "2.0.0"
  }, { status: 200 });
}
