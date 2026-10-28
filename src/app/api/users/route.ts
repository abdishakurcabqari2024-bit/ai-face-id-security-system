import { NextRequest, NextResponse } from "next/server";
import { getUsers } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  try {
    const users = await getUsers();
    return NextResponse.json({ users, count: users.length }, { status: 200 });
  } catch (err: any) {
    return NextResponse.json({ error: "Failed to fetch users", details: err.message }, { status: 500 });
  }
}
