// Server-Side Authorization & Credential Security
import bcrypt from "bcryptjs";

const ADMIN_API_KEY = process.env.ADMIN_API_KEY || "admin_face_id_secret_key";

export function verifyAdminAccess(request: Request): { authorized: boolean; reason?: string } {
  // Check x-admin-key header
  const headerKey = request.headers.get("x-admin-key");
  if (headerKey && headerKey === ADMIN_API_KEY) {
    return { authorized: true };
  }

  // Check Authorization Bearer header
  const authHeader = request.headers.get("authorization");
  if (authHeader && authHeader.startsWith("Bearer ")) {
    const token = authHeader.substring(7);
    if (token === ADMIN_API_KEY) {
      return { authorized: true };
    }
  }

  return {
    authorized: false,
    reason: "Unauthorized: Invalid or missing administrator credentials (x-admin-key)"
  };
}

export async function hashSecret(secret: string): Promise<string> {
  const salt = await bcrypt.genSalt(10);
  return bcrypt.hash(secret, salt);
}

export async function verifySecret(secret: string, hash: string): Promise<boolean> {
  return bcrypt.compare(secret, hash);
}
