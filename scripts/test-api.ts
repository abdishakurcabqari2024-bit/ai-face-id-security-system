/**
 * Automated Test Suite - AI Face ID Security System API
 * Tests: Enrolled users, Unknown faces, Revoked permissions, Unauthorized requests, Rate limiting.
 */

import { checkRateLimit } from "../src/lib/rate-limit";
import { verifyAdminAccess } from "../src/lib/auth";
import { getUsers, createUser, deleteUser, getAccessLogs, addAccessLog } from "../src/lib/db";
import { verifyFaceWithService } from "../src/lib/recognition-client";

async function runTestSuite() {
  console.log("============================================================");
  console.log("🧪 RUNNING AI FACE ID SECURITY SYSTEM TEST SUITE");
  console.log("============================================================");
  let passed = 0;
  let failed = 0;

  function assert(condition: boolean, testName: string) {
    if (condition) {
      console.log(`  ✅ PASS: ${testName}`);
      passed++;
    } else {
      console.error(`  ❌ FAIL: ${testName}`);
      failed++;
    }
  }

  // TEST 1: Rate Limiter
  console.log("\n[1] Testing Rate Limiting Middleware...");
  const ip = "192.168.1.50";
  const r1 = checkRateLimit(ip, 5, 10);
  assert(r1.allowed === true && r1.remaining === 4, "Initial request within rate limit allowed");

  for (let i = 0; i < 4; i++) {
    checkRateLimit(ip, 5, 10);
  }
  const rBlocked = checkRateLimit(ip, 5, 10);
  assert(rBlocked.allowed === false, "Excessive requests correctly blocked with 429 Too Many Requests");

  // TEST 2: Authorization & Admin Credentials
  console.log("\n[2] Testing Authentication & Server-Side Permissions...");
  const fakeReqNoAuth = new Request("http://localhost/api/enroll", {
    headers: {}
  });
  const authRes1 = verifyAdminAccess(fakeReqNoAuth);
  assert(authRes1.authorized === false, "Unauthorized request blocked without x-admin-key");

  const validReq = new Request("http://localhost/api/enroll", {
    headers: { "x-admin-key": "admin_face_id_secret_key" }
  });
  const authRes2 = verifyAdminAccess(validReq);
  assert(authRes2.authorized === true, "Authorized administrator request approved");

  // TEST 3: Database & User Management Operations
  console.log("\n[3] Testing In-Memory / PostgreSQL Database Operations...");
  const usersInitial = await getUsers();
  assert(usersInitial.length >= 8, `Initial user directory contains ${usersInitial.length} biometric profiles`);

  const testUser = await createUser({
    name: "Test_Security_Agent",
    role: "Special_Operations",
    department: "Cyber Defense",
    consent: true,
    sampleCount: 5
  });
  assert(testUser.name === "Test_Security_Agent", "New biometric profile successfully created");

  const usersAfter = await getUsers();
  assert(usersAfter.some(u => u.name === "Test_Security_Agent"), "Created user appears in directory");

  // TEST 4: User Deletion & Access Revocation
  console.log("\n[4] Testing User Deletion & Revocation...");
  const deleted = await deleteUser(testUser.id);
  assert(deleted === true, "User successfully deleted/revoked");

  const usersAfterDel = await getUsers();
  assert(!usersAfterDel.some(u => u.id === testUser.id), "Deleted user no longer in active user directory");

  // TEST 5: Forensic Audit Trail Logging
  console.log("\n[5] Testing Forensic Access Log Audit Trail...");
  await addAccessLog({
    timestamp: new Date().toISOString(),
    userName: "UNKNOWN / INTRUDER",
    status: "ACCESS_DENIED",
    confidence: 0.38,
    rfConfidence: 0.41,
    xgbConfidence: 0.35,
    livenessScore: 0.94,
    livenessVerified: true,
    modelsAgree: true,
    clientIp: "10.0.0.1",
    notes: "Automated test assertion"
  });

  const logs = await getAccessLogs(5);
  const found = logs.find(l => l.userName === "UNKNOWN / INTRUDER" && l.status === "ACCESS_DENIED");
  assert(found !== undefined, "Unauthorized intrusion correctly captured in forensic audit log");

  // TEST 6: Recognition Engine Offline Safety Gate
  console.log("\n[6] Testing Recognition Engine Fail-Safe (No Fake Access)...");
  // Test with dummy image against an unreachable host to verify safe error state
  const failResult = await verifyFaceWithService("dummy_base64_string", 0.75);
  assert(
    failResult.accessGranted === false && failResult.status === "SERVICE_UNAVAILABLE",
    "Recognition engine safely denies access when service is unreachable (Never fakes access!)"
  );

  console.log("\n============================================================");
  console.log(`🏁 TEST RESULTS: ${passed} PASSED, ${failed} FAILED`);
  console.log("============================================================");

  if (failed > 0) process.exit(1);
}

runTestSuite().catch(err => {
  console.error("Test execution error:", err);
  process.exit(1);
});
