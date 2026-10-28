// Serverless-Compatible In-Memory Sliding-Window Rate Limiter
// Enforces IP-based rate limiting to prevent brute-force presentation attacks and API flooding.

interface RateLimitRecord {
  timestamps: number[];
}

const rateLimitStore = new Map<string, RateLimitRecord>();

// Cleanup stale records periodically
setInterval(() => {
  const now = Date.now();
  rateLimitStore.forEach((record, key) => {
    record.timestamps = record.timestamps.filter(t => now - t < 60000);
    if (record.timestamps.length === 0) {
      rateLimitStore.delete(key);
    }
  });
}, 30000);

export function checkRateLimit(
  ip: string,
  maxRequests: number = 30,
  windowSeconds: number = 60
): { allowed: boolean; remaining: number; resetTime: number } {
  const now = Date.now();
  const windowMs = windowSeconds * 1000;

  let record = rateLimitStore.get(ip);
  if (!record) {
    record = { timestamps: [] };
    rateLimitStore.set(ip, record);
  }

  // Filter timestamps within current sliding window
  record.timestamps = record.timestamps.filter(t => now - t < windowMs);

  if (record.timestamps.length >= maxRequests) {
    const oldestTimestamp = record.timestamps[0];
    const resetTime = oldestTimestamp + windowMs;
    return {
      allowed: false,
      remaining: 0,
      resetTime: Math.ceil((resetTime - now) / 1000)
    };
  }

  record.timestamps.push(now);
  return {
    allowed: true,
    remaining: maxRequests - record.timestamps.length,
    resetTime: windowSeconds
  };
}
