import crypto from "crypto";

export function sha256Hex(input) {
  return crypto.createHash("sha256").update(input).digest("hex");
}

// body MUST be the raw JSON string you will send in fetch()
export function signPayload(userId, bodyString, secret) {
  // Ensure bodyString is a string, not an object
  if (typeof bodyString !== "string") {
    throw new Error("bodyString must be a JSON string");
  }

  const timestamp = Date.now().toString();

  // Compute deterministic SHA256 of the actual body
  const bodyHash = sha256Hex(bodyString);

  const payload = `${userId}:${timestamp}:${bodyHash}`;

  const signature = crypto
    .createHmac("sha256", secret)
    .update(payload)
    .digest("hex");

  return {
    "x-user-id": String(userId),
    "x-timestamp": timestamp,
    "x-body-hash": bodyHash,
    "x-signature": signature,
  };
}
