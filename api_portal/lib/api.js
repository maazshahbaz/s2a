/**
 * Generic fetch wrapper calling the Next.js API Proxy
 */
async function apiRequest(path, method, body = null) {
  // Remove leading slash if present to avoid double slashes with proxy prefix
  const cleanPath = path.startsWith("/") ? path.slice(1) : path;
  
  const res = await fetch(`/api/proxy/${cleanPath}`, {
    method,
    headers: {
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  const json = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(json.detail || json.message || json.error || "Request failed");
  }

  return json;
}

export async function apiPost(path, data) {
  return apiRequest(path, "POST", data);
}

export async function apiGet(path) {
  return apiRequest(path, "GET");
}

export async function apiDelete(path) {
  return apiRequest(path, "DELETE");
}
