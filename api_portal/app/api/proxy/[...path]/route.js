import { getServerSession } from "next-auth";
import { NextResponse } from "next/server";
import { authOptions } from "../../auth/[...nextauth]/route";
import { signPayload } from "@/lib/hmac";

async function handler(req, { params }) {
  // Skip session check for static Next.js assets
  if (req.nextUrl.pathname.startsWith("/_next/")) {
    return NextResponse.next();
  }

  // Get session
  const session = await getServerSession(authOptions);
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Construct backend URL (await params for Next.js 15+)
  const { path: pathArray } = await params;
  const path = pathArray.join("/");
  const backendUrl = process.env.BACKEND_URL;
  const url = `${backendUrl}/${path}`;

  const method = req.method;
  let body = null;
  let bodyStr = "";

  if (method !== "GET" && method !== "HEAD") {
    try {
      body = await req.json();
      bodyStr = JSON.stringify(body);
    } catch (e) {
      // Empty body is okay
    }
  }

  const userId = session.user.id;
  const secret = process.env.HMAC_SECRET;

  if (!secret) {
    console.error("HMAC_SECRET not set");
    return NextResponse.json({ error: "Configuration error" }, { status: 500 });
  }

  const hmacHeaders = signPayload(userId, bodyStr, secret);

  const headers = {
    "Content-Type": "application/json",
    ...hmacHeaders,
    "x-user-id": userId,
  };

  try {
    const res = await fetch(url, {
      method,
      headers,
      body: bodyStr || undefined,
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      console.error(`[Proxy] Backend error:`, data);
      return NextResponse.json(data, { status: res.status });
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Proxy error:", error);
    return NextResponse.json({ error: "Backend request failed" }, { status: 500 });
  }
}

// Export all methods
export { handler as GET, handler as POST, handler as PUT, handler as PATCH, handler as DELETE };
