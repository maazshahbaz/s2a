// middleware.js
import { withAuth } from "next-auth/middleware";
import { NextResponse } from "next/server";

import { getToken } from "next-auth/jwt";

export default async function middleware(req) {
  // Bypass CORS issues for _next/static in development over Cloudflare Tunnel
  if (process.env.NODE_ENV === "development" && req.nextUrl.pathname.startsWith("/_next/")) {
    const res = NextResponse.next();
    res.headers.set("Access-Control-Allow-Origin", "https://dev.bytepulseai.com");
    res.headers.set("Access-Control-Allow-Credentials", "true");
    return res;
  }

  const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });
  console.log("[Middleware] Path:", req.nextUrl.pathname);
  console.log("[Middleware] Token found:", !!token);
  if (token) {
     console.log("[Middleware] Token sub:", token.sub);
  } else {
     console.log("[Middleware] Cookies:", req.cookies.getAll().map(c => c.name).join(", "));
  }

  // Apply NextAuth protection for dashboard routes
  return withAuth({
    pages: {
      signIn: "/login",
    },
  })(req);
}

export const config = {
  matcher: ["/dashboard/:path*", "/_next/:path*"], // ensure middleware runs on static assets too
};
