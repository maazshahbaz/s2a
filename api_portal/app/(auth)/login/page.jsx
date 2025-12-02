"use client";
import { signIn } from "next-auth/react";

export default function LoginPage() {
  return (
    <button onClick={() => signIn("azure-ad", { callbackUrl: "/auth-redirect" })}>
      Sign in with Microsoft
    </button>
  );
}
