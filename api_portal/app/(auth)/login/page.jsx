"use client";
import { signIn } from "next-auth/react";

export default function LoginPage() {
  return (
    <button onClick={() => signIn("azure-ad")}>
      Sign in with Microsoft
    </button>
  );
}
