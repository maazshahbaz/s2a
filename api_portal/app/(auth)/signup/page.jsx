"use client";

import { useState } from "react";
import { apiPost } from "@/lib/api";
import { useRouter } from "next/navigation";

export default function SignupPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSignup(e) {
    e.preventDefault();
    setLoading(true);
    setError("");

    const form = new FormData(e.target);
    const email = form.get("email");
    const password = form.get("password");
    const name = form.get("name");

    try {
      await apiPost("/auth/signup", { email, password, name });
      router.push(`/verify-otp?email=${email}`);
    } catch (err) {
      setError(err.message);
    }

    setLoading(false);
  }

  return (
    <div className="max-w-sm mx-auto mt-20">
      <h1 className="text-2xl font-bold mb-4">Create Account</h1>

      {error && <p className="text-red-500 mb-2">{error}</p>}

      <form onSubmit={handleSignup} className="space-y-3">
        <input name="name" placeholder="Full Name" required className="input" />
        <input name="email" type="email" placeholder="Email" required className="input" />
        <input name="password" type="password" placeholder="Password" required className="input" />

        <button disabled={loading} className="btn w-full">
          {loading ? "Creating..." : "Create Account"}
        </button>
      </form>
    </div>
  );
}
