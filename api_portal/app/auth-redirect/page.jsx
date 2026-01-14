import { getServerSession } from "next-auth";
import { authOptions } from "../api/auth/[...nextauth]/route";
import { redirect } from "next/navigation";

export default async function AuthRedirect() {
  const session = await getServerSession(authOptions);
  console.log("[AuthRedirect] Session:", !!session);

  if (!session) {
    console.log("[AuthRedirect] No session, redirecting to /login");
    redirect("/login");
  }

  console.log("[AuthRedirect] Session found, redirecting to /dashboard");
  redirect("/dashboard");
}
