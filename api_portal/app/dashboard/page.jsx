import { getServerSession } from "next-auth";
import { authOptions } from "../api/auth/[...nextauth]/route";

export default async function DashboardPage() {
  const session = await getServerSession(authOptions);

  if (!session) {
    // Redirect client to login page if not signed in
    redirect("/login");
  }

  return (
    <div>
      Welcome, {session.user.name} ({session.user.email})
    </div>
  );
}
