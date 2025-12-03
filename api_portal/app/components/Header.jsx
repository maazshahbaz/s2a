"use client";

import { signOut, useSession } from "next-auth/react";

export default function Header() {
  const { data: session } = useSession();

  return (
    <header className="header">
      <div className="header-content">
        <div className="header-left">
          <h1 className="header-title">API Portal</h1>
          {session?.user?.name && (
            <p className="header-subtitle">Welcome, {session.user.name}</p>
          )}
        </div>
        <div className="header-right">
          <button onClick={() => signOut()} className="button-primary">
            Sign Out
          </button>
        </div>
      </div>
    </header>
  );
}
