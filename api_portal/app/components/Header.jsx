"use client";

import { signOut, useSession } from "next-auth/react";

export default function Header() {
  const { data: session } = useSession();

  const getInitials = (name) => {
    if (!name) return "U";
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <header className="header">
      <div className="header-content">
        <div className="header-right">
          <button className="header-user" onClick={() => signOut()}>
            <div className="header-avatar">
              {getInitials(session?.user?.name)}
            </div>
            <span className="header-user-name">
              {session?.user?.name || "User"}
            </span>
          </button>
        </div>
      </div>
    </header>
  );
}
