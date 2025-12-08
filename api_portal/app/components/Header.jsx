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
          <button className="header-notification">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              style={{ width: 20, height: 20 }}
            >
              <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 0 1-3.46 0" />
            </svg>
            <span className="header-notification-dot"></span>
          </button>
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
