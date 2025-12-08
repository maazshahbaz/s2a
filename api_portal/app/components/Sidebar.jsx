"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Sidebar() {
  const pathname = usePathname();
  const [isCollapsed, setIsCollapsed] = useState(false);

  const navItems = [
    {
      id: "dashboard",
      label: "Dashboard",
      href: "/dashboard",
      icon: (
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
        >
          <rect x="3" y="3" width="7" height="7" rx="1" />
          <rect x="14" y="3" width="7" height="7" rx="1" />
          <rect x="3" y="14" width="7" height="7" rx="1" />
          <rect x="14" y="14" width="7" height="7" rx="1" />
        </svg>
      ),
    },
    {
      id: "api-keys",
      label: "API Keys",
      href: "/dashboard/api-keys",
      icon: (
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" />
        </svg>
      ),
    },
    {
      id: "auth-keys",
      label: "Auth Keys",
      href: "/dashboard/auth-keys",
      icon: (
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        </svg>
      ),
    },
    {
      id: "sdk-docs",
      label: "SDK Docs",
      href: "/dashboard/sdk-docs",
      icon: (
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path d="M16 18l6-6-6-6M8 6l-6 6 6 6" />
        </svg>
      ),
    },
    
  ];

  const isActive = (href) => {
    if (href === "/dashboard") {
      return pathname === "/dashboard";
    }
    return pathname.startsWith(href);
  };

  return (
    <aside className={`sidebar ${isCollapsed ? "collapsed" : ""}`}>
      <div className="sidebar-header">
        <div 
          className="sidebar-logo"
          onClick={() => isCollapsed && setIsCollapsed(false)}
          style={{ cursor: isCollapsed ? "pointer" : "default" }}
        >
          <div className="sidebar-logo-icon">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2.5}
              style={{ width: 18, height: 18 }}
            >
              <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
            </svg>
          </div>
          {!isCollapsed && <span className="sidebar-logo-text">DevAPI</span>}
        </div>
        {isCollapsed ? (
          <button
            className="sidebar-toggle"
            onClick={() => setIsCollapsed(false)}
            title="Expand sidebar"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              style={{ width: 16, height: 16 }}
            >
              <path d="M9 18l6-6-6-6" />
            </svg>
          </button>
        ) : (
          <button
            className="sidebar-toggle"
            onClick={() => setIsCollapsed(true)}
            title="Collapse sidebar"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              style={{ width: 16, height: 16 }}
            >
              <path d="M15 18l-6-6 6-6" />
            </svg>
          </button>
        )}
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <Link
            key={item.id}
            href={item.href}
            className={`sidebar-item ${isActive(item.href) ? "active" : ""}`}
          >
            <span className="sidebar-icon">{item.icon}</span>
            {!isCollapsed && <span className="sidebar-text">{item.label}</span>}
          </Link>
        ))}
      </nav>

      {!isCollapsed && (
        <div className="sidebar-footer">
          <div className="sidebar-version">
            <span className="sidebar-version-label">API Version</span>
            <span className="sidebar-version-number">v1.0.0</span>
          </div>
        </div>
      )}
    </aside>
  );
}
