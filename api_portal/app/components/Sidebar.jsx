"use client";

import { useState } from "react";

export default function Sidebar() {
  const [activeItem, setActiveItem] = useState("api-keys");

  return (
    <>
      <style jsx>{`
        .sidebar {
          width: 240px;
          background: #1e293b;
          border-right: 1px solid #334155;
          padding: 1.5rem 0;
          height: calc(100vh - 64px);
          position: sticky;
          top: 64px;
          overflow-y: auto;
        }

        .sidebar-nav {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          padding: 0 0.75rem;
        }

        .sidebar-item {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.75rem 1rem;
          background: transparent;
          border: none;
          border-radius: 0.5rem;
          color: #94a3b8;
          font-size: 0.875rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.15s;
          width: 100%;
          text-align: left;
        }

        .sidebar-item:hover {
          background: #0f172a;
          color: #cbd5e1;
        }

        .sidebar-item.active {
          background: rgba(37, 99, 235, 0.1);
          color: #3b82f6;
          border: 1px solid rgba(37, 99, 235, 0.2);
        }

        .sidebar-item.active:hover {
          background: rgba(37, 99, 235, 0.15);
          color: #60a5fa;
        }

        .sidebar-icon {
          width: 20px;
          height: 20px;
          flex-shrink: 0;
        }

        .sidebar-text {
          flex: 1;
        }

        @media (max-width: 768px) {
          .sidebar {
            width: 64px;
            padding: 1rem 0;
          }

          .sidebar-nav {
            padding: 0 0.5rem;
          }

          .sidebar-item {
            justify-content: center;
            padding: 0.75rem;
          }

          .sidebar-text {
            display: none;
          }
        }
      `}</style>

      <aside className="sidebar">
        <nav className="sidebar-nav">
          <button
            className={`sidebar-item ${
              activeItem === "api-keys" ? "active" : ""
            }`}
            onClick={() => setActiveItem("api-keys")}
          >
            <svg
              className="sidebar-icon"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"
              />
            </svg>
            <span className="sidebar-text">API Keys</span>
          </button>
        </nav>
      </aside>
    </>
  );
}