"use client";

import { useSession } from "next-auth/react";
import { redirect } from "next/navigation";
import { useEffect, useState } from "react";
import { apiGet, apiPost } from "@/lib/api";
import Header from "../components/Header";
import Sidebar from "../components/Sidebar";
import CreateKeyModal from "../components/CreateKeyModal";

export default function DashboardPage() {
  const { data: session, status } = useSession({
    required: true,
    onUnauthenticated() {
      redirect("/login");
    },
  });
  const [loading, setLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [apiKeys, setApiKeys] = useState([]);

  useEffect(() => {
    if (session) {
      fetchKeys();
    }
  }, [session]);

  const fetchKeys = async () => {
    try {
      const keys = await apiGet("/api-keys/");
      setApiKeys(keys);
    } catch (error) {
      console.error("Failed to fetch keys:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateKey = async (name) => {
    const newKey = await apiPost("/api-keys/", { name });
    await fetchKeys();
    return newKey;
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  const maskKey = (key) => {
    if (!key) return "sk_live_...****";
    return key.substring(0, 12) + "..." + key.slice(-4);
  };
console.log(apiKeys)
  const getActiveKeys = () => apiKeys.filter((key) => key.is_active === true);
  const getRecentKeys = () => apiKeys.slice(0, 2);
  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  if (status === "loading" || loading) {
    return (
      <div className="dashboard-layout">
        <Sidebar />
        <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
          <Header />
          <main
            className="dashboard-content"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <div
              style={{
                color: "var(--color-text-secondary)",
                fontSize: "1.125rem",
              }}
            >
              Loading...
            </div>
          </main>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-layout">
      <Sidebar />
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <Header />
        <main className="dashboard-content">
          {/* Page Header */}
          <div className="content-header">
            <h1 className="content-title">Dashboard</h1>
            <p className="content-subtitle">
              Manage your API keys and monitor usage.
            </p>
          </div>

          {/* Stats Cards */}
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-content">
                <span className="stat-label">Total API Keys</span>
                <span className="stat-value">{apiKeys.length}</span>
              </div>
              <div className="stat-icon accent">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" />
                </svg>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-content">
                <span className="stat-label">Auth Keys</span>
                <span className="stat-value">{getActiveKeys().length}</span>
              </div>
              <div className="stat-icon primary">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                </svg>
              </div>
            </div>

          

            <div className="stat-card">
              <div className="stat-content">
                <span className="stat-label">Revoked Keys</span>
                <span className="stat-value">{apiKeys.length-getActiveKeys().length}</span>
              </div>
             <div className="stat-icon danger">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="red"
                  strokeWidth={2}
                >
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                </svg>
              </div>
            </div>
          </div>

          {/* Dashboard Grid */}
          <div className="dashboard-grid">
            {/* Recent API Keys */}
            <div className="section-card">
              <div className="section-header">
                <h3 className="section-title">Recent API Keys</h3>
                <span
                  className="section-link"
                  onClick={() => setIsModalOpen(true)}
                >
                  + Create new
                </span>
              </div>

              {apiKeys.length === 0 ? (
                <div
                  className="empty-state"
                  style={{ border: "none", padding: "2rem" }}
                >
                  <svg
                    className="empty-state-icon"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  >
                    <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" />
                  </svg>
                  <p className="empty-state-text">
                    No API keys yet. Create your first key to get started.
                  </p>
                </div>
              ) : (
                <div className="api-key-list">
                  {getRecentKeys().map((key) => (
                    <div key={key.id} className="api-key-card">
                      <div className="api-key-header">
                        <div className="api-key-name">
                          {key.name || "Production API Key"}
                          <span className="api-key-badge">
                            {key.status || "active"}
                          </span>
                        </div>
                       
                      </div>
                      <div className="api-key-value">
                        <span className="api-key-masked">
                          {maskKey(key.key)}
                        </span>
                        <div className="api-key-actions">
                       
                          <button
                            className="api-key-action"
                            title="Copy key"
                            onClick={() => copyToClipboard(key.masked_key)}
                          >
                            <svg
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth={2}
                            >
                              <rect
                                x="9"
                                y="9"
                                width="13"
                                height="13"
                                rx="2"
                                ry="2"
                              />
                              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                            </svg>
                          </button>
                        </div>
                      </div>
                      <div className="api-key-meta">
                        <span>Created: {formatDate(key.created_at)}</span>
                        
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <CreateKeyModal
            isOpen={isModalOpen}
            onClose={() => setIsModalOpen(false)}
            onCreate={handleCreateKey}
          />
        </main>
      </div>
    </div>
  );
}
