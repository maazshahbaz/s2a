"use client";

import { useSession } from "next-auth/react";
import { redirect } from "next/navigation";
import { useEffect, useState } from "react";
import { apiGet, apiPost } from "@/lib/api";
import Header from "../../components/Header";
import Sidebar from "../../components/Sidebar";
import CreateKeyModal from "../../components/CreateKeyModal";

export default function ApiKeysPage() {
  const { data: session, status } = useSession({
    required: true,
    onUnauthenticated() {
      redirect("/login");
    },
  });
  const [loading, setLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [apiKeys, setApiKeys] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedEnvironment, setSelectedEnvironment] = useState("all");

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
    if (!dateString) return "Never";
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  const formatLastUsed = (dateString) => {
    if (!dateString) return "Never";
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 60) return `${diffMins} mins ago`;
    if (diffHours < 24) return `${diffHours} hours ago`;
    return `${diffDays} days ago`;
  };

  const maskKey = (key) => {
    if (!key) return "sk_live_...****";
    const prefix = key.substring(0, 7);
    const suffix = key.slice(-4);
    return `${prefix}...${suffix}`;
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  const filteredKeys = apiKeys.filter((key) => {
    const matchesSearch = key.name?.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

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
      <div style={{ flex: 1, display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden" }}>
        <Header />
        <main className="dashboard-content" style={{ flex: 1, overflow: "auto", height: "calc(100vh - 60px)" }}>
          {/* Page Header */}
          <div className="content-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div>
              <h1 className="content-title">Auth Keys</h1>
              <p className="content-subtitle">
                Create and manage auth keys for your applications.
              </p>
            </div>
            <button
              className="button-primary"
              onClick={() => setIsModalOpen(true)}
              style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
                style={{ width: 16, height: 16 }}
              >
                <path d="M12 5v14M5 12h14" />
              </svg>
              Create Key
            </button>
          </div>

          {/* Search and Filter */}
          <div style={{ display: "flex", gap: "1rem", marginBottom: "1.5rem" }}>
            <div style={{ position: "relative", flex: 1, maxWidth: "320px" }}>
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
                style={{
                  width: 18,
                  height: 18,
                  position: "absolute",
                  left: "12px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  color: "var(--color-text-muted)",
                }}
              >
                <circle cx="11" cy="11" r="8" />
                <path d="m21 21-4.3-4.3" />
              </svg>
              <input
                type="text"
                placeholder="Search keys..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  width: "100%",
                  padding: "0.625rem 0.75rem 0.625rem 2.5rem",
                  background: "var(--color-bg-tertiary)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "8px",
                  color: "var(--color-text-primary)",
                  fontSize: "0.875rem",
                }}
              />
            </div>
          </div>

          {/* API Keys List */}
          <div className="api-key-list">
            {filteredKeys.length === 0 ? (
              <div
                className="empty-state"
                style={{ border: "none", padding: "3rem" }}
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
                  No API keys found. Create your first key to get started.
                </p>
              </div>
            ) : (
              filteredKeys.map((key) => (
                <div key={key.id} className="api-key-card">
                  <div className="api-key-header">
                    <div className="api-key-name">
                      {key.name || "Unnamed Key"}
                      <span className="api-key-badge">
                        {key.is_active ? "active" : "revoked"}
                      </span>
                    </div>
                    <button className="api-key-menu">
                      <svg
                        viewBox="0 0 24 24"
                        fill="currentColor"
                        style={{ width: 16, height: 16 }}
                      >
                        <circle cx="12" cy="5" r="1.5" />
                        <circle cx="12" cy="12" r="1.5" />
                        <circle cx="12" cy="19" r="1.5" />
                      </svg>
                    </button>
                  </div>
                  <div className="api-key-value">
                    <span className="api-key-masked">
                      {maskKey(key.masked_key || key.key)}
                    </span>
                    <div className="api-key-actions">
                      <button className="api-key-action" title="View key">
                        <svg
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth={2}
                        >
                          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                          <circle cx="12" cy="12" r="3" />
                        </svg>
                      </button>
                      <button
                        className="api-key-action"
                        title="Copy key"
                        onClick={() => copyToClipboard(key.masked_key || key.key)}
                      >
                        <svg
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth={2}
                        >
                          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  <div className="api-key-meta">
                    <span>Created: {formatDate(key.created_at)}</span>
                    <span>Last used: {formatLastUsed(key.last_used)}</span>
                  </div>
                </div>
              ))
            )}
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
