"use client";

import { useSession } from "next-auth/react";
import { redirect } from "next/navigation";
import { useEffect, useState } from "react";
import { apiGet, apiPost } from "@/lib/api";
import Header from "../components/Header";
import Sidebar from "../components/Sidebar";
import ApiKeyTable from "../components/ApiKeyTable";
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
    await fetchKeys(); // Refresh list
    return newKey;
  };

  const handleRevokeKey = async (keyId) => {
    await apiPost(`/api-keys/${keyId}/revoke`, {});
    await fetchKeys(); // Refresh list
  };

  if (status === "loading" || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="text-xl text-gray-600 dark:text-gray-300">
          Loading...
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-layout">
      <Header />
      <div className="dashboard-main">
        <Sidebar />
        <main className="dashboard-content">
          <div className="content-header">
            <div>
              <h2 className="content-title">API Keys</h2>
              <p className="content-subtitle">
                Manage your API keys for accessing the service
              </p>
            </div>
            <button
              onClick={() => setIsModalOpen(true)}
              className="button-primary"
            >
              Create New API Key
            </button>
          </div>

          <ApiKeyTable apiKeys={apiKeys} onRevoke={handleRevokeKey} />

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
