"use client";

import { useState } from "react";

export default function ApiKeyTable({ apiKeys, onRevoke }) {
  const [revokingId, setRevokingId] = useState(null);

  const handleRevoke = async (keyId) => {
    if (
      !confirm(
        "Are you sure you want to revoke this API key? This action cannot be undone."
      )
    ) {
      return;
    }

    setRevokingId(keyId);
    try {
      await onRevoke(keyId);
    } catch (error) {
      alert("Failed to revoke key: " + error.message);
    } finally {
      setRevokingId(null);
    }
  };

  if (apiKeys.length === 0) {
    return (
      <div className="empty-state">
        <svg
          className="empty-state-icon"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"
          />
        </svg>
        <p className="empty-state-text">
          You haven&apos;t created any API keys yet.
        </p>
      </div>
    );
  }

  return (
    <div className="table-container">
      <table className="api-key-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Key</th>
            <th>Status</th>
            <th>Created</th>
            <th>Last Used</th>
            <th>Usage</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {apiKeys.map((key) => (
            <tr key={key.id}>
              <td className="font-medium">{key.name}</td>
              <td>
                <code className="masked-key">{key.masked_key}</code>
              </td>
              <td>
                <span
                  className={`status-badge ${
                    key.is_active ? "status-active" : "status-revoked"
                  }`}
                >
                  {key.is_active ? "Active" : "Revoked"}
                </span>
              </td>
              <td>{new Date(key.created_at).toLocaleDateString()}</td>
              <td>
                {key.last_used
                  ? new Date(key.last_used).toLocaleDateString()
                  : "Never"}
              </td>
              <td>{key.usage_count} requests</td>
              <td>
                {key.is_active ? (
                  <button
                    onClick={() => handleRevoke(key.id)}
                    disabled={revokingId === key.id}
                    className="button-danger button-sm"
                  >
                    {revokingId === key.id ? "Revoking..." : "Revoke"}
                  </button>
                ) : (
                  <span className="text-gray-400">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
