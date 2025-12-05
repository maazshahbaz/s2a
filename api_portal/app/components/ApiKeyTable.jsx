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
      <>
        <style jsx>{`
          .empty-state {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 0.75rem;
            padding: 4rem 2rem;
            text-align: center;
          }

          .empty-state-icon {
            width: 64px;
            height: 64px;
            color: #475569;
            margin: 0 auto 1rem;
          }

          .empty-state-text {
            color: #94a3b8;
            font-size: 0.875rem;
            margin: 0;
          }
        `}</style>

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
      </>
    );
  }

  return (
    <>
      <style jsx>{`
        .table-container {
          background: #1e293b;
          border: 1px solid #334155;
          border-radius: 0.75rem;
          overflow: hidden;
        }

        .api-key-table {
          width: 100%;
          border-collapse: collapse;
        }

        .api-key-table thead {
          background: #0f172a;
          border-bottom: 1px solid #334155;
        }

        .api-key-table th {
          padding: 0.875rem 1rem;
          text-align: left;
          font-size: 0.75rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: #94a3b8;
        }

        .api-key-table tbody tr {
          border-bottom: 1px solid #334155;
          transition: background-color 0.15s;
        }

        .api-key-table tbody tr:last-child {
          border-bottom: none;
        }

        .api-key-table tbody tr:hover {
          background: #0f172a;
        }

        .api-key-table td {
          padding: 1rem;
          font-size: 0.875rem;
          color: #e2e8f0;
        }

        .font-medium {
          font-weight: 500;
          color: #f1f5f9;
        }

        .masked-key {
          background: #0f172a;
          border: 1px solid #334155;
          border-radius: 0.375rem;
          padding: 0.25rem 0.5rem;
          font-family: 'Monaco', 'Courier New', monospace;
          font-size: 0.75rem;
          color: #cbd5e1;
        }

        .status-badge {
          display: inline-flex;
          align-items: center;
          padding: 0.25rem 0.625rem;
          border-radius: 9999px;
          font-size: 0.75rem;
          font-weight: 500;
        }

        .status-active {
          background: rgba(34, 197, 94, 0.1);
          color: #22c55e;
          border: 1px solid rgba(34, 197, 94, 0.2);
        }

        .status-revoked {
          background: rgba(239, 68, 68, 0.1);
          color: #ef4444;
          border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .button-danger {
          background: transparent;
          color: #ef4444;
          border: 1px solid #ef4444;
          padding: 0.375rem 0.75rem;
          border-radius: 0.375rem;
          font-size: 0.75rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.15s;
        }

        .button-danger:hover:not(:disabled) {
          background: #ef4444;
          color: white;
        }

        .button-danger:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .button-sm {
          font-size: 0.75rem;
        }

        .text-gray-400 {
          color: #94a3b8;
        }

        @media (max-width: 1024px) {
          .table-container {
            overflow-x: auto;
          }

          .api-key-table {
            min-width: 800px;
          }
        }
      `}</style>

      <div className="table-container">
        <table className="api-key-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Key</th>
              <th>Status</th>
              <th>Created At</th>
             
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
    </>
  );
}