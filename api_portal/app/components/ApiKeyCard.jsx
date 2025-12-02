"use client";

import { useState } from "react";

export default function ApiKeyCard({ apiKey, onRevoke }) {
  const [isRevoking, setIsRevoking] = useState(false);

  const handleRevoke = async () => {
    if (
      !confirm(
        "Are you sure you want to revoke this API key? This action cannot be undone."
      )
    ) {
      return;
    }

    setIsRevoking(true);
    try {
      await onRevoke(apiKey.id);
    } catch (error) {
      alert("Failed to revoke key: " + error.message);
    } finally {
      setIsRevoking(false);
    }
  };

  return (
    <div className="card mb-4">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            {apiKey.name}
          </h3>
          <div className="mt-1 font-mono text-sm text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-900 px-2 py-1 rounded inline-block">
            {apiKey.masked_key}
          </div>
        </div>
        <div
          className={`px-2 py-1 text-xs font-semibold rounded-full ${
            apiKey.is_active
              ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
              : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
          }`}
        >
          {apiKey.is_active ? "Active" : "Revoked"}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-500 dark:text-gray-400">Created:</span>
          <span className="ml-2 text-gray-900 dark:text-white">
            {new Date(apiKey.created_at).toLocaleDateString()}
          </span>
        </div>
        <div>
          <span className="text-gray-500 dark:text-gray-400">Last Used:</span>
          <span className="ml-2 text-gray-900 dark:text-white">
            {apiKey.last_used
              ? new Date(apiKey.last_used).toLocaleDateString()
              : "Never"}
          </span>
        </div>
        <div>
          <span className="text-gray-500 dark:text-gray-400">Usage:</span>
          <span className="ml-2 text-gray-900 dark:text-white">
            {apiKey.usage_count} requests
          </span>
        </div>
      </div>

      {apiKey.is_active && (
        <div className="mt-6 flex justify-end">
          <button
            onClick={handleRevoke}
            disabled={isRevoking}
            className="button-danger text-sm disabled:opacity-50"
          >
            {isRevoking ? "Revoking..." : "Revoke Key"}
          </button>
        </div>
      )}
    </div>
  );
}
