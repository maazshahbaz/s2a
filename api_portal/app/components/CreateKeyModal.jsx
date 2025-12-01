"use client";

import { useState } from "react";

export default function CreateKeyModal({ isOpen, onClose, onCreate }) {
  const [name, setName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [newKey, setNewKey] = useState(null);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      const result = await onCreate(name);
      setNewKey(result);
      setName("");
    } catch (error) {
      alert("Failed to create key: " + error.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    setNewKey(null);
    setName("");
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full p-6">
        <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">
          {newKey ? "API Key Created" : "Create New API Key"}
        </h2>

        {newKey ? (
          <div className="space-y-4">
            <div className="bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-700 p-4 rounded-md">
              <p className="text-sm text-yellow-800 dark:text-yellow-200 font-medium">
                Please copy your API key now. You won't be able to see it again!
              </p>
            </div>
            
            <div className="bg-gray-100 dark:bg-gray-900 p-3 rounded border border-gray-200 dark:border-gray-700 break-all font-mono text-sm">
              {newKey.api_key}
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => navigator.clipboard.writeText(newKey.api_key)}
                className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
              >
                Copy
              </button>
              <button
                onClick={handleClose}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Done
              </button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Key Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Production App"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                required
              />
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                type="button"
                onClick={handleClose}
                className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSubmitting}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              >
                {isSubmitting ? "Creating..." : "Create Key"}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
