"use client";

import { useState } from "react";

export default function CreateKeyModal({ isOpen, onClose, onCreate }) {
  const [name, setName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [newKey, setNewKey] = useState(null);
  const [copied, setCopied] = useState(false);

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
    setCopied(false);
    onClose();
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(newKey.api_key);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <>
      <style jsx>{`
        .modal-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.75);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 50;
          padding: 1rem;
          animation: fadeIn 0.2s ease-out;
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        .modal-card {
          background: #1e293b;
          border: 1px solid #334155;
          border-radius: 0.75rem;
          max-width: 28rem;
          width: 100%;
          padding: 1.5rem;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
          animation: slideUp 0.2s ease-out;
        }

        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(1rem);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .modal-title {
          font-size: 1.25rem;
          font-weight: 700;
          color: #f1f5f9;
          margin-bottom: 1.5rem;
        }

        .warning-box {
          background: rgba(234, 179, 8, 0.1);
          border: 1px solid rgba(234, 179, 8, 0.3);
          padding: 1rem;
          border-radius: 0.5rem;
          margin-bottom: 1rem;
        }

        .warning-text {
          font-size: 0.875rem;
          color: #fbbf24;
          font-weight: 500;
          display: flex;
          align-items: flex-start;
          gap: 0.5rem;
        }

        .warning-icon {
          width: 18px;
          height: 18px;
          flex-shrink: 0;
          margin-top: 1px;
        }

        .key-display {
          background: #0f172a;
          border: 1px solid #334155;
          padding: 0.875rem;
          border-radius: 0.5rem;
          word-break: break-all;
          font-family: 'Monaco', 'Courier New', monospace;
          font-size: 0.813rem;
          color: #e2e8f0;
          margin-bottom: 1.5rem;
        }

        .form-label {
          display: block;
          font-size: 0.875rem;
          font-weight: 500;
          color: #cbd5e1;
          margin-bottom: 0.5rem;
        }

        .form-input {
          width: 100%;
          padding: 0.625rem 0.875rem;
          background: #0f172a;
          border: 1px solid #334155;
          border-radius: 0.5rem;
          color: #e2e8f0;
          font-size: 0.875rem;
          transition: all 0.15s;
          box-sizing: border-box;
        }

        .form-input:focus {
          outline: none;
          border-color: #2563eb;
          box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .form-input::placeholder {
          color: #64748b;
        }

        .button-group {
          display: flex;
          justify-content: flex-end;
          gap: 0.75rem;
          margin-top: 1.5rem;
        }

        .button {
          padding: 0.625rem 1.25rem;
          border-radius: 0.5rem;
          font-size: 0.875rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.15s;
          border: none;
        }

        .button-secondary {
          background: transparent;
          color: #94a3b8;
          border: 1px solid #334155;
        }

        .button-secondary:hover {
          background: #0f172a;
          color: #cbd5e1;
        }

        .button-primary {
          background: #2563eb;
          color: white;
        }

        .button-primary:hover:not(:disabled) {
          background: #1d4ed8;
        }

        .button-primary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .button-copy {
          background: transparent;
          color: #3b82f6;
          border: 1px solid #3b82f6;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .button-copy:hover {
          background: #3b82f6;
          color: white;
        }

        .button-copy.copied {
          background: #22c55e;
          border-color: #22c55e;
          color: white;
        }

        .button-icon {
          width: 16px;
          height: 16px;
        }

        .form-group {
          margin-bottom: 1rem;
        }

        .space-y-4 > * + * {
          margin-top: 1rem;
        }
      `}</style>

      <div className="modal-overlay" onClick={handleClose}>
        <div className="modal-card" onClick={(e) => e.stopPropagation()}>
          <h2 className="modal-title">
            {newKey ? "API Key Created" : "Create New API Key"}
          </h2>

          {newKey ? (
            <div className="space-y-4">
              <div className="warning-box">
                <p className="warning-text">
                  <svg className="warning-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                  </svg>
                  <span>Please copy your API key now. You won't be able to see it again!</span>
                </p>
              </div>

              <div className="key-display">
                {newKey.api_key}
              </div>

              <div className="button-group">
                <button
                  onClick={handleCopy}
                  className={`button button-copy ${copied ? 'copied' : ''}`}
                >
                  {copied ? (
                    <>
                      <svg className="button-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      Copied!
                    </>
                  ) : (
                    <>
                      <svg className="button-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                      </svg>
                      Copy
                    </>
                  )}
                </button>
                <button
                  onClick={handleClose}
                  className="button button-primary"
                >
                  Done
                </button>
              </div>
            </div>
          ) : (
            <div>
              <div className="form-group">
                <label className="form-label">
                  Key Name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Production App"
                  className="form-input"
                  required
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && name.trim()) {
                      handleSubmit(e);
                    }
                  }}
                />
              </div>

              <div className="button-group">
                <button
                  onClick={handleClose}
                  className="button button-secondary"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={isSubmitting || !name.trim()}
                  className="button button-primary"
                >
                  {isSubmitting ? "Creating..." : "Create Key"}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}