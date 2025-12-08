"use client";

import { useSession } from "next-auth/react";
import { redirect } from "next/navigation";
import { useState } from "react";
import Header from "../../components/Header";
import Sidebar from "../../components/Sidebar";

export default function SdkDocsPage() {
  const { data: session, status } = useSession({
    required: true,
    onUnauthenticated() {
      redirect("/login");
    },
  });
  const [activeTab, setActiveTab] = useState("javascript");
  const [installTab, setInstallTab] = useState("npm");

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  if (status === "loading") {
    return (
      <div className="dashboard-layout">
        <Sidebar />
        <div style={{ flex: 1, display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden" }}>
          <Header />
          <main
            className="dashboard-content"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <div style={{ color: "var(--color-text-secondary)", fontSize: "1.125rem" }}>
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
          <div className="content-header" style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1.5rem" }}>
            <h1 className="content-title">SDK Documentation</h1>
            <span style={{
              background: "var(--color-accent)",
              color: "#0a1a14",
              padding: "0.25rem 0.75rem",
              borderRadius: "9999px",
              fontSize: "0.75rem",
              fontWeight: 600,
            }}>
              v1.0.4
            </span>
          </div>
          <p className="content-subtitle" style={{ marginTop: "-1rem", marginBottom: "2rem" }}>
            Integrate our API into your application with our official SDK.
          </p>

          {/* External Links */}
          <div style={{ display: "flex", gap: "1rem", marginBottom: "2rem", flexWrap: "wrap" }}>
            <a
              href="https://www.npmjs.com/package/@99technologies/s2a-sdk"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
                padding: "1rem 1.5rem",
                background: "var(--color-bg-card)",
                border: "1px solid var(--color-border)",
                borderRadius: "12px",
                color: "var(--color-text-primary)",
                textDecoration: "none",
                flex: 1,
                minWidth: "200px",
              }}
            >
              <div style={{
                width: "40px",
                height: "40px",
                background: "linear-gradient(135deg, #cb3837, #cb3837)",
                borderRadius: "8px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}>
                <svg viewBox="0 0 24 24" fill="white" style={{ width: 20, height: 20 }}>
                  <path d="M0 7.334v8h6.666v1.332H12v-1.332h12v-8H0zm6.666 6.664H5.334v-4H3.999v4H1.335V8.667h5.331v5.331zm4 0v1.336H8.001V8.667h5.334v5.332h-2.669v-.001zm12.001 0h-1.33v-4H20v4h-1.336v-4h-1.335v4h-1.33V8.667h6.668v5.331z"/>
                </svg>
              </div>
              <div>
                <div style={{ fontWeight: 600 }}>NPM Package</div>
                <div style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>@99technologies/s2a-sdk</div>
              </div>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} style={{ width: 16, height: 16, marginLeft: "auto", color: "var(--color-text-muted)" }}>
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3" />
              </svg>
            </a>

            <a
              href="https://pypi.org/project/s2a-sdk/"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
                padding: "1rem 1.5rem",
                background: "var(--color-bg-card)",
                border: "1px solid var(--color-border)",
                borderRadius: "12px",
                color: "var(--color-text-primary)",
                textDecoration: "none",
                flex: 1,
                minWidth: "200px",
              }}
            >
              <div style={{
                width: "40px",
                height: "40px",
                background: "linear-gradient(135deg, #3776ab, #3776ab)",
                borderRadius: "8px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}>
                <svg viewBox="0 0 24 24" fill="white" style={{ width: 20, height: 20 }}>
                  <path d="M14.31.18l.9.2.73.26.59.3.45.32.34.34.25.34.16.33.1.3.04.26.02.2-.01.13V8.5l-.05.63-.13.55-.21.46-.26.38-.3.31-.33.25-.35.19-.35.14-.33.1-.3.07-.26.04-.21.02H8.83l-.69.05-.59.14-.5.22-.41.27-.33.32-.27.35-.2.36-.15.37-.1.35-.07.32-.04.27-.02.21v3.06H3.23l-.21-.03-.28-.07-.32-.12-.35-.18-.36-.26-.36-.36-.35-.46-.32-.59-.28-.73-.21-.88-.14-1.05L0 11.97l.06-1.22.16-1.04.24-.87.32-.71.36-.57.4-.44.42-.33.42-.24.4-.16.36-.1.32-.05.24-.01h.16l.06.01h8.16v-.83H6.24l-.01-2.75-.02-.37.05-.34.11-.31.17-.28.25-.26.31-.23.38-.2.44-.18.51-.15.58-.12.64-.1.71-.06.77-.04.84-.02 1.27.05 1.07.13zm-6.3 1.98l-.23.33-.08.41.08.41.23.34.33.22.41.09.41-.09.33-.22.23-.34.08-.41-.08-.41-.23-.33-.33-.22-.41-.09-.41.09-.33.22z"/>
                </svg>
              </div>
              <div>
                <div style={{ fontWeight: 600 }}>PyPI Package</div>
                <div style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>s2a-sdk</div>
              </div>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} style={{ width: 16, height: 16, marginLeft: "auto", color: "var(--color-text-muted)" }}>
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3" />
              </svg>
            </a>

            <a
              href="https://github.com/99Technologies-ai/s2a"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
                padding: "1rem 1.5rem",
                background: "var(--color-bg-card)",
                border: "1px solid var(--color-border)",
                borderRadius: "12px",
                color: "var(--color-text-primary)",
                textDecoration: "none",
                flex: 1,
                minWidth: "200px",
              }}
            >
              <div style={{
                width: "40px",
                height: "40px",
                background: "linear-gradient(135deg, #333, #333)",
                borderRadius: "8px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}>
                <svg viewBox="0 0 24 24" fill="white" style={{ width: 20, height: 20 }}>
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
              </div>
              <div>
                <div style={{ fontWeight: 600 }}>GitHub</div>
                <div style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>View source</div>
              </div>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} style={{ width: 16, height: 16, marginLeft: "auto", color: "var(--color-text-muted)" }}>
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3" />
              </svg>
            </a>
          </div>

          {/* Language Toggle */}
          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "2rem" }}>
            <button
              onClick={() => setActiveTab("javascript")}
              style={{
                padding: "0.5rem 1rem",
                background: activeTab === "javascript" ? "var(--color-accent)" : "var(--color-bg-tertiary)",
                color: activeTab === "javascript" ? "#0a1a14" : "var(--color-text-primary)",
                border: "none",
                borderRadius: "8px",
                cursor: "pointer",
                fontWeight: 500,
                fontSize: "0.875rem",
              }}
            >
              JavaScript
            </button>
            <button
              onClick={() => setActiveTab("python")}
              style={{
                padding: "0.5rem 1rem",
                background: activeTab === "python" ? "var(--color-accent)" : "var(--color-bg-tertiary)",
                color: activeTab === "python" ? "#0a1a14" : "var(--color-text-primary)",
                border: "none",
                borderRadius: "8px",
                cursor: "pointer",
                fontWeight: 500,
                fontSize: "0.875rem",
              }}
            >
              Python
            </button>
          </div>

          {/* Installation Section */}
          <section style={{ marginBottom: "2.5rem" }}>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 600, marginBottom: "1rem", color: "var(--color-text-primary)" }}>
              Installation
            </h2>

            {activeTab === "javascript" && (
              <>
                <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
                  {["npm", "yarn", "pnpm"].map((pkg) => (
                    <button
                      key={pkg}
                      onClick={() => setInstallTab(pkg)}
                      style={{
                        padding: "0.375rem 0.75rem",
                        background: installTab === pkg ? "var(--color-bg-card)" : "transparent",
                        color: installTab === pkg ? "var(--color-text-primary)" : "var(--color-text-muted)",
                        border: "none",
                        borderRadius: "6px",
                        cursor: "pointer",
                        fontSize: "0.8125rem",
                      }}
                    >
                      {pkg}
                    </button>
                  ))}
                </div>
                <div style={{ position: "relative" }}>
                  <pre className="code-block" style={{ margin: 0, padding: "1rem 1.25rem" }}>
                    <code>
                      {installTab === "npm" && "npm install @99technologies/s2a-sdk"}
                      {installTab === "yarn" && "yarn add @99technologies/s2a-sdk"}
                      {installTab === "pnpm" && "pnpm add @99technologies/s2a-sdk"}
                    </code>
                  </pre>
                  <button
                    onClick={() => copyToClipboard(
                      installTab === "npm" ? "npm install @99technologies/s2a-sdk" :
                      installTab === "yarn" ? "yarn add @99technologies/s2a-sdk" :
                      "pnpm add @99technologies/s2a-sdk"
                    )}
                    style={{
                      position: "absolute",
                      right: "0.75rem",
                      top: "50%",
                      transform: "translateY(-50%)",
                      background: "transparent",
                      border: "none",
                      color: "var(--color-text-muted)",
                      cursor: "pointer",
                      padding: "0.5rem",
                    }}
                    title="Copy"
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} style={{ width: 16, height: 16 }}>
                      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                    </svg>
                  </button>
                </div>
              </>
            )}

            {activeTab === "python" && (
              <div style={{ position: "relative" }}>
                <pre className="code-block" style={{ margin: 0, padding: "1rem 1.25rem" }}>
                  <code>pip install s2a-sdk</code>
                </pre>
                <button
                  onClick={() => copyToClipboard("pip install s2a-sdk")}
                  style={{
                    position: "absolute",
                    right: "0.75rem",
                    top: "50%",
                    transform: "translateY(-50%)",
                    background: "transparent",
                    border: "none",
                    color: "var(--color-text-muted)",
                    cursor: "pointer",
                    padding: "0.5rem",
                  }}
                  title="Copy"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} style={{ width: 16, height: 16 }}>
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                </button>
              </div>
            )}
          </section>

          {/* Initialization Section */}
          <section style={{ marginBottom: "2.5rem" }}>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 600, marginBottom: "0.5rem", color: "var(--color-text-primary)" }}>
              Initialization
            </h2>
            <p style={{ color: "var(--color-text-secondary)", fontSize: "0.9rem", marginBottom: "1rem" }}>
              Initialize the SDK with your API key. You can find your API key in the{" "}
              <a href="/dashboard/api-keys" style={{ color: "var(--color-accent)" }}>API Keys</a> section.
            </p>

            <div style={{
              background: "var(--color-bg-card)",
              border: "1px solid var(--color-border)",
              borderRadius: "12px",
              overflow: "hidden",
            }}>
              <div style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "0.75rem 1rem",
                borderBottom: "1px solid var(--color-border)",
              }}>
                <span style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>Initialize Client</span>
                <span style={{ color: "var(--color-accent)", fontSize: "0.75rem", fontWeight: 500 }}>
                  {activeTab === "javascript" ? "JavaScript" : "Python"}
                </span>
              </div>
              <div style={{ position: "relative" }}>
                <pre className="code-block" style={{ margin: 0, borderRadius: 0 }}>
                  <code>
                    {activeTab === "javascript" ? `import { S2AClient } from '@99technologies/s2a-sdk';

// Initialize the client with your API key
const client = new S2AClient({
  apiKey: process.env.S2A_API_KEY,
});

// Make your first request
const job = await client.transcribeAsync('meeting.mp3', {
  callbackUrl: 'https://yourapp.com/webhook'
});

console.log(\`Job ID: \${job.jobId}\`);` : `from s2a_sdk import S2AClient

# Initialize the client with your API key
client = S2AClient(api_key="bp-proj-your-api-key")

# Make your first request
job = client.transcribe_async(
    "meeting.mp3",
    callback_url="https://yourapp.com/webhook"
)

print(f"Job ID: {job.job_id}")`}
                  </code>
                </pre>
                <button
                  onClick={() => copyToClipboard(activeTab === "javascript" ?
                    `import { S2AClient } from '@99technologies/s2a-sdk';

const client = new S2AClient({
  apiKey: process.env.S2A_API_KEY,
});

const job = await client.transcribeAsync('meeting.mp3', {
  callbackUrl: 'https://yourapp.com/webhook'
});

console.log(\`Job ID: \${job.jobId}\`);` :
                    `from s2a_sdk import S2AClient

client = S2AClient(api_key="bp-proj-your-api-key")

job = client.transcribe_async(
    "meeting.mp3",
    callback_url="https://yourapp.com/webhook"
)

print(f"Job ID: {job.job_id}")`
                  )}
                  style={{
                    position: "absolute",
                    right: "0.75rem",
                    top: "0.75rem",
                    background: "var(--color-bg-tertiary)",
                    border: "1px solid var(--color-border)",
                    borderRadius: "6px",
                    color: "var(--color-text-muted)",
                    cursor: "pointer",
                    padding: "0.5rem",
                  }}
                  title="Copy code"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} style={{ width: 14, height: 14 }}>
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                </button>
              </div>
            </div>
          </section>

          {/* Key Features */}
          <section style={{ marginBottom: "2.5rem" }}>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 600, marginBottom: "1.5rem", color: "var(--color-text-primary)" }}>
              Key Features
            </h2>
            <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
              <div style={{
                padding: "1.25rem",
                background: "var(--color-bg-card)",
                border: "1px solid var(--color-border)",
                borderRadius: "12px",
              }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.5rem", color: "var(--color-text-primary)" }}>
                  🚀 Multi-Stage Intelligence
                </h3>
                <p style={{ fontSize: "0.875rem", color: "var(--color-text-secondary)", margin: 0 }}>
                  Quick Intelligence (1-2s) for real-time insights, Enhanced Intelligence (5-15s) for comprehensive 50+ field analysis.
                </p>
              </div>
              <div style={{
                padding: "1.25rem",
                background: "var(--color-bg-card)",
                border: "1px solid var(--color-border)",
                borderRadius: "12px",
              }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.5rem", color: "var(--color-text-primary)" }}>
                  🎯 Business Intelligence
                </h3>
                <p style={{ fontSize: "0.875rem", color: "var(--color-text-secondary)", margin: 0 }}>
                  Extract action items, entities, financial data, and conversation metrics automatically.
                </p>
              </div>
              <div style={{
                padding: "1.25rem",
                background: "var(--color-bg-card)",
                border: "1px solid var(--color-border)",
                borderRadius: "12px",
              }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.5rem", color: "var(--color-text-primary)" }}>
                  ⚡ Async Processing
                </h3>
                <p style={{ fontSize: "0.875rem", color: "var(--color-text-secondary)", margin: 0 }}>
                  Process audio up to 5 hours with webhook callbacks for long-running operations.
                </p>
              </div>
            </div>
          </section>

          {/* API Methods */}
          <section style={{ marginBottom: "2.5rem" }}>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 600, marginBottom: "1.5rem", color: "var(--color-text-primary)" }}>
              Core Methods
            </h2>
            
            <div style={{
              background: "var(--color-bg-card)",
              border: "1px solid var(--color-border)",
              borderRadius: "12px",
              overflow: "hidden",
              marginBottom: "1rem",
            }}>
              <div style={{ padding: "1rem 1.25rem", borderBottom: "1px solid var(--color-border)" }}>
                <code style={{ color: "var(--color-accent)", fontFamily: "var(--font-mono)", fontSize: "0.9rem" }}>
                  {activeTab === "javascript" ? "transcribeAsync(audioFile, options)" : "transcribe_async(audio_file, callback_url, **options)"}
                </code>
              </div>
              <div style={{ padding: "1rem 1.25rem" }}>
                <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", marginBottom: "1rem" }}>
                  Asynchronous transcription for audio files between 1 second and 5 hours.
                </p>
                <div style={{ fontSize: "0.8125rem" }}>
                  <strong style={{ color: "var(--color-text-primary)" }}>Parameters:</strong>
                  <ul style={{ color: "var(--color-text-secondary)", marginTop: "0.5rem", paddingLeft: "1.25rem" }}>
                    <li><code>callbackUrl</code> (required) - URL to receive webhook notifications</li>
                    <li><code>enhanceAudio</code> - Enable audio enhancement (default: true)</li>
                    <li><code>priority</code> - Processing priority (LOW, NORMAL, HIGH)</li>
                  </ul>
                </div>
              </div>
            </div>

            <div style={{
              background: "var(--color-bg-card)",
              border: "1px solid var(--color-border)",
              borderRadius: "12px",
              overflow: "hidden",
            }}>
              <div style={{ padding: "1rem 1.25rem", borderBottom: "1px solid var(--color-border)" }}>
                <code style={{ color: "var(--color-accent)", fontFamily: "var(--font-mono)", fontSize: "0.9rem" }}>
                  {activeTab === "javascript" ? "extractIntelligence(transcript, options)" : "extract_intelligence(transcript, **options)"}
                </code>
              </div>
              <div style={{ padding: "1rem 1.25rem" }}>
                <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", marginBottom: "1rem" }}>
                  Extract comprehensive business intelligence from existing transcripts.
                </p>
                <div style={{ fontSize: "0.8125rem" }}>
                  <strong style={{ color: "var(--color-text-primary)" }}>Parameters:</strong>
                  <ul style={{ color: "var(--color-text-secondary)", marginTop: "0.5rem", paddingLeft: "1.25rem" }}>
                    <li><code>mode</code> - Intelligence mode (AUTO_DETECT, SALES, SUPPORT, GENERAL)</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
