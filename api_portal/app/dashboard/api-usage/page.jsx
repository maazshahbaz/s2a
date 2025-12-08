"use client";

import { useSession } from "next-auth/react";
import { redirect } from "next/navigation";
import { useState } from "react";
import Header from "../../components/Header";
import Sidebar from "../../components/Sidebar";

export default function ApiUsagePage() {
  const { data: session, status } = useSession({
    required: true,
    onUnauthenticated() {
      redirect("/login");
    },
  });

  const [copied, setCopied] = useState(false);

  const pythonScript = `#!/usr/bin/env python3
"""
Test script for S2A Python SDK
Tests transcribe sync, async, and get status functions
"""

import asyncio
import os
from s2a_sdk import S2AClient

def test_sync_transcribe():
    """Test synchronous transcription"""
    print("Testing Python SDK - Sync Transcribe...")
    
    # Initialize client (you'll need to set your API key)
    client = S2AClient(
        api_key=os.getenv("S2A_API_KEY", "bp-proj-bWaL9OAgxq2DWYKeMQ3qEV9Hf3tzH98mvgxarhLN26I"),
        base_url=os.getenv("S2A_BASE_URL", "https://bytepulseai.com")
    )
    
    try:
        # Test with a dummy audio file path (you can replace with actual file)
        # Note: This will likely fail without a real server, but tests the SDK structure
        result = client.transcribe(
            audio_file="veryShort.wav",
            enhance_audio=True
        )
        print(f"Sync transcribe result: {result}")
        return True
    except Exception as e:
        print(f"Sync transcribe error (expected without server): {e}")
        return False

async def test_async_transcribe():
    """Test asynchronous transcription"""
    print("Testing Python SDK - Async Transcribe...")
    
    client = S2AClient(
        api_key=os.getenv("S2A_API_KEY", "bp-proj-Ec6V8XFRbWdagZWIhiy5bBVlMrC2B0T3RimpvqV7a2A"), 
        base_url=os.getenv("S2A_BASE_URL", "https://bytepulseai.com")
    )
    
    try:
        # Test async transcription
        job = client.transcribe_async(
            audio_file="47minslong.wav",
            callback_url="https://webhook-test.com/0ae27b4d5fd100fec9cc90b7cb84bf8a",
            enhance_audio=True
        )
        print(f"Async transcribe job: {job}")
        return job.job_id if hasattr(job, 'job_id') else None
    except Exception as e:
        print(f"Async transcribe error (expected without server): {e}")
        return None

async def test_get_status(job_id):
    """Test getting transcription status"""
    print("Testing Python SDK - Get Status...")
    
    client = S2AClient(
        api_key=os.getenv("S2A_API_KEY", "bp-proj-bWaL9OAgxq2DWYKeMQ3qEV9Hf3tzH98mvgxarhLN26I"),
        base_url=os.getenv("S2A_BASE_URL", "https://bytepulseai.com")
    )
    
    try:
        if job_id:
            print('why')
            status = client.get_job_status(job_id)
            print(f"Status result: {status}")
        else:
            # Test with dummy job ID
            status = client.get_job_status("test-job-123")
            print(f"Status result: {status}")
        return True
    except Exception as e:
        print(f"Get status error (expected without server): {e}")
        return False

def test_import():
    """Test that the SDK can be imported and initialized"""
    print("Testing Python SDK - Import and Initialize...")
    
    try:
        from s2a_sdk import S2AClient
        client = S2AClient(api_key="bp-test", base_url="http://localhost:8000")
        print("✓ SDK imported and client initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Import/initialization failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=== S2A Python SDK Tests ===\\n")
    
    # Test basic import and initialization
    import_success = test_import()
    
    if not import_success:
        print("Failed to import SDK, stopping tests")
        return
    
    # Test async transcribe and status
    async def run_jobs():
        for i in range(2):  # 0..39
            job_id = await test_async_transcribe()  # Await the async function
            print(job_id, 'job_id')
    
    # ✅ Correct: just await the coroutine instead of asyncio.run()
    await run_jobs()
    
    print("=== Python SDK Tests Complete ===")

if __name__ == "__main__":
    # Only one asyncio.run() at the top level
    asyncio.run(main())
`;

  const copyToClipboard = () => {
    navigator.clipboard.writeText(pythonScript);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (status === "loading") {
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
          <div className="content-header">
            <h1 className="content-title">API Usage</h1>
            <p className="content-subtitle">
              Learn how to integrate the S2A SDK into your Python applications.
            </p>
          </div>

          <div className="section-card">
            <div className="section-header">
              <h3 className="section-title">Python SDK Demo</h3>
              <div className="quickstart-lang">Python</div>
            </div>

            <div className="quickstart-code" style={{ padding: 0 }}>
              <div style={{ position: "relative" }}>
                <button
                  className="quickstart-copy"
                  onClick={copyToClipboard}
                  style={{
                    position: "absolute",
                    top: "1rem",
                    right: "1rem",
                    zIndex: 10,
                  }}
                  title="Copy code"
                >
                  {copied ? (
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="var(--color-success)"
                      strokeWidth={2}
                      style={{ width: 14, height: 14 }}
                    >
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  ) : (
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={2}
                      style={{ width: 14, height: 14 }}
                    >
                      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                    </svg>
                  )}
                </button>
                <div
                  className="code-block"
                  style={{
                    borderRadius: "0 0 12px 12px",
                    margin: 0,
                    maxHeight: "calc(100vh - 300px)",
                    overflowY: "auto",
                    backgroundColor: "var(--color-bg-tertiary)",
                    fontSize: "0.85rem",
                  }}
                >
                  <pre style={{ margin: 0, fontFamily: "var(--font-mono)" }}>
                    {pythonScript}
                  </pre>
                </div>
              </div>
            </div>
          </div>

          {/* API Reference Section */}
          <div className="section-card" style={{ marginTop: "2rem" }}>
            <div className="section-header">
              <h3 className="section-title">API Reference</h3>
            </div>

            <div className="api-doc-content" style={{ paddingBottom: "1rem" }}>
              <h4
                style={{
                  fontSize: "0.95rem",
                  fontWeight: "600",
                  color: "var(--color-text-primary)",
                  marginBottom: "1rem",
                  marginTop: "0",
                }}
              >
                Async Transcription
              </h4>

              <div
                style={{
                  display: "flex",
                  gap: "0.75rem",
                  alignItems: "center",
                  marginBottom: "1.5rem",
                }}
              >
                <span
                  style={{
                    background: "var(--color-success-bg)",
                    color: "var(--color-success)",
                    padding: "0.25rem 0.6rem",
                    borderRadius: "6px",
                    fontSize: "0.75rem",
                    fontWeight: "700",
                  }}
                >
                  POST
                </span>
                <code
                  style={{
                    fontFamily: "var(--font-mono)",
                    color: "var(--color-text-primary)",
                    background: "var(--color-bg-tertiary)",
                    padding: "0.25rem 0.6rem",
                    borderRadius: "6px",
                    fontSize: "0.85rem",
                  }}
                >
                  /transcribe
                </code>
              </div>

              <p
                style={{
                  color: "var(--color-text-secondary)",
                  fontSize: "0.9rem",
                  marginBottom: "1.5rem",
                  lineHeight: "1.6",
                }}
              >
                Initiate an asynchronous transcription job for audio/video
                files. This endpoint handles large files and processes them in
                the background.
              </p>

              <h5
                style={{
                  fontSize: "0.85rem",
                  color: "var(--color-text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: "1rem",
                }}
              >
                Headers
              </h5>
              <div className="table-container" style={{ marginBottom: "2rem" }}>
                <table className="api-key-table">
                  <thead>
                    <tr>
                      <th style={{ width: "30%" }}>Name</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          color: "var(--color-accent)",
                        }}
                      >
                        X-API-Key
                      </td>
                      <td>Your secret API key</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <h5
                style={{
                  fontSize: "0.85rem",
                  color: "var(--color-text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: "1rem",
                }}
              >
                Form Data Parameters
              </h5>
              <div className="table-container" style={{ marginBottom: "2rem" }}>
                <table className="api-key-table">
                  <thead>
                    <tr>
                      <th style={{ width: "30%" }}>Parameter</th>
                      <th style={{ width: "20%" }}>Type</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          color: "var(--color-text-primary)",
                        }}
                      >
                        audio_file{" "}
                        <span
                          style={{
                            color: "var(--color-danger)",
                            marginLeft: "0.25rem",
                          }}
                        >
                          *
                        </span>
                      </td>
                      <td style={{ color: "var(--color-text-secondary)" }}>
                        File
                      </td>
                      <td>Audio or video file to transcribe.</td>
                    </tr>
                    <tr>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          color: "var(--color-text-primary)",
                        }}
                      >
                        callback_url{" "}
                        <span
                          style={{
                            color: "var(--color-danger)",
                            marginLeft: "0.25rem",
                          }}
                        >
                          *
                        </span>
                      </td>
                      <td style={{ color: "var(--color-text-secondary)" }}>
                        String
                      </td>
                      <td>Webhook URL where results will be sent.</td>
                    </tr>
                    <tr>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          color: "var(--color-text-primary)",
                        }}
                      >
                        enhance_audio
                      </td>
                      <td style={{ color: "var(--color-text-secondary)" }}>
                        Boolean
                      </td>
                      <td>
                        Apply audio enhancement. Default: <code>true</code>
                      </td>
                    </tr>
                    <tr>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          color: "var(--color-text-primary)",
                        }}
                      >
                        remove_silence
                      </td>
                      <td style={{ color: "var(--color-text-secondary)" }}>
                        Boolean
                      </td>
                      <td>
                        Remove silent segments. Default: <code>false</code>
                      </td>
                    </tr>
                    <tr>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          color: "var(--color-text-primary)",
                        }}
                      >
                        include_intelligence
                      </td>
                      <td style={{ color: "var(--color-text-secondary)" }}>
                        Boolean
                      </td>
                      <td>
                        Perform analysis. Default: <code>true</code>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <h5
                style={{
                  fontSize: "0.85rem",
                  color: "var(--color-text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: "1rem",
                }}
              >
                Response
              </h5>
              <div
                className="code-block"
                style={{
                  backgroundColor: "var(--color-bg-tertiary)",
                  padding: "1rem",
                  borderRadius: "8px",
                }}
              >
                <pre
                  style={{
                    margin: 0,
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.85rem",
                  }}
                >
                  {`{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "accepted"
}`}
                </pre>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
