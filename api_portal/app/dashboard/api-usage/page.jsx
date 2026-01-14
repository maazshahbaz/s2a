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

  const pythonScript = `import requests

def transcribe_audio(file_path, api_key):
    url = "https://bytepulseai.com/transcribe"
    
    headers = {
        "X-API-Key": api_key
    }
    
    # Open the file in binary mode
    with open(file_path, "rb") as audio_file:
        files = {
            "audio_file": audio_file
        }
        
        data = {
            "callback_url": "https://your-server.com/webhook",
            "enhance_audio": "true",
            "include_intelligence": "true"
        }
        
        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None

# Usage
job = transcribe_audio("meeting_recording.mp3", "your_api_key_here")
if job:
    print(f"Job started! ID: {job.get('job_id')}")`;

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

              <h5
                style={{
                  fontSize: "0.85rem",
                  color: "var(--color-text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: "1rem",
                  marginTop: "2rem",
                }}
              >
                Webhook Response
              </h5>
              <p
                style={{
                  color: "var(--color-text-secondary)",
                  fontSize: "0.9rem",
                  marginBottom: "1rem",
                  lineHeight: "1.6",
                }}
              >
                When the job is completed or failed, a POST request is sent to
                your <code>callback_url</code>.
              </p>
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
  "status": "completed",
  "error": null,
  "transcription": "This is the transcribed text from the audio file.",
  "ai_analysis": {
    "summary": "Meeting discussion...",
    "topics": ["Q3 Planning", "Budget"]
  },
  "diarized_transcription": "Speaker 1: Hello\\nSpeaker 2: Hi there"
}`}
                </pre>
              </div>
            </div>
          </div>
           <div className="section-card">
            <div className="section-header">
              <h3 className="section-title">Python API Example</h3>
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
        </main>
      </div>
    </div>
  );
}
