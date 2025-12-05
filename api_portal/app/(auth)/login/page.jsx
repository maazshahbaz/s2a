"use client";
import { signIn } from "next-auth/react";

export default function LoginPage() {
  return (
    <>
      <style jsx>{`
        .login-container {
          min-height: 100vh;
          background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f172a 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 1rem;
          position: relative;
          overflow: hidden;
        }

        .bg-orb {
          position: absolute;
          border-radius: 50%;
          filter: blur(60px);
          opacity: 0.2;
          animation: float 8s ease-in-out infinite;
        }

        .orb-1 {
          width: 300px;
          height: 300px;
          background: #3b82f6;
          top: 10%;
          left: 5%;
          animation-delay: 0s;
        }

        .orb-2 {
          width: 300px;
          height: 300px;
          background: #8b5cf6;
          top: 20%;
          right: 5%;
          animation-delay: 2s;
        }

        .orb-3 {
          width: 300px;
          height: 300px;
          background: #06b6d4;
          bottom: -5%;
          left: 10%;
          animation-delay: 4s;
        }

        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-30px); }
        }

        .login-card-wrapper {
          position: relative;
          width: 100%;
          max-width: 450px;
        }

        .card-glow {
          position: absolute;
          inset: 0;
          background: linear-gradient(to right, #3b82f6, #8b5cf6);
          border-radius: 1.5rem;
          filter: blur(40px);
          opacity: 0.3;
        }

        .login-card {
          position: relative;
          background: rgba(30, 41, 59, 0.9);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(148, 163, 184, 0.3);
          border-radius: 1.5rem;
          padding: 2.5rem;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .logo-container {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 64px;
          height: 64px;
          background: linear-gradient(135deg, #3b82f6, #8b5cf6);
          border-radius: 1rem;
          margin-bottom: 1.5rem;
          box-shadow: 0 10px 30px rgba(59, 130, 246, 0.4);
        }

        .logo-icon {
          width: 32px;
          height: 32px;
          color: white;
        }

        .title {
          font-size: 2rem;
          font-weight: 700;
          color: white;
          margin-bottom: 0.5rem;
          text-align: center;
        }

        .subtitle {
          color: #94a3b8;
          text-align: center;
          margin-bottom: 2rem;
        }

        .features-list {
          margin-bottom: 2rem;
        }

        .feature-item {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          margin-bottom: 0.75rem;
          color: #cbd5e1;
          font-size: 0.875rem;
        }

        .feature-icon {
          width: 32px;
          height: 32px;
          background: rgba(59, 130, 246, 0.1);
          border-radius: 0.5rem;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .feature-icon svg {
          width: 16px;
          height: 16px;
        }

        .shield-icon {
          color: #60a5fa;
        }

        .code-icon {
          color: #a78bfa;
        }

        .sign-in-button {
          width: 100%;
          background: linear-gradient(to right, #2563eb, #1d4ed8);
          color: white;
          font-weight: 600;
          padding: 1rem 1.5rem;
          border-radius: 0.75rem;
          border: none;
          cursor: pointer;
          transition: all 0.2s;
          box-shadow: 0 10px 25px rgba(37, 99, 235, 0.3);
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.75rem;
          font-size: 1rem;
        }

        .sign-in-button:hover {
          background: linear-gradient(to right, #1d4ed8, #1e40af);
          box-shadow: 0 15px 35px rgba(37, 99, 235, 0.5);
          transform: translateY(-2px);
        }

        .sign-in-button:active {
          transform: translateY(0);
        }

        .microsoft-icon {
          width: 20px;
          height: 20px;
        }

        .arrow {
          margin-left: auto;
          transition: transform 0.2s;
        }

        .sign-in-button:hover .arrow {
          transform: translateX(4px);
        }

        .footer {
          margin-top: 1.5rem;
          padding-top: 1.5rem;
          border-top: 1px solid rgba(148, 163, 184, 0.3);
        }

        .footer-text {
          text-align: center;
          font-size: 0.75rem;
          color: #64748b;
        }

   
   
      `}</style>

      <div className="login-container">
        <div className="bg-orb orb-1"></div>
        <div className="bg-orb orb-2"></div>
        <div className="bg-orb orb-3"></div>

        <div className="login-card-wrapper">
          <div className="card-glow"></div>
          
          <div className="login-card">
            <div style={{ textAlign: 'center' }}>
              <div className="logo-container">
                <svg className="logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="16 18 22 12 16 6"></polyline>
                  <polyline points="8 6 2 12 8 18"></polyline>
                </svg>
              </div>
              <h1 className="title">API Portal</h1>
              <p className="subtitle">Sign in to access your developer dashboard</p>
            </div>

           

            <button
              onClick={() => signIn("azure-ad", { callbackUrl: "/auth-redirect" })}
              className="sign-in-button"
            >
              <svg className="microsoft-icon" viewBox="0 0 23 23" fill="currentColor">
                <path d="M11 0H0V11H11V0Z"/>
                <path d="M23 0H12V11H23V0Z"/>
                <path d="M11 12H0V23H11V12Z"/>
                <path d="M23 12H12V23H23V12Z"/>
              </svg>
              <span>Sign in with Microsoft</span>
              <span className="arrow">→</span>
            </button>

            <div className="footer">
              <p className="footer-text">
                By signing in, you agree to our Terms of Service
              </p>
            </div>
          </div>

          
        </div>
      </div>
    </>
  );
}