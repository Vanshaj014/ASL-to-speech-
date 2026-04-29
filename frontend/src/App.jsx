import { useState, useEffect, useCallback, useRef } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import VideoFeed from "./components/VideoFeed";
import PredictionDisplay from "./components/PredictionDisplay";
import SentenceBuilder from "./components/SentenceBuilder";
import SignVocabulary from "./components/SignVocabulary";
import HistoryLog from "./components/HistoryLog";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [prediction, setPrediction] = useState(null);
  const [mode, setMode] = useState("static");       // 'static' | 'dynamic'
  const [isCapturing, setIsCapturing] = useState(false);
  const [history, setHistory] = useState([]);
  const [currentSentence, setCurrentSentence] = useState("");
  const [bufferFill, setBufferFill] = useState(0);
  const [bufferTotal, setBufferTotal] = useState(30);
  const [modelStatus, setModelStatus] = useState({ static_model: false, dynamic_model: false });
  const [landmarks, setLandmarks] = useState(null);
  const modelStatusLoaded = useRef(false);

  // Fetch model status on mount
  useEffect(() => {
    if (modelStatusLoaded.current) return;
    modelStatusLoaded.current = true;
    fetch(`${API_URL}/api/status/`)
      .then((r) => r.json())
      .then(setModelStatus)
      .catch(() => {});
  }, []);

  // Handle incoming WebSocket messages
  const handleMessage = useCallback((data) => {
    if (data.type === "prediction") {
      setPrediction(data);
      setBufferFill(data.buffer_fill || 0);
      setBufferTotal(data.buffer_total || 30);
      setLandmarks(data.landmarks || null);
    } else if (data.type === "buffering") {
      setBufferFill(data.buffer_fill || 0);
      setBufferTotal(data.buffer_total || 30);
      setLandmarks(data.landmarks || null);
    }
  }, []);

  const { status: wsStatus, sendMessage, canSendFrame } = useWebSocket(handleMessage);

  // Switch mode and inform server
  const handleModeChange = (newMode) => {
    setMode(newMode);
    setPrediction(null);
    sendMessage({ type: "set_mode", mode: newMode });
  };

  // Start / Stop capturing
  const handleToggleCapture = () => {
    if (isCapturing && currentSentence) {
      // Save current sentence to history before stopping
      setHistory((prev) => [
        { text: currentSentence, time: Date.now() },
        ...prev,
      ].slice(0, 50));
    }
    setIsCapturing((v) => !v);
    setPrediction(null);
  };

  return (
    <>
      {/* Main layout */}
      <div className="app-layout">
        {/* ─── Header ─────────────────────────────────────── */}
        <header className="app-header">
          <div className="header-brand">
            <div className="brand-logo" aria-hidden="true">
              <svg viewBox="0 0 36 36" fill="none" width="26" height="26">
                <path d="M14 9v6M14 9c0-1.1.9-2 2-2s2 .9 2 2v6M18 9c0-1.1.9-2 2-2s2 .9 2 2v6M22 11c0-1.1.9-2 2-2s2 .9 2 2v7a8 8 0 01-8 8h-2a6 6 0 01-6-6v-4M10 15v4" stroke="var(--accent-cyan)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <div>
              <h1 className="brand-title">SignSpeak AI</h1>
              <p className="brand-sub">Real-Time ASL Translator</p>
            </div>
          </div>

          <div className="header-controls">
            {/* Mode switcher */}
            <div className="mode-switcher" role="group" aria-label="Translation mode">
              {[
                { key: "static", label: "A–Z Signs" },
                { key: "dynamic", label: "Word Signs" },
              ].map((m) => (
                <button
                  key={m.key}
                  id={`mode-${m.key}`}
                  className={`mode-btn ${mode === m.key ? "active" : ""}`}
                  onClick={() => handleModeChange(m.key)}
                  disabled={isCapturing}
                >
                  {m.label}
                </button>
              ))}
            </div>

            {/* WS Status */}
            <div className={`badge ${wsStatus === "connected" ? "badge-green" : wsStatus === "connecting" ? "badge-amber" : "badge-red"}`}>
              <span className="pulse-dot" />
              {wsStatus === "connected" ? "Backend Online" : wsStatus === "connecting" ? "Connecting…" : "Backend Offline"}
            </div>
          </div>
        </header>

        {/* ─── Main Grid ──────────────────────────────────── */}
        <main className="app-main">
          {/* Left column: video */}
          <section className="col-left">
            <div className="section-card">
              <div className="section-label">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="23 7 16 12 23 17 23 7"/>
                  <rect x="1" y="5" width="15" height="14" rx="2"/>
                </svg>
                Camera Feed
              </div>
              <VideoFeed
                sendMessage={sendMessage}
                canSendFrame={canSendFrame}
                wsStatus={wsStatus}
                isCapturing={isCapturing}
                landmarks={landmarks}
              />

              {/* Start/Stop capture button */}
              <button
                id="btn-capture-toggle"
                className={`btn ${isCapturing ? "btn-danger" : "btn-primary"}`}
                style={{ width: "100%", marginTop: 16, padding: "14px", fontSize: "1rem" }}
                onClick={handleToggleCapture}
                disabled={wsStatus !== "connected"}
              >
                {isCapturing ? (
                  <>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                      <rect x="6" y="6" width="12" height="12" rx="2"/>
                    </svg>
                    Stop Translating
                  </>
                ) : (
                  <>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                      <circle cx="12" cy="12" r="10" fill="transparent" stroke="currentColor" strokeWidth="2"/>
                      <circle cx="12" cy="12" r="4"/>
                    </svg>
                    Start Translating
                  </>
                )}
              </button>

              {/* Model status */}
              <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
                <span className={`badge ${modelStatus.static_model ? "badge-green" : "badge-red"}`}>
                  {modelStatus.static_model ? "✓" : "✗"} Static Model
                </span>
                <span className={`badge ${modelStatus.dynamic_model ? "badge-green" : "badge-amber"}`}>
                  {modelStatus.dynamic_model ? "✓" : "○"} Word Model
                </span>
              </div>
            </div>
          </section>

          {/* Middle column: prediction + sentence */}
          <section className="col-middle">
            <PredictionDisplay
              prediction={prediction}
              mode={mode}
              isCapturing={isCapturing}
              bufferFill={bufferFill}
              bufferTotal={bufferTotal}
            />
            <SentenceBuilder
              prediction={isCapturing ? prediction : null}
              onSentenceChange={setCurrentSentence}
            />
            <HistoryLog entries={history} />
          </section>

          {/* Right column: vocabulary */}
          <section className="col-right">
            <div className="section-card">
              <div className="section-label">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/>
                  <path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/>
                </svg>
                Sign Vocabulary
              </div>
              <SignVocabulary />
            </div>

            {/* Tips card */}
            <div className="glass-card" style={{ padding: "20px 22px" }}>
              <p className="text-xs" style={{ textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--text-muted)", marginBottom: 12, fontWeight: 600 }}>
                Tips for best accuracy
              </p>
              <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: 8 }}>
                {[
                  "Ensure good, even lighting on your hands",
                  "Keep hand centered in the camera frame",
                  "Sign slowly and hold each gesture briefly",
                  "Use A–Z mode for fingerspelling, Word mode for phrases",
                  "Plain background gives better detection",
                ].map((tip, i) => (
                  <li key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                    <span style={{ color: "var(--text-muted)", fontSize: "0.9rem", flexShrink: 0 }}>→</span>
                    <span className="text-xs text-muted" style={{ lineHeight: 1.5 }}>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          </section>
        </main>

        {/* ─── Footer ─────────────────────────────────────── */}
        <footer className="app-footer">
          <span className="text-xs text-muted">SignSpeak AI — Final Year Project © 2025</span>
          <span className="text-xs text-muted">MediaPipe · TensorFlow · Django Channels · React</span>
        </footer>
      </div>
    </>
  );
}
