/**
 * PredictionDisplay.jsx
 * ----------------------
 * Shows the current predicted ASL sign with animated letter, 
 * confidence bar, and top-3 alternatives.
 */

import { useEffect, useRef, useState } from "react";

function ConfidenceBar({ value, color = "var(--accent-cyan)" }) {
  const pct = Math.round((value || 0) * 100);
  const barColor =
    pct >= 85 ? "var(--accent-green)"
    : pct >= 60 ? "var(--accent-cyan)"
    : pct >= 40 ? "var(--accent-amber)"
    : "var(--accent-red)";

  return (
    <div style={{ width: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
        <span className="text-xs text-muted">Confidence</span>
        <span className="text-xs mono" style={{ color: barColor, fontWeight: 600 }}>{pct}%</span>
      </div>
      <div className="confidence-track">
        <div
          className="confidence-fill"
          style={{ width: `${pct}%`, background: barColor }}
        />
      </div>
    </div>
  );
}

export default function PredictionDisplay({ prediction, mode, isCapturing, bufferFill, bufferTotal }) {
  const [displaySign, setDisplaySign] = useState(null);
  const [animKey, setAnimKey] = useState(0);
  const prevSignRef = useRef(null);

  useEffect(() => {
    if (prediction?.sign && prediction.sign !== prevSignRef.current) {
      setDisplaySign(prediction);
      setAnimKey((k) => k + 1);
      prevSignRef.current = prediction.sign;
    } else if (!prediction?.sign && prediction?.no_hand) {
      setDisplaySign(null);
    }
  }, [prediction]);

  const pct = Math.round((prediction?.confidence || 0) * 100);
  const bufferPct = bufferTotal ? Math.round((bufferFill / bufferTotal) * 100) : 0;

  return (
    <div className="glass-card" style={{ padding: "28px 24px", textAlign: "center" }}>
      {/* Mode indicator */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
        <span className="text-xs text-muted" style={{ textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Current Sign
        </span>
        <span className={`badge ${mode === "static" ? "badge-cyan" : "badge-green"}`}>
          {mode === "static" ? "A–Z Mode" : "Word Mode"}
        </span>
      </div>

      {/* Main sign display */}
      <div className="sign-display-area" style={{ minHeight: 130, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
        {!isCapturing ? (
          <p className="text-muted" style={{ fontSize: "0.95rem" }}>
            Press <strong style={{ color: "var(--accent-cyan)" }}>Start Translating</strong> to begin
          </p>
        ) : displaySign?.sign ? (
          <>
            <div
              key={animKey}
              className="sign-display animate-pop-in"
            >
              {displaySign.sign.toUpperCase()}
            </div>
            <ConfidenceBar value={displaySign.confidence} />
          </>
        ) : prediction?.no_hand ? (
          <div style={{ textAlign: "center" }}>
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" style={{ marginBottom: 8 }}>
              <path d="M18 11V6a2 2 0 00-2-2v0a2 2 0 00-2 2v0" />
              <path d="M14 10V4a2 2 0 00-2-2v0a2 2 0 00-2 2v2" />
              <path d="M10 10.5V6a2 2 0 00-2-2v0a2 2 0 00-2 2v8" />
              <path d="M18 11a2 2 0 114 0v3a8 8 0 01-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 012.83-2.82L7 15" />
            </svg>
            <p className="text-muted text-sm">No hand detected</p>
          </div>
        ) : mode === "dynamic" && bufferFill < bufferTotal ? (
          <div style={{ width: "100%" }}>
            <p className="text-sm text-muted" style={{ marginBottom: 10 }}>Capturing sequence…</p>
            <div className="confidence-track">
              <div
                className="confidence-fill"
                style={{
                  width: `${bufferPct}%`,
                  background: "var(--accent-primary)"
                }}
              />
            </div>
            <p className="text-xs text-muted" style={{ marginTop: 6 }}>{bufferFill}/{bufferTotal} frames</p>
          </div>
        ) : (
          <div style={{ display: "flex", gap: 6 }}>
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                style={{
                  width: 8, height: 8, borderRadius: "50%",
                  background: "var(--accent-cyan)",
                  animation: `pulseDot 1.2s ease-in-out ${i * 0.2}s infinite`
                }}
              />
            ))}
          </div>
        )}
      </div>

      {/* Top 3 alternatives */}
      {displaySign?.top3?.length > 1 && (
        <div style={{ marginTop: 20, paddingTop: 16, borderTop: "1px solid var(--border)" }}>
          <p className="text-xs text-muted" style={{ marginBottom: 10, textAlign: "left" }}>Top alternatives</p>
          <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
            {displaySign.top3.slice(1).map((item, i) => (
              <div
                key={i}
                className="glass-card"
                style={{
                  padding: "6px 14px",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  borderRadius: "var(--radius-sm)",
                }}
              >
                <span className="mono" style={{ fontWeight: 700, fontSize: "1rem" }}>
                  {item.sign.toUpperCase()}
                </span>
                <span className="text-xs text-muted">{Math.round(item.confidence * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
