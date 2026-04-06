/**
 * HistoryLog.jsx
 * ---------------
 * Shows the last N translated sentences from the current session.
 */

import { useState } from "react";

export default function HistoryLog({ entries }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const visible = isExpanded ? entries : entries.slice(0, 5);

  if (!entries.length) {
    return (
      <div className="glass-card" style={{ padding: "20px 24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--text-muted)" }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M12 20h9M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/>
          </svg>
          <span className="text-sm">Translation history will appear here</span>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card" style={{ padding: "24px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
        <h3 style={{ fontSize: "0.875rem", color: "var(--text-secondary)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em" }}>
          History
        </h3>
        <span className="badge badge-cyan">{entries.length}</span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {visible.map((entry, i) => (
          <div
            key={i}
            className="animate-fade-in"
            style={{
              display: "flex",
              gap: 12,
              padding: "10px 14px",
              background: "rgba(0,0,0,0.25)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-sm)",
              alignItems: "flex-start",
            }}
          >
            <span style={{
              fontSize: "0.7rem",
              color: "var(--text-muted)",
              fontFamily: "var(--font-mono)",
              minWidth: 26,
              paddingTop: 2
            }}>
              #{entries.length - i}
            </span>
            <span style={{ fontSize: "0.9rem", color: "var(--text-primary)", flex: 1 }}>
              {entry.text}
            </span>
            <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", whiteSpace: "nowrap" }}>
              {new Date(entry.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
          </div>
        ))}
      </div>

      {entries.length > 5 && (
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => setIsExpanded(!isExpanded)}
          style={{ marginTop: 12, width: "100%" }}
        >
          {isExpanded ? "Show less" : `Show all ${entries.length} entries`}
        </button>
      )}
    </div>
  );
}
