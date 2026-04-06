/**
 * SignVocabulary.jsx
 * ------------------
 * Visual reference grid showing all supported ASL signs with letter/word badges.
 * Data comes from the Django /api/vocabulary/ endpoint.
 */

import { useState, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const SIGN_DESCRIPTIONS = {
  hello: "Wave hand", thanks: "Flat hand to chin", yes: "Fist nod",
  no: "Two fingers snap", please: "Rub chest", sorry: "Fist rub chest",
  iloveyou: "Pinky, index, thumb up", help: "Thumb on palm, rise",
  good: "Hand to chin, forward", bad: "Hand to chin, flip down",
  more: "Bunny fingers together", stop: "Flat palm chop",
  eat: "Fingers to mouth", drink: "Thumb to mouth", where: "Index wag",
};

export default function SignVocabulary() {
  const [vocab, setVocab] = useState({ static_signs: [], dynamic_signs: [] });
  const [activeTab, setActiveTab] = useState("static");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/api/vocabulary/`)
      .then((r) => r.json())
      .then((data) => {
        setVocab(data);
        setLoading(false);
      })
      .catch(() => {
        // Fallback data when backend is not running
        setVocab({
          static_signs: "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split(""),
          dynamic_signs: Object.keys(SIGN_DESCRIPTIONS),
        });
        setLoading(false);
      });
  }, []);

  const signs = activeTab === "static" ? vocab.static_signs : vocab.dynamic_signs;

  return (
    <div className="glass-card" style={{ padding: "24px" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h3 style={{ fontSize: "0.875rem", color: "var(--text-secondary)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Supported Signs
        </h3>
        <span className="badge badge-cyan">{signs.length} signs</span>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 6, marginBottom: 16, background: "rgba(0,0,0,0.3)", padding: 4, borderRadius: "var(--radius-md)" }}>
        {[
          { key: "static", label: "A–Z Alphabet" },
          { key: "dynamic", label: "Common Words" },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            style={{
              flex: 1,
              padding: "8px 12px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              background: activeTab === tab.key ? "rgba(99,220,219,0.15)" : "transparent",
              color: activeTab === tab.key ? "var(--accent-cyan)" : "var(--text-secondary)",
              fontFamily: "var(--font-main)",
              fontSize: "0.8rem",
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.2s ease",
              borderBottom: activeTab === tab.key ? "2px solid var(--accent-cyan)" : "2px solid transparent",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Signs grid */}
      {loading ? (
        <div style={{ textAlign: "center", padding: "20px", color: "var(--text-muted)" }}>Loading vocabulary…</div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: activeTab === "static"
              ? "repeat(auto-fill, minmax(52px, 1fr))"
              : "repeat(auto-fill, minmax(110px, 1fr))",
            gap: 8,
            maxHeight: 220,
            overflowY: "auto",
            paddingRight: 4,
          }}
        >
          {signs.map((sign) => (
            <div
              key={sign}
              title={SIGN_DESCRIPTIONS[sign.toLowerCase()] || sign}
              style={{
                padding: activeTab === "static" ? "10px 4px" : "8px 12px",
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.07)",
                borderRadius: "var(--radius-sm)",
                textAlign: "center",
                cursor: "default",
                transition: "all 0.2s ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "rgba(99,220,219,0.3)";
                e.currentTarget.style.background = "rgba(99,220,219,0.06)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.07)";
                e.currentTarget.style.background = "rgba(255,255,255,0.03)";
              }}
            >
              {activeTab === "static" ? (
                <span className="mono" style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
                  {sign}
                </span>
              ) : (
                <div>
                  <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-primary)", textTransform: "capitalize" }}>
                    {sign}
                  </span>
                  {SIGN_DESCRIPTIONS[sign.toLowerCase()] && (
                    <p style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: 3, lineHeight: 1.3 }}>
                      {SIGN_DESCRIPTIONS[sign.toLowerCase()]}
                    </p>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
