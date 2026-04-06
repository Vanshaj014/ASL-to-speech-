/**
 * SentenceBuilder.jsx
 * --------------------
 * Accumulates predicted signs into a sentence.
 * Provides clear, delete-last-word, copy, and TTS trigger buttons.
 */

import { useState, useEffect, useRef } from "react";

const SPEAK_URL = `${import.meta.env.VITE_API_URL || "http://localhost:8000"}/api/tts/`;
const MIN_CONFIDENCE_TO_ADD = 0.82;
const SAME_SIGN_COOLDOWN_MS = 1500; // Prevent rapid duplicate additions

export default function SentenceBuilder({ prediction, onSentenceChange }) {
  const [words, setWords] = useState([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [copied, setCopied] = useState(false);
  const lastAddedRef = useRef({ sign: null, time: 0 });

  // Auto-add confirmed predictions to sentence
  useEffect(() => {
    if (!prediction?.sign || prediction.confidence < MIN_CONFIDENCE_TO_ADD) return;

    const now = Date.now();
    const { sign, time } = lastAddedRef.current;

    // Ignore rapid repeats of the same sign
    if (sign === prediction.sign && now - time < SAME_SIGN_COOLDOWN_MS) return;

    lastAddedRef.current = { sign: prediction.sign, time: now };
    setWords((prev) => {
      const updated = [...prev, prediction.sign];
      return updated;
    });
  }, [prediction]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    onSentenceChange?.(words.join(" "));
  }, [words, onSentenceChange]);

  const sentence = words.join(" ");

  const handleClear = () => {
    setWords([]);
    lastAddedRef.current = { sign: null, time: 0 };
  };

  const handleDeleteLast = () => {
    setWords((prev) => prev.slice(0, -1));
  };

  const handleCopy = async () => {
    if (!sentence) return;
    await navigator.clipboard.writeText(sentence);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSpeak = async () => {
    if (!sentence || isSpeaking) return;
    setIsSpeaking(true);
    try {
      const resp = await fetch(SPEAK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sentence }),
      });
      const data = await resp.json();
      if (!data.success) console.warn("[TTS] Server TTS failed, using browser TTS");
    } catch {
      // Fallback: browser Web Speech API
      const utterance = new SpeechSynthesisUtterance(sentence);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      window.speechSynthesis.speak(utterance);
    } finally {
      setTimeout(() => setIsSpeaking(false), 2000);
    }
  };

  const handleAddSpace = () => {
    setWords((prev) => {
      const updated = [...prev, ""];
      return updated;
    });
  };

  return (
    <div className="glass-card" style={{ padding: "24px" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h3 style={{ fontSize: "0.875rem", color: "var(--text-secondary)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Translated Sentence
        </h3>
        <span className="badge badge-cyan">{words.length} word{words.length !== 1 ? "s" : ""}</span>
      </div>

      {/* Sentence display */}
      <div
        id="sentence-output"
        style={{
          minHeight: 80,
          padding: "16px",
          background: "rgba(0,0,0,0.3)",
          borderRadius: "var(--radius-md)",
          border: "1px solid var(--border)",
          fontSize: "1.4rem",
          fontWeight: 600,
          letterSpacing: "0.02em",
          color: sentence ? "var(--text-primary)" : "var(--text-muted)",
          fontStyle: sentence ? "normal" : "italic",
          lineHeight: 1.4,
          display: "flex",
          flexWrap: "wrap",
          gap: "6px",
          alignContent: "flex-start",
        }}
      >
        {sentence ? (
          words.filter(Boolean).map((word, i) => (
            <span
              key={`${word}-${i}`}
              className="animate-fade-in"
              style={{
                background: "rgba(99,220,219,0.08)",
                border: "1px solid rgba(99,220,219,0.2)",
                borderRadius: "6px",
                padding: "2px 10px",
                color: "var(--accent-cyan)",
              }}
            >
              {word}
            </span>
          ))
        ) : (
          "Your translation will appear here…"
        )}
      </div>

      {/* Action buttons */}
      <div style={{ display: "flex", gap: 8, marginTop: 16, flexWrap: "wrap" }}>
        <button
          id="btn-speak"
          className="btn btn-primary"
          onClick={handleSpeak}
          disabled={!sentence || isSpeaking}
        >
          {isSpeaking ? (
            <>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
              </svg>
              Speaking…
            </>
          ) : (
            <>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
              </svg>
              Speak
            </>
          )}
        </button>

        <button
          id="btn-copy"
          className={`btn btn-secondary`}
          onClick={handleCopy}
          disabled={!sentence}
        >
          {copied ? (
            <>✓ Copied!</>
          ) : (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="9" y="9" width="13" height="13" rx="2"/>
                <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
              </svg>
              Copy
            </>
          )}
        </button>

        <button id="btn-delete-last" className="btn btn-secondary" onClick={handleDeleteLast} disabled={!words.length}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 4H8l-7 8 7 8h13a2 2 0 002-2V6a2 2 0 00-2-2z"/>
            <line x1="18" y1="9" x2="12" y2="15"/>
            <line x1="12" y1="9" x2="18" y2="15"/>
          </svg>
          Undo
        </button>

        <button id="btn-add-space" className="btn btn-secondary" onClick={handleAddSpace} disabled={!words.length}>
          Space
        </button>

        <button id="btn-clear" className="btn btn-danger" onClick={handleClear} disabled={!words.length}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a1 1 0 011-1h4a1 1 0 011 1v2"/>
          </svg>
          Clear
        </button>
      </div>
    </div>
  );
}
