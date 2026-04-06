/**
 * useWebSocket.js
 * ----------------
 * Custom React hook that manages a WebSocket connection to the Django
 * Channels backend. Handles connect/disconnect, message serialization,
 * automatic heartbeat, and reconnection logic.
 */

import { useEffect, useRef, useCallback, useState } from "react";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/translate/";
const HEARTBEAT_INTERVAL_MS = 15000;
const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

export function useWebSocket(onMessage) {
  const wsRef = useRef(null);
  const heartbeatRef = useRef(null);
  const reconnectRef = useRef(null);
  const reconnectCount = useRef(0);
  const isMounted = useRef(true);

  const [status, setStatus] = useState("disconnected"); // connected | disconnected | connecting | error

  const clearTimers = () => {
    clearInterval(heartbeatRef.current);
    clearTimeout(reconnectRef.current);
  };

  const connect = useCallback(() => {
    if (!isMounted.current) return;
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    setStatus("connecting");

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!isMounted.current) { ws.close(); return; }
      setStatus("connected");
      reconnectCount.current = 0;

      // Heartbeat ping
      clearInterval(heartbeatRef.current);
      heartbeatRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, HEARTBEAT_INTERVAL_MS);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "pong") return; // Ignore heartbeat responses
        if (onMessage) onMessage(data);
      } catch (e) {
        console.error("[WS] Failed to parse message:", e);
      }
    };

    ws.onerror = (err) => {
      console.error("[WS] Error:", err);
      setStatus("error");
    };

    ws.onclose = (event) => {
      clearTimers();
      if (!isMounted.current) return;
      setStatus("disconnected");
      console.warn(`[WS] Closed (code=${event.code}). Reconnecting...`);

      if (reconnectCount.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectCount.current++;
        reconnectRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
      } else {
        setStatus("error");
        console.error("[WS] Max reconnection attempts reached.");
      }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const disconnect = useCallback(() => {
    clearTimers();
    if (wsRef.current) {
      wsRef.current.onclose = null; // Prevent auto-reconnect on manual close
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus("disconnected");
  }, []);

  const sendMessage = useCallback((data) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
      return true;
    }
    return false;
  }, []);

  useEffect(() => {
    isMounted.current = true;
    connect();
    return () => {
      isMounted.current = false;
      clearTimers();
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { status, sendMessage, connect, disconnect };
}
