/**
 * VideoFeed.jsx
 * -------------
 * Captures webcam via browser getUserMedia API.
 * Samples frames at a configurable FPS and sends them to the Django backend
 * via the WebSocket sendMessage callback (base64-encoded JPEG).
 */

import { useEffect, useRef, useState, useCallback } from "react";
import "./VideoFeed.css";

const FRAME_INTERVAL_MS = 100;  // 10 FPS — balance between latency and CPU
const JPEG_QUALITY = 0.7;       // Reduce size for WS throughput

export default function VideoFeed({ sendMessage, canSendFrame, wsStatus, isCapturing, onHandDetected }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const frameTimerRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraError, setCameraError] = useState(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [fps, setFps] = useState(0);
  const fpsCounterRef = useRef({ count: 0, last: Date.now() });

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
          facingMode: "user",
        },
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsCameraOn(true);
        setCameraError(null);
      }
    } catch (err) {
      console.error("[Camera]", err);
      let msg = "Could not access camera.";
      if (err.name === "NotAllowedError") msg = "Camera permission denied. Please allow camera access.";
      else if (err.name === "NotFoundError") msg = "No camera found on this device.";
      setCameraError(msg);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCameraOn(false);
    clearInterval(frameTimerRef.current);
  }, []);

  // Start/stop frame capture loop when capturing state changes
  useEffect(() => {
    if (!isCapturing || !isCameraOn || wsStatus !== "connected") {
      clearInterval(frameTimerRef.current);
      return;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext("2d");

    frameTimerRef.current = setInterval(() => {
      if (!video || video.readyState < 2) return;
      if (canSendFrame && !canSendFrame()) return; // Backpressure: drop frame if backend hasn't responded

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      // Draw original un-mirrored frame to send to backend (matches training dataset)
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const frameB64 = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
      sendMessage({ type: "frame", frame: frameB64 });

      // FPS counter
      fpsCounterRef.current.count++;
      const now = Date.now();
      if (now - fpsCounterRef.current.last >= 1000) {
        setFps(fpsCounterRef.current.count);
        fpsCounterRef.current = { count: 0, last: now };
      }
    }, FRAME_INTERVAL_MS);

    return () => clearInterval(frameTimerRef.current);
  }, [isCapturing, isCameraOn, wsStatus, sendMessage]);

  // Auto-start camera on mount
  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, [startCamera, stopCamera]);

  return (
    <div className="video-feed-wrapper">
      {/* Status badges */}
      <div className="video-status-row">
        <span className={`badge ${wsStatus === "connected" ? "badge-green" : wsStatus === "connecting" ? "badge-amber" : "badge-red"}`}>
          <span className="pulse-dot" />
          {wsStatus === "connected" ? "Live" : wsStatus === "connecting" ? "Connecting…" : "Offline"}
        </span>
        {isCameraOn && isCapturing && (
          <span className="badge badge-cyan">
            <span className="pulse-dot" />
            {fps} FPS
          </span>
        )}
      </div>

      {/* Video Display */}
      <div className="video-container">
        {cameraError ? (
          <div className="camera-error">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M3 3l18 18M17 17H3a2 2 0 01-2-2V7a2 2 0 012-2h3m5.5 0H21a2 2 0 012 2v10c0 .55-.22 1.05-.58 1.42M8 8l8 8" />
            </svg>
            <p>{cameraError}</p>
            <button className="btn btn-secondary btn-sm" onClick={startCamera}>
              Retry
            </button>
          </div>
        ) : (
          <video
            ref={videoRef}
            id="webcam-feed"
            className="webcam-video"
            autoPlay
            playsInline
            muted
            style={{ transform: "scaleX(-1)" }}
          />
        )}
        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} style={{ display: "none" }} />

        {/* Overlay frame for active capture */}
        {isCameraOn && isCapturing && (
          <div className="capture-frame-overlay" aria-hidden="true">
            <div className="corner tl" />
            <div className="corner tr" />
            <div className="corner bl" />
            <div className="corner br" />
          </div>
        )}
      </div>

      {/* Camera toggle */}
      {!cameraError && (
        <div style={{ textAlign: "center", marginTop: "12px" }}>
          <button
            className={`btn btn-sm ${isCameraOn ? "btn-danger" : "btn-secondary"}`}
            onClick={isCameraOn ? stopCamera : startCamera}
          >
            {isCameraOn ? (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="6" width="12" height="12" rx="2" />
                </svg>
                Stop Camera
              </>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="23 7 16 12 23 17 23 7" />
                  <rect x="1" y="5" width="15" height="14" rx="2" />
                </svg>
                Start Camera
              </>
            )}
          </button>
        </div>
      )}

    </div>
  );
}
