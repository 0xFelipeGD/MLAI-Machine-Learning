// WebSocket connection manager for the live stream.

import type { WSMessage } from "./types";

export type WSHandler = (msg: WSMessage) => void;

export function connectLive(onMessage: WSHandler, onStatus?: (open: boolean) => void): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  let retry = 1000;

  const url = (() => {
    if (typeof window === "undefined") return "";
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const apiBase = process.env.NEXT_PUBLIC_API_BASE;
    if (apiBase) return apiBase.replace(/^http/, "ws") + "/ws/live";
    return `${proto}://${window.location.host}/ws/live`;
  })();

  const open = () => {
    if (closed) return;
    ws = new WebSocket(url);
    ws.onopen = () => {
      retry = 1000;
      onStatus?.(true);
    };
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data) as WSMessage;
        onMessage(msg);
      } catch {
        // ignore parse errors
      }
    };
    ws.onclose = () => {
      onStatus?.(false);
      if (!closed) {
        setTimeout(open, retry);
        retry = Math.min(retry * 1.5, 8000);
      }
    };
    ws.onerror = () => ws?.close();
  };

  open();

  return () => {
    closed = true;
    ws?.close();
  };
}

export function sendCommand(ws: WebSocket | null, action: "pause" | "resume" | "capture") {
  if (!ws || ws.readyState !== ws.OPEN) return;
  ws.send(JSON.stringify({ type: "command", action }));
}
