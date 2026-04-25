// REST API helpers — all endpoints proxied through Next.js rewrites in next.config.ts.

import type {
  AgroHistoryPage,
  AgroStats,
  AgroStatus,
  CameraConfig,
  CameraControls,
  HealthResponse,
  PauseState,
} from "./types";

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

export const api = {
  // System
  health: () => get<HealthResponse>("/api/system/health"),
  pauseState: () => get<PauseState>("/api/system/pause"),
  setPause: (paused: boolean) => post<PauseState>("/api/system/pause", { paused }),

  // Camera
  cameraConfig: () => get<CameraConfig>("/api/camera/config"),
  cameraControls: () => get<CameraControls>("/api/camera/controls"),
  setCameraControls: (body: CameraControls) =>
    post<CameraControls>("/api/camera/controls", body),

  // AGRO
  agro: {
    status: () => get<AgroStatus>("/api/agro/status"),
    history: (limit = 50, offset = 0) =>
      get<AgroHistoryPage>(`/api/agro/history?limit=${limit}&offset=${offset}`),
    stats: () => get<AgroStats>("/api/agro/stats"),
    config: (body: { detection_threshold?: number; fruit_classes?: string[] }) =>
      post<AgroStatus>("/api/agro/config", body),
  },
};
