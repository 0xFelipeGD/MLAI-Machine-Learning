// TypeScript types matching api/schemas.py.

export type Verdict = "PASS" | "FAIL" | "WARN";

export interface HealthResponse {
  status: string;
  cpu_percent: number;
  ram_percent: number;
  temperature_c: number | null;
  uptime_seconds: number;
  fps: number;
  version: string;
}

export interface CameraConfig {
  width: number;
  height: number;
  fps: number;
  source: string;
  calibrated: boolean;
  px_per_mm: number | null;
}

export interface CameraControls {
  target_fps: number;
  jpeg_quality: number;
}

export interface PauseState {
  paused: boolean;
}

export interface AgroDetection {
  fruit_class: string;
  confidence: number;
  bbox_x1: number;
  bbox_y1: number;
  bbox_x2: number;
  bbox_y2: number;
  diameter_mm: number;
  quality?: string | null;
  quality_confidence?: number | null;
}

export interface AgroResult {
  id?: number;
  timestamp: string;
  total_detections: number;
  avg_diameter_mm: number;
  inference_ms: number;
  frame_path?: string | null;
  annotated_frame_path?: string | null;
  detections: AgroDetection[];
}

export interface AgroStatus {
  running: boolean;
  fruit_classes: string[];
  detection_threshold: number;
  last_result: AgroResult | null;
}

export interface AgroHistoryPage {
  items: AgroResult[];
  total: number;
  limit: number;
  offset: number;
}

export interface AgroStats {
  total_detections: number;
  by_class: Record<string, number>;
  by_quality: Record<string, number>;
  size_histogram: { range_mm: string; count: number }[];
}

export type WSMessage =
  | { type: "frame"; frame_b64: string; fps: number; timestamp: string }
  | {
      type: "agro_result";
      detections: AgroDetection[];
      total_count: number;
      inference_ms: number;
    };
