"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import { api } from "@/lib/api";

/**
 * Live stream tuning + engine pause control.
 *
 * Two knobs:
 *  - target_fps  — how often the engine runs inference and pushes a
 *                  frame to the dashboard (1–30). Lower = less CPU,
 *                  rougher live feel.
 *  - jpeg_quality — encode quality of the frame sent over WebSocket
 *                   (30–95). Lower = smaller frame, less bandwidth,
 *                   more compression artefacts.
 *
 * Pause writes to /api/system/pause; while paused the engine keeps
 * streaming (preview alive) but skips inference + DB writes.
 */
const DEFAULT_FPS = 10;
const DEFAULT_QUALITY = 80;

export function CameraTuner() {
  const [targetFps, setTargetFps] = useState<number>(DEFAULT_FPS);
  const [jpegQuality, setJpegQuality] = useState<number>(DEFAULT_QUALITY);
  const [paused, setPaused] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Initial load — pull current values from the API.
  useEffect(() => {
    (async () => {
      try {
        const [c, p] = await Promise.all([api.cameraControls(), api.pauseState()]);
        setTargetFps(c.target_fps);
        setJpegQuality(c.jpeg_quality);
        setPaused(p.paused);
      } catch {
        // Engine probably not running — keep defaults silently.
      }
    })();
  }, []);

  const push = async (fps: number, q: number) => {
    setBusy(true);
    setError(null);
    try {
      await api.setCameraControls({ target_fps: fps, jpeg_quality: q });
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed");
    } finally {
      setBusy(false);
    }
  };

  const togglePause = async () => {
    setBusy(true);
    setError(null);
    try {
      const next = await api.setPause(!paused);
      setPaused(next.paused);
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed");
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    setTargetFps(DEFAULT_FPS);
    setJpegQuality(DEFAULT_QUALITY);
    push(DEFAULT_FPS, DEFAULT_QUALITY);
  };

  const onFps = (v: number) => {
    const intV = Math.round(v);
    setTargetFps(intV);
    push(intV, jpegQuality);
  };
  const onQuality = (v: number) => {
    const intV = Math.round(v);
    setJpegQuality(intV);
    push(targetFps, intV);
  };

  return (
    <div className="panel p-3">
      <div className="flex items-center justify-between border-b border-[var(--color-border)] pb-2 mb-3">
        <span className="label">STREAM TUNING</span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={togglePause}
            disabled={busy}
            className={clsx(
              "px-2 py-1 text-[11px] font-mono uppercase tracking-wider rounded border",
              paused
                ? "border-[var(--color-accent)] text-[var(--color-accent)]"
                : "border-[var(--color-fault)] text-[var(--color-fault)]"
            )}
          >
            {paused ? "▶ resume" : "⏸ pause"}
          </button>
          <button
            type="button"
            onClick={reset}
            disabled={busy}
            className="px-2 py-1 text-[11px] font-mono uppercase tracking-wider rounded border border-[var(--color-border)] text-[var(--color-text-dim)]"
          >
            reset
          </button>
        </div>
      </div>

      <Slider
        label="Target FPS"
        hint="Engine + dashboard cadence. Lower if the feed stutters or CPU runs hot."
        min={1}
        max={30}
        step={1}
        value={targetFps}
        onChange={onFps}
        format={(v) => `${v}`}
      />
      <Slider
        label="JPEG quality"
        hint="Frame compression for the live feed. Lower = smaller frames over the wire."
        min={30}
        max={95}
        step={1}
        value={jpegQuality}
        onChange={onQuality}
        format={(v) => `${v}`}
      />

      {error && (
        <p className="mt-2 text-[11px] font-mono text-[var(--color-fault)]">{error}</p>
      )}
      <p className="mt-2 text-[10px] font-mono text-[var(--color-text-mute)]">
        Changes apply live. Bake favourites into <code>config/system_config.yaml</code> (<code>camera.fps</code>, <code>camera.jpeg_quality</code>).
      </p>
    </div>
  );
}

function Slider({
  label,
  hint,
  min,
  max,
  step,
  value,
  onChange,
  format,
}: {
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
}) {
  return (
    <div className="mb-3">
      <div className="flex items-center justify-between text-[11px] font-mono">
        <span className="text-[var(--color-text-dim)] uppercase tracking-wider">{label}</span>
        <span className="text-[var(--color-accent)]">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full mt-1 accent-[var(--color-accent)]"
      />
      <p className="text-[10px] font-mono text-[var(--color-text-mute)] mt-0.5">{hint}</p>
    </div>
  );
}
