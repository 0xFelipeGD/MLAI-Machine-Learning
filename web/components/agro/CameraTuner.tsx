"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

/**
 * Live camera tuning + engine pause control.
 *
 * Sliders write to /api/camera/controls (POST) which calls
 * picamera2.set_controls() on the running camera — no service restart.
 * Pause button writes to /api/system/pause; while paused the engine
 * still streams the live feed but skips inference + DB writes.
 */
export function CameraTuner() {
  const [redGain, setRedGain] = useState<number>(2.0);
  const [blueGain, setBlueGain] = useState<number>(1.0);
  const [paused, setPaused] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Initial load — pull current values from the API.
  useEffect(() => {
    (async () => {
      try {
        const [c, p] = await Promise.all([api.cameraControls(), api.pauseState()]);
        setRedGain(c.red_gain);
        setBlueGain(c.blue_gain);
        setPaused(p.paused);
      } catch {
        // Engine probably not running — keep defaults silently.
      }
    })();
  }, []);

  // Push gains to the API. Fire-and-forget; no debounce because the user
  // wants instant feedback while dragging.
  const pushGains = async (red: number, blue: number) => {
    setBusy(true);
    setError(null);
    try {
      await api.setCameraControls({
        red_gain: red,
        blue_gain: blue,
        color_matrix: null,
      });
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
    setRedGain(2.0);
    setBlueGain(1.0);
    pushGains(2.0, 1.0);
  };

  return (
    <div className="panel p-3">
      <div className="flex items-center justify-between border-b border-[var(--color-border)] pb-2 mb-3">
        <span className="label">CAMERA TUNING</span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={togglePause}
            disabled={busy}
            className={
              "px-2 py-1 text-[11px] font-mono uppercase tracking-wider rounded border " +
              (paused
                ? "border-[var(--color-accent)] text-[var(--color-accent)]"
                : "border-[var(--color-fault)] text-[var(--color-fault)]")
            }
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
        label="Red gain"
        hint="Boost reds and oranges. NoIR cameras need a high value (2.5–4.0)."
        min={0.2}
        max={5.0}
        step={0.05}
        value={redGain}
        onChange={(v) => {
          setRedGain(v);
          pushGains(v, blueGain);
        }}
      />
      <Slider
        label="Blue gain"
        hint="Lower this if the image looks too blue/cyan (try 0.3–0.7 for NoIR)."
        min={0.1}
        max={3.0}
        step={0.05}
        value={blueGain}
        onChange={(v) => {
          setBlueGain(v);
          pushGains(redGain, v);
        }}
      />

      {error && (
        <p className="mt-2 text-[11px] font-mono text-[var(--color-fault)]">{error}</p>
      )}
      <p className="mt-2 text-[10px] font-mono text-[var(--color-text-mute)]">
        Changes apply live to the running camera. Bake favourites into{" "}
        <code>config/system_config.yaml</code>.
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
}: {
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mb-3">
      <div className="flex items-center justify-between text-[11px] font-mono">
        <span className="text-[var(--color-text-dim)] uppercase tracking-wider">{label}</span>
        <span className="text-[var(--color-accent)]">{value.toFixed(2)}</span>
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
