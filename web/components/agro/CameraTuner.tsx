"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import { api } from "@/lib/api";

/**
 * Live camera tuning + engine pause control.
 *
 * AUTO mode lets the camera's AWB algorithm run (preferred when a proper
 * tuning_file is loaded — e.g. imx708.json on a NoIR sensor).
 * MANUAL mode disables AWB and uses the slider gains.
 *
 * Pause writes to /api/system/pause; while paused the engine keeps
 * streaming so you can preview colour tweaks but skips inference + DB.
 */
export function CameraTuner() {
  const [redGain, setRedGain] = useState<number>(2.0);
  const [blueGain, setBlueGain] = useState<number>(1.0);
  const [awbAuto, setAwbAuto] = useState<boolean>(true);
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
        setAwbAuto(c.awb_auto);
        setPaused(p.paused);
      } catch {
        // Engine probably not running — keep defaults silently.
      }
    })();
  }, []);

  const push = async (red: number, blue: number, auto: boolean) => {
    setBusy(true);
    setError(null);
    try {
      await api.setCameraControls({
        red_gain: red,
        blue_gain: blue,
        awb_auto: auto,
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
    setAwbAuto(true);
    push(2.0, 1.0, true);
  };

  // Switching to MANUAL pushes current sliders so the camera is held
  // at exactly what the user sees on screen at that moment.
  const setMode = (auto: boolean) => {
    setAwbAuto(auto);
    push(redGain, blueGain, auto);
  };

  // Slider drags push gains AND switch to MANUAL automatically — there is
  // no point dragging if AWB is going to override the values.
  const onRedGain = (v: number) => {
    setRedGain(v);
    setAwbAuto(false);
    push(v, blueGain, false);
  };
  const onBlueGain = (v: number) => {
    setBlueGain(v);
    setAwbAuto(false);
    push(redGain, v, false);
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

      {/* AUTO / MANUAL toggle */}
      <div className="mb-3">
        <div className="text-[11px] font-mono uppercase tracking-wider text-[var(--color-text-dim)] mb-1">
          White balance
        </div>
        <div className="flex gap-1">
          <button
            type="button"
            onClick={() => setMode(true)}
            disabled={busy}
            className={clsx(
              "flex-1 px-2 py-1 text-[11px] font-mono uppercase tracking-wider rounded border",
              awbAuto
                ? "border-[var(--color-accent)] text-[var(--color-accent)] bg-[var(--color-accent)]/10"
                : "border-[var(--color-border)] text-[var(--color-text-dim)]"
            )}
          >
            auto
          </button>
          <button
            type="button"
            onClick={() => setMode(false)}
            disabled={busy}
            className={clsx(
              "flex-1 px-2 py-1 text-[11px] font-mono uppercase tracking-wider rounded border",
              !awbAuto
                ? "border-[var(--color-accent)] text-[var(--color-accent)] bg-[var(--color-accent)]/10"
                : "border-[var(--color-border)] text-[var(--color-text-dim)]"
            )}
          >
            manual
          </button>
        </div>
        <p className="text-[10px] font-mono text-[var(--color-text-mute)] mt-1">
          AUTO uses libcamera tuning. MANUAL applies the sliders below.
        </p>
      </div>

      <div className={clsx("transition-opacity", awbAuto && "opacity-50")}>
        <Slider
          label="Red gain"
          hint="Boost reds and oranges. NoIR cameras typically want 2.0–3.5."
          min={0.2}
          max={5.0}
          step={0.05}
          value={redGain}
          onChange={onRedGain}
        />
        <Slider
          label="Blue gain"
          hint="Lower this if the image looks too blue/cyan."
          min={0.1}
          max={3.0}
          step={0.05}
          value={blueGain}
          onChange={onBlueGain}
        />
      </div>

      {error && (
        <p className="mt-2 text-[11px] font-mono text-[var(--color-fault)]">{error}</p>
      )}
      <p className="mt-2 text-[10px] font-mono text-[var(--color-text-mute)]">
        Changes apply live. Bake favourites into <code>config/system_config.yaml</code>.
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
