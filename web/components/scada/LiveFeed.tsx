"use client";

import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { connectLive } from "@/lib/ws";
import type { WSMessage } from "@/lib/types";

interface Props {
  onResult?: (msg: WSMessage) => void;
}

export function LiveFeed({ onResult }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [connected, setConnected] = useState(false);
  const [fps, setFps] = useState(0);
  const [stamp, setStamp] = useState<string>("");

  useEffect(() => {
    const dispose = connectLive(
      (msg) => {
        if (msg.type === "frame") {
          if (imgRef.current) imgRef.current.src = `data:image/jpeg;base64,${msg.frame_b64}`;
          setFps(msg.fps);
          setStamp(msg.timestamp.replace("T", " ").slice(0, 19) + "Z");
        }
        onResult?.(msg);
      },
      (open) => setConnected(open)
    );
    return dispose;
  }, [onResult]);

  return (
    <div className="panel relative overflow-hidden">
      <div className="flex items-center justify-between border-b border-[var(--color-border)] px-3 py-2">
        <div className="flex items-center gap-2">
          <span
            className={clsx(
              "dot dot-pulse",
              connected ? "text-[var(--color-ok)]" : "text-[var(--color-fault)]"
            )}
          />
          <span className="label">{connected ? "LIVE" : "OFFLINE"}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="value text-[11px] text-[var(--color-text-dim)]">{stamp}</span>
          <span className="value text-[11px] text-[var(--color-accent)]">{fps.toFixed(1)} FPS</span>
        </div>
      </div>
      <div className="aspect-video bg-black scanline relative">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imgRef}
          alt="live feed"
          className="absolute inset-0 h-full w-full object-contain"
        />
        {!connected && (
          <div className="absolute inset-0 flex items-center justify-center text-[var(--color-text-mute)] font-mono text-xs uppercase tracking-widest">
            ⎯ awaiting signal ⎯
          </div>
        )}
      </div>
    </div>
  );
}
