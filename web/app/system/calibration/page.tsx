"use client";

import { useEffect, useState } from "react";
import { PageHeader } from "@/components/scada/PageHeader";
import { api } from "@/lib/api";
import type { CameraConfig } from "@/lib/types";

export default function CalibrationPage() {
  const [camera, setCamera] = useState<CameraConfig | null>(null);

  useEffect(() => {
    api.cameraConfig().then(setCamera).catch(() => {});
  }, []);

  return (
    <div>
      <PageHeader
        tag="SYSTEM"
        title="Camera Calibration"
        subtitle="Use a checkerboard to teach the camera its focal length and lens distortion"
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="panel p-4 col-span-1">
          <div className="label mb-3">Status</div>
          <div className="value text-[28px]">{camera?.calibrated ? "Calibrated" : "Not calibrated"}</div>
          {camera?.calibrated && (
            <div className="mt-2 text-[12px] text-[var(--color-text-dim)]">
              {camera.px_per_mm?.toFixed(3)} px/mm
            </div>
          )}
        </div>

        <div className="panel p-4 col-span-2">
          <div className="label mb-3">How to Calibrate</div>
          <ol className="list-decimal pl-5 space-y-2 text-[12px] text-[var(--color-text-dim)]">
            <li>Print a 10×7 checkerboard with 25 mm squares.</li>
            <li>SSH into the Pi and run:</li>
            <li>
              <code className="value text-[11px] bg-[var(--color-bg)] px-2 py-1 inline-block">
                python scripts/calibrate_camera.py
              </code>
            </li>
            <li>Hold the board in front of the camera at different angles.</li>
            <li>Press SPACE to capture each pose. ENTER when 15+ captures done.</li>
            <li>The new calibration is written to <code className="value">config/camera_calibration.json</code>.</li>
            <li>Restart <code className="value">mlai-engine</code> to pick it up.</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
