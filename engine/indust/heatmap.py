"""
engine/indust/heatmap.py — Heatmap colorization, overlay, and base64 encoding.
"""

from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np


def colorize_heatmap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a [0,1] float heatmap to a BGR uint8 colored image."""
    h = np.clip(heatmap, 0.0, 1.0)
    h_u8 = (h * 255.0).astype(np.uint8)
    return cv2.applyColorMap(h_u8, colormap)


def overlay_heatmap(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a heatmap on top of the original frame."""
    H, W = frame_bgr.shape[:2]
    h_resized = cv2.resize(heatmap, (W, H))
    colored = colorize_heatmap(h_resized, colormap)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, colored, alpha, 0)


def encode_b64(image_bgr: np.ndarray, quality: int = 80, ext: str = ".jpg") -> str:
    """Encode an image to a base64 string suitable for JSON/WebSocket."""
    ok, buf = cv2.imencode(ext, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
