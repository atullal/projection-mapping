"""Webcam capture helpers (OpenCV only -- never import pygame here).

OpenCV's bundled ffmpeg ships its own SDL2, which clashes with pygame's, so the
vision side and the projector side live in separate processes.
"""
import sys
import time

import cv2
import numpy as np


def open_camera(index=0, width=1920, height=1080):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"camera {index} did not open (check macOS camera permission)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def grab(cap, settle=0.35, average=3):
    """Return a fresh frame: flush buffered frames, wait out exposure, then average."""
    deadline = time.time() + settle
    while time.time() < deadline:
        cap.grab()
    frames = []
    for _ in range(average):
        ok, frame = cap.read()
        if ok:
            frames.append(frame.astype(np.float32))
    if not frames:
        raise RuntimeError("camera returned no frames")
    return np.mean(frames, axis=0).astype(np.uint8)


if __name__ == "__main__":
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out = sys.argv[2] if len(sys.argv) > 2 else f"cam{index}.jpg"
    cap = open_camera(index)
    frame = grab(cap, settle=1.5)
    cap.release()
    cv2.imwrite(out, frame)
    print(f"camera {index}: {frame.shape[1]}x{frame.shape[0]} mean={frame.mean():.1f} -> {out}")
