"""Gray-code structured-light calibration: which projector pixel lands on which camera pixel.

Projects column and row Gray-code stripes (each with its inverse), photographs them, and
decodes a dense camera->projector map.  Writes calib/cam2proj.npz plus debug images.

    python calibrate.py [camera_index]
"""
import os
import sys

import cv2
import numpy as np

from camera import grab, open_camera
from projector_client import send

W, H = 1920, 1080
STEP = 4  # finest stripe width in projector px; 1px stripes just blur in the camera
ROOT = os.path.dirname(os.path.abspath(__file__))
PATTERNS = os.path.join(ROOT, "patterns", "graycode")
CALIB = os.path.join(ROOT, "calib")


def gray_bits(n_cells):
    return int(np.ceil(np.log2(n_cells)))


def make_patterns():
    """Return [(name, path)] in projection order, writing PNGs once."""
    os.makedirs(PATTERNS, exist_ok=True)
    items = []

    def add(name, image):
        path = os.path.join(PATTERNS, name + ".png")
        if not os.path.exists(path):
            cv2.imwrite(path, image)
        items.append((name, path))

    add("white", np.full((H, W), 255, np.uint8))
    add("black", np.zeros((H, W), np.uint8))
    for axis, length in (("x", W), ("y", H)):
        cells = np.arange(length) // STEP
        gray = cells ^ (cells >> 1)
        for bit in range(gray_bits(length // STEP) - 1, -1, -1):
            stripe = (((gray >> bit) & 1) * 255).astype(np.uint8)
            image = np.tile(stripe, (H, 1)) if axis == "x" else np.tile(stripe[:, None], (1, W))
            add(f"{axis}{bit:02d}p", image)
            add(f"{axis}{bit:02d}n", 255 - image)
    return items


def capture_all(camera_index):
    cap = open_camera(camera_index)
    shots = {}
    try:
        grab(cap, settle=1.5)  # let exposure settle before the run
        for name, path in make_patterns():
            send(f"show {path}")
            shots[name] = cv2.cvtColor(grab(cap, settle=0.45), cv2.COLOR_BGR2GRAY).astype(np.int16)
            print(f"  captured {name}", flush=True)
    finally:
        cap.release()
        send("black")
    return shots


def decode(shots, contrast=12):
    lit = shots["white"] - shots["black"]
    valid = lit > contrast * 2
    coords = {}
    for axis, length in (("x", W), ("y", H)):
        n = gray_bits(length // STEP)
        gray = np.zeros(lit.shape, np.int32)
        for bit in range(n - 1, -1, -1):
            diff = shots[f"{axis}{bit:02d}p"] - shots[f"{axis}{bit:02d}n"]
            # The finest bits are legitimately low-contrast at stripe edges; only gate on coarse ones.
            # The gate is relative to each pixel's own lit/dark range: in a dark room the camera's
            # noise reduction flattens fine stripes on dark ink far below any fixed threshold.
            if bit >= 2:
                valid &= np.abs(diff) > np.maximum(5, 0.10 * lit)
            gray |= (diff > 0).astype(np.int32) << bit
        binary = gray.copy()
        shift = gray >> 1
        while shift.any():
            binary ^= shift
            shift >>= 1
        coords[axis] = binary.astype(np.float32) * STEP + (STEP - 1) / 2
    valid &= (coords["x"] < W) & (coords["y"] < H)
    return coords["x"], coords["y"], valid, lit


def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    os.makedirs(CALIB, exist_ok=True)
    shots = capture_all(camera_index)
    px, py, valid, lit = decode(shots)
    np.savez_compressed(os.path.join(CALIB, "cam2proj.npz"), px=px, py=py, valid=valid)
    cv2.imwrite(os.path.join(CALIB, "white.png"), np.clip(shots["white"], 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(CALIB, "black.png"), np.clip(shots["black"], 0, 255).astype(np.uint8))

    debug = np.zeros(valid.shape + (3,), np.uint8)
    debug[..., 2] = np.where(valid, px / W * 255, 0)
    debug[..., 1] = np.where(valid, py / H * 255, 0)
    cv2.imwrite(os.path.join(CALIB, "decode_debug.png"), debug)
    print(f"decoded {int(valid.sum())} camera pixels ({valid.mean() * 100:.1f}% of frame)")


if __name__ == "__main__":
    main()
