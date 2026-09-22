"""Shared helpers for authoring looped animations in artwork space (cv2/numpy only)."""
import os
from multiprocessing import Pool

import cv2
import numpy as np

from mapping import CALIB
from render import ProjectorWarp

TAU = 2 * np.pi


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def lobe(phase, sharpness=1.0):
    """0..1 periodic bump at phase = 0 (phase in turns)."""
    return (0.5 + 0.5 * np.cos(TAU * phase)) ** sharpness


def wrapped(u):
    """Signed distance to the nearest integer, for glints that wrap around a loop."""
    return u - np.round(u)


def rgb(r, g, b):
    return np.array([b, g, r], np.float32)  # frames are BGR


def cycle(colours, phase):
    """Smoothly loop through a list of colours; phase in turns (array or scalar)."""
    phase = np.asarray(phase, np.float32) % 1.0 * len(colours)
    i = np.floor(phase).astype(int) % len(colours)
    f = smoothstep(phase - np.floor(phase))[..., None]
    table = np.stack(colours)
    return table[i] * (1 - f) + table[(i + 1) % len(colours)] * f


def soft(mask, sigma=1.2):
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)


def classify_inks(reference, palette_rgb):
    """Per-pixel index into palette_rgb by nearest Lab colour, after flattening print texture."""
    smooth = cv2.bilateralFilter(cv2.medianBlur(reference, 5), 9, 40, 9)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    swatches = np.uint8([[c[::-1] for c in palette_rgb]])
    palette = cv2.cvtColor(swatches, cv2.COLOR_BGR2LAB).astype(np.float32)[0]
    return np.linalg.norm(lab[:, :, None, :] - palette[None, None], axis=3).argmin(2).astype(np.uint8)


def geodesic(mask, seeds, bridge=3):
    """Steps needed to reach each mask pixel from the seeds while staying on the mask.

    This is distance *along the ink*: animating it sends light down each connected line
    instead of sweeping across them.  Unreached pixels are np.inf.
    """
    walkable = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((bridge, bridge), np.uint8)) > 0
    reached = seeds & walkable
    distance = np.full(mask.shape, np.inf, np.float32)
    distance[reached] = 0
    kernel = np.ones((3, 3), np.uint8)
    step = 0
    while True:
        step += 1
        grown = (cv2.dilate(reached.astype(np.uint8), kernel) > 0) & walkable & ~reached
        if not grown.any():
            break
        distance[grown] = step
        reached |= grown
    distance[~mask] = np.inf
    return distance


def beam_alpha(feather=38):
    """Mask to the sleeve, tight at its own edges, long fade where the projector beam runs out."""
    sleeve = cv2.imread(os.path.join(CALIB, "sleeve.png"), 0) > 0
    covered = cv2.imread(os.path.join(CALIB, "covered.png"), 0) > 0
    alpha = soft(cv2.erode(sleeve.astype(np.uint8), np.ones((5, 5), np.uint8)), 2.0)
    # pad first: the artwork's own border is not a beam limit and must not be feathered
    padded = cv2.copyMakeBorder(covered.astype(np.uint8), 60, 60, 60, 60, cv2.BORDER_REPLICATE)
    return alpha * smoothstep((cv2.distanceTransform(padded, cv2.DIST_L2, 5)[60:-60, 60:-60] - 2) / feather)


def edge_lines(reference, low=60, high=140):
    grey = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    return soft(cv2.dilate(cv2.Canny(grey, low, high), np.ones((3, 3), np.uint8)) > 0, 1.0)


_frame_light, _warp, _out = None, None, None


def _init(setup, frame_light, out):
    global _frame_light, _warp, _out
    setup()
    _frame_light, _warp, _out = frame_light, ProjectorWarp(), out


def _render(job):
    index, t = job
    image = (np.clip(_frame_light(t), 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(_out, f"f{index:04d}.jpg"), _warp(image), [cv2.IMWRITE_JPEG_QUALITY, 93])
    return index


def render_loop(setup, frame_light, out, seconds, fps=30, workers=8, frames_per_beat=None):
    os.makedirs(out, exist_ok=True)
    for stale in os.listdir(out):
        os.remove(os.path.join(out, stale))
    frames = int(seconds * fps)
    with Pool(workers, initializer=_init, initargs=(setup, frame_light, out)) as pool:
        for done in pool.imap_unordered(_render, [(i, i / fps) for i in range(frames)], chunksize=4):
            if done % 120 == 0:
                print(f"  frame {done}/{frames}", flush=True)
    ProjectorWarp().write_meta(out, fps, frames_per_beat)
    print(f"rendered {frames} frames -> {out}")


def contact_sheet(setup, frame_light, times, path, tile=466, columns=3):
    setup()
    tiles = []
    for t in times:
        image = (np.clip(frame_light(t), 0, 1) * 255).astype(np.uint8)
        image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(image, f"{t:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(image)
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]))
