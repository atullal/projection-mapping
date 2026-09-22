"""Segment the Life & Love artwork (REF x REF) into inks and named regions.

Everything downstream animates these masks, so the geometry here is specific to this cover;
the ink classification and helpers are reusable.
"""
import os

import cv2
import numpy as np

from mapping import CALIB

INKS = {  # name: RGB of the printed ink in the reference file
    "black": (28, 24, 24), "teal": (40, 160, 150), "orange": (236, 128, 28),
    "gold": (244, 190, 72), "cream": (247, 226, 170),
}


def classify_inks(reference):
    """Per-pixel ink index (order of INKS) by nearest Lab colour, after flattening paper texture."""
    smooth = cv2.bilateralFilter(cv2.medianBlur(reference, 5), 9, 40, 9)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    swatches = np.uint8([[c[::-1] for c in INKS.values()]])
    palette = cv2.cvtColor(swatches, cv2.COLOR_BGR2LAB).astype(np.float32)[0]
    distance = np.linalg.norm(lab[:, :, None, :] - palette[None, None], axis=3)
    return distance.argmin(2).astype(np.uint8)


def disc(shape, centre, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= radius ** 2


def box(shape, x0, y0, x1, y1):
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    return mask


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def fit_circle(mask, rounds=5):
    """Least-squares circle through mask pixels, trimming strays (e.g. the teal bass inside the ring)."""
    ys, xs = np.nonzero(mask)
    points = np.stack([xs, ys], 1).astype(np.float64)
    keep = np.ones(len(points), bool)
    for _ in range(rounds):
        p = points[keep]
        A = np.column_stack([2 * p, np.ones(len(p))])
        (cx, cy, c), *_ = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)
        radius = np.sqrt(c + cx * cx + cy * cy)
        keep = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius) < 10
    return (float(cx), float(cy)), float(radius)


def largest_blob(mask, near=None):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if near is not None and labels[near[1], near[0]]:
        return labels == labels[near[1], near[0]]
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def build(reference):
    ink = classify_inks(reference)
    names = list(INKS)
    is_ink = {n: ink == i for i, n in enumerate(names)}
    light = is_ink["cream"] | is_ink["gold"]
    shape = ink.shape
    r = {"ink": ink}

    # title lettering: everything that is not teal field inside the title band
    band = box(shape, 0, 0, 1400, 318)
    r["life"] = band & box(shape, 20, 20, 600, 310) & ~is_ink["teal"]
    r["love"] = band & box(shape, 750, 20, 1385, 310) & ~is_ink["teal"]
    r["amp"] = band & box(shape, 585, 95, 765, 300) & is_ink["black"]

    # medallion: fit from the teal ring around the bassist
    ring_zone = disc(shape, (300, 1105), 285) & ~disc(shape, (300, 1105), 215) & is_ink["teal"]
    medal_c, medal_r = fit_circle(ring_zone)
    r["medal_centre"], r["medal_radius"] = medal_c, medal_r
    r["medal_ring"] = is_ink["teal"] & disc(shape, medal_c, medal_r + 16) & ~disc(shape, medal_c, medal_r - 16)
    r["medal_inside"] = disc(shape, medal_c, medal_r - 22)
    medal_all = disc(shape, medal_c, medal_r + 30)

    # sunburst: eye centre from the teal ring of the eye, rays are the light ink around it
    eye_teal = largest_blob(is_ink["teal"] & disc(shape, (185, 650), 120), near=None)
    sun_c = centroid(eye_teal)
    r["burst_centre"] = sun_c
    burst_zone = box(shape, 0, 335, 510, 905) & ~medal_all
    r["eye"] = disc(shape, sun_c, 128) & burst_zone
    r["rays"] = burst_zone & light & ~disc(shape, sun_c, 128)

    # instrument panel
    panel_zone = box(shape, 575, 338, 1400, 805)
    r["panel_zone"] = panel_zone
    r["panel_orange"] = panel_zone & is_ink["orange"]
    r["panel_stroke"] = panel_zone & light & ~box(shape, 640, 385, 1375, 765)
    r["panel_items"] = box(shape, 640, 385, 1375, 765) & ~is_ink["orange"] & ~is_ink["black"]
    r["keys"] = box(shape, 1105, 555, 1330, 610) & light
    r["speaker"] = disc(shape, (748, 690), 58)
    r["bolt"] = box(shape, 860, 515, 950, 630) & light
    r["bass"] = box(shape, 900, 385, 1370, 520) & ~is_ink["orange"]
    # the arcs themselves are black ink (unlightable), so the orange between them carries the motion
    r["waves_cymbal"] = box(shape, 755, 395, 870, 565) & is_ink["black"]
    r["waves_mic"] = box(shape, 935, 535, 1045, 600) & is_ink["black"]
    r["waves_cymbal_zone"] = box(shape, 750, 390, 875, 570) & is_ink["orange"]
    r["waves_mic_zone"] = box(shape, 930, 528, 1050, 605) & is_ink["orange"]
    r["notes"] = (box(shape, 700, 395, 770, 450) | box(shape, 1045, 470, 1125, 550)) & is_ink["black"]

    # sky, sun, mountains, wordmark
    low = box(shape, 0, 880, 1400, 1400) & ~medal_all
    sun_blob = largest_blob(is_ink["orange"] & box(shape, 1040, 930, 1230, 1100), near=(1132, 1015))
    r["sun"] = sun_blob
    r["sun_centre"] = centroid(sun_blob)
    r["sun_radius"] = float(np.sqrt(sun_blob.sum() / np.pi))
    r["wordmark"] = box(shape, 785, 1215, 1320, 1365) & light
    r["sky"] = low & light & ~r["wordmark"] & box(shape, 0, 880, 1400, 1260)
    r["clouds"] = r["sky"] & is_ink["cream"]
    mountains = low & is_ink["black"] & box(shape, 0, 1000, 1400, 1400)
    ridge = cv2.dilate(mountains.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & r["sky"]
    r["ridge"] = ridge

    r["teal"] = is_ink["teal"] & ~r["eye"] & ~medal_all & ~panel_zone | (is_ink["teal"] & panel_zone & ~box(shape, 640, 385, 1375, 765))
    r["black"] = is_ink["black"]
    return r


if __name__ == "__main__":
    reference = cv2.imread(os.path.join(CALIB, "reference.png"))
    regions = build(reference)
    colours = {"life": (60, 140, 255), "love": (80, 220, 255), "amp": (255, 0, 255), "rays": (120, 255, 255),
               "eye": (255, 255, 0), "panel_orange": (0, 90, 200), "panel_stroke": (255, 255, 255),
               "panel_items": (0, 255, 0), "keys": (255, 0, 0), "speaker": (200, 0, 200), "bolt": (0, 255, 255),
               "bass": (255, 160, 0), "waves_cymbal": (0, 0, 255), "waves_mic": (0, 0, 255), "notes": (180, 0, 90),
               "medal_ring": (255, 255, 0), "sun": (0, 0, 255), "sky": (200, 200, 120), "clouds": (255, 255, 255),
               "ridge": (0, 255, 0), "wordmark": (255, 120, 255), "teal": (140, 110, 0)}
    debug = (reference * 0.25).astype(np.uint8)
    for name, colour in colours.items():
        debug[regions[name]] = colour
    for key in ("burst_centre", "medal_centre", "sun_centre"):
        cv2.drawMarker(debug, tuple(int(v) for v in regions[key]), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    print({k: np.round(regions[k], 1).tolist() for k in ("burst_centre", "medal_centre", "medal_radius", "sun_centre", "sun_radius")})
    cv2.imwrite(os.path.join(CALIB, "regions_debug.jpg"), cv2.resize(debug, (1000, 1000), interpolation=cv2.INTER_AREA))
