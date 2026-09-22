"""Black Market Brass - Hox: seamless 16 s loop authored in artwork space.

    python animate_hox.py [--sheet | --regions]

A two-ink print (mint green + violet on black) lets light colour select a layer: green light
blazes the green ink and blacks out the violet, violet/magenta light does the reverse.  The
orb is treated as the power source -- pulses leave it *along the linework* (geodesic distance
through the ink), and once per loop a shockwave splits the print into its two layers.
"""
import os
import sys

import cv2
import numpy as np

from animkit import (TAU, beam_alpha, classify_inks, contact_sheet, cycle, edge_lines, geodesic, lobe,
                     render_loop, rgb, smoothstep, soft, wrapped)
from mapping import CALIB, ROOT

LOOP, FPS, BEATS = 16.0, 30, 28  # 105 BPM
OUT = os.path.join(ROOT, "frames", "hox")
FIELDS = os.path.join(CALIB, "hox_fields.npz")
PALETTE = {"black": (8, 8, 8), "green": (115, 170, 87), "violet": (102, 76, 147), "red": (225, 30, 40)}
ORB, ORB_R = (677, 695), 90
EYES = ((1063, 300), (1180, 297))
DOOR, BEAM_END = (1130, 690), (230, 1250)

GREENS = [rgb(0.15, 1.0, 0.45), rgb(0.05, 1.0, 0.90), rgb(0.60, 1.0, 0.15)]
VIOLETS = [rgb(0.55, 0.25, 1.0), rgb(1.0, 0.15, 0.85), rgb(0.25, 0.40, 1.0)]
LINES = [rgb(0.3, 1.0, 0.6), rgb(0.8, 0.4, 1.0)]

S = None


def box(shape, x0, y0, x1, y1):
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    return mask


def build_fields():
    """Slow, one-off: ink masks and along-the-ink distances.  Cached for the render workers."""
    reference = cv2.imread(os.path.join(CALIB, "reference.png"))
    ink = classify_inks(reference, list(PALETTE.values()))
    names = list(PALETTE)
    green, violet = ink == names.index("green"), ink == names.index("violet")
    n = ink.shape[0]
    yy, xx = np.mgrid[:n, :n]
    orb_r = np.hypot(xx - ORB[0], yy - ORB[1]).astype(np.float32)
    chip = box(ink.shape, 205, 35, 450, 165)
    geo = {}
    for name, mask, seeds in (("green", green, (orb_r < ORB_R + 12) | chip),
                              ("violet", violet, (orb_r < ORB_R + 70) | chip)):
        d = geodesic(mask, seeds & mask)
        reached = np.isfinite(d)
        print(f"  {name}: {reached.sum() / mask.sum() * 100:.0f}% of the ink is connected to the source,"
              f" longest path {d[reached].max():.0f}px")
        # islands the flood never reaches still need a phase: fall back to straight-line distance
        d[~reached] = orb_r[~reached] * 1.25
        geo[name] = d
    np.savez_compressed(FIELDS, ink=ink, geo_green=geo["green"], geo_violet=geo["violet"])


def setup():
    global S
    reference = cv2.imread(os.path.join(CALIB, "reference.png"))
    data = np.load(FIELDS)
    ink, names = data["ink"], list(PALETTE)
    is_ink = {k: ink == i for i, k in enumerate(names)}
    shape = ink.shape
    n = shape[0]
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    green, violet = is_ink["green"], is_ink["violet"]

    beam_poly = np.zeros(shape, np.uint8)
    cv2.fillPoly(beam_poly, [np.int32([(1160, 672), (235, 1040), (235, 1265), (640, 1235), (1160, 725)])], 1)
    bars = box(shape, 245, 1195, 905, 1300)
    R = {
        "green": green, "violet": violet, "red": is_ink["red"], "black": is_ink["black"],
        "orb": green & (np.hypot(xx - ORB[0], yy - ORB[1]) < ORB_R + 8),
        "beam": green & (beam_poly > 0) & ~bars,
        "bars": green & bars,
        "skull": green & box(shape, 1000, 80, 1250, 480),
        "columns": green & box(shape, 890, 585, 1270, 1045),
        "rollers": green & box(shape, 675, 90, 1005, 440),
        "name": green & box(shape, 1282, 105, 1360, 1300),
        "stipple": green & box(shape, 0, 880, 255, 1400),
    }
    rng = np.random.default_rng(7)
    sparkle = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), 2.2)
    sparkle = (sparkle - sparkle.min()) / (np.ptp(sparkle) + 1e-6)

    axis = np.float32(BEAM_END) - np.float32(DOOR)
    along = ((xx - DOOR[0]) * axis[0] + (yy - DOOR[1]) * axis[1]) / (axis @ axis)
    orb_r = np.hypot(xx - ORB[0], yy - ORB[1])
    S = dict(R=R, xx=xx, yy=yy, orb_r=orb_r, delay=orb_r / orb_r.max(), along=along, sparkle=sparkle,
             geo_g=data["geo_green"], geo_v=data["geo_violet"],
             eye_r=[np.hypot(xx - ex, yy - ey) for ex, ey in EYES],
             edges=edge_lines(reference, 50, 130), alpha=beam_alpha(),
             soft={k: soft(v) for k, v in R.items()})


def frame_light(t):
    R, xx, yy, soft_mask = S["R"], S["xx"], S["yy"], S["soft"]
    u = t / LOOP
    beat = u * BEATS
    thump = np.exp(-5.0 * (beat % 1.0))
    thump4 = np.exp(-2.5 * ((beat / 4) % 1.0))
    orb_r = S["orb_r"]
    add = np.zeros(xx.shape + (3,), np.float32)

    def glow(name, amount, colour):
        add[...] += (soft_mask[name] * amount)[..., None] * colour

    # power leaves the orb along the ink: sharp pulses riding the linework
    pulse_g = lobe(S["geo_g"] / 150 - 14 * u, 3.0)
    pulse_v = lobe(S["geo_v"] / 210 - 7 * u + 0.5, 2.0)
    gain_g = 0.38 + 0.62 * pulse_g
    plasma = (np.sin(xx / 140 + yy / 210 + TAU * 3 * u) + np.sin(xx / 80 - yy / 120 - TAU * 2 * u)
              + np.sin(orb_r / 60 - TAU * 7 * u)) / 3
    gain_v = 0.42 + 0.22 * plasma + 0.36 * pulse_v
    colour_g = cycle(GREENS, S["geo_g"] / 1700 - 2 * u)
    colour_v = cycle(VIOLETS, orb_r / 1500 + plasma * 0.12 - 3 * u)

    # features on the green layer
    gain_g = np.where(R["orb"], 0.62 + 0.38 * thump, gain_g)
    packets = lobe(S["along"] * 3.0 - 14 * u, 2.5)
    gain_g = np.where(R["beam"], 0.28 + 0.72 * packets, gain_g)
    gain_g = np.where(R["bars"], 0.35 + 0.65 * np.exp(-(wrapped((xx - 245) / 660 + 7 * u) * 660 / 55) ** 2), gain_g)
    gain_g = np.where(R["columns"], 0.35 + 0.65 * lobe((xx - 890) / 125 - 4 * u, 2.0), gain_g)
    gain_g = np.where(R["rollers"], 0.35 + 0.65 * lobe((xx + 0.45 * yy) / 260 - 4 * u, 2.0), gain_g)
    scan = np.exp(-(wrapped((yy - 80) / 400 - 4 * u) * 400 / 38) ** 2)
    gain_g = np.where(R["skull"], 0.50 + 0.18 * thump + 0.32 * scan, gain_g)
    gain_g = np.where(R["stipple"], 0.30 + 0.70 * lobe(S["sparkle"] * 3 + 7 * u, 4.0), gain_g)
    chase = np.exp(-(wrapped((yy - 105) / 1195 - 4 * u) * 1195 / 70) ** 2)
    gain_g = np.where(R["name"], 0.55 + 0.45 * chase, gain_g)
    colour_g = np.where(R["name"][..., None], rgb(0.75, 1.0, 0.25), colour_g)

    # mid-loop shockwave: the leading edge shows only the green layer, the trailing edge only violet
    radius = (t - 7.0) / 2.4 * 1700 if 7.0 <= t <= 9.6 else -1e4
    lead = np.exp(-((orb_r - radius) / 95) ** 2)
    trail = np.exp(-((orb_r - radius + 210) / 95) ** 2)
    gain_g = gain_g * (1 - trail) + 0.9 * lead
    gain_v = gain_v * (1 - lead) + 0.9 * trail

    # additive accents: white-hot orb core, corona on the tendrils, eye flares, glints
    add += (np.exp(-(orb_r / 58) ** 2) * (0.25 + 0.75 * thump))[..., None] * rgb(0.9, 1.0, 0.85)
    add += (soft_mask["violet"] * np.exp(-orb_r / 190) * (0.25 + 0.60 * thump4))[..., None] * rgb(1.0, 0.35, 0.95)
    for eye in S["eye_r"]:
        add += (np.exp(-(eye / 24) ** 2) * (0.35 + 0.85 * thump))[..., None] * rgb(1.0, 0.45, 1.0)
        add += (np.exp(-(eye / 75) ** 2) * 0.55 * thump4)[..., None] * rgb(0.8, 0.2, 1.0)
    glow("name", 0.55 * chase, rgb(1, 1, 0.8))
    glow("beam", 0.45 * packets * np.clip(1.2 - S["along"], 0, 1), rgb(0.85, 1.0, 0.9))

    # open: lines draw out of the orb, then the green layer floods, then the violet; close in reverse
    d = S["delay"]
    lines_on = smoothstep((t - 1.6 * d) / 0.45) * (1 - smoothstep((t - 14.2 - 1.4 * (1 - d)) / 0.45))
    green_on = smoothstep((t - 1.0 - 1.6 * d) / 0.8) * (1 - smoothstep((t - 13.3 - 1.3 * (1 - d)) / 0.8))
    violet_on = smoothstep((t - 1.9 - 1.6 * d) / 0.8) * (1 - smoothstep((t - 12.7 - 1.3 * (1 - d)) / 0.8))
    front = np.exp(-((d - t / 1.6) * 9) ** 2) + np.exp(-((d - (LOOP - 0.05 - t) / 1.4) * 9) ** 2)

    light = (colour_g * (gain_g * green_on * R["green"])[..., None]
             + colour_v * (gain_v * violet_on * R["violet"])[..., None]
             + (R["red"] * violet_on * (0.7 + 0.3 * thump))[..., None] * rgb(1.0, 0.12, 0.15)
             + add * np.maximum(green_on, violet_on)[..., None])
    light[R["black"]] *= 0.12  # a whisper of spill keeps accents soft-edged; black ink eats the rest
    line_level = lines_on * (1 - 0.85 * np.maximum(green_on, violet_on)) + 0.6 * front * lines_on
    light += (S["edges"] * line_level)[..., None] * cycle(LINES, d * 2 - 2 * u)
    return np.clip(light, 0, 1) * S["alpha"][..., None]


if __name__ == "__main__":
    if not os.path.exists(FIELDS) or "--rebuild" in sys.argv:
        build_fields()
    if "--regions" in sys.argv:
        setup()
        debug = (cv2.imread(os.path.join(CALIB, "reference.png")) * 0.3).astype(np.uint8)
        for name, colour in {"beam": (0, 255, 255), "bars": (255, 255, 0), "skull": (255, 255, 255), "columns": (0, 160, 255),
                             "rollers": (255, 0, 255), "name": (0, 255, 0), "stipple": (255, 120, 0), "orb": (0, 0, 255),
                             "red": (60, 60, 255)}.items():
            debug[S["R"][name]] = colour
        for ex, ey in EYES:
            cv2.circle(debug, (ex, ey), 24, (255, 0, 255), 2)
        geo = np.clip(S["geo_g"] / 1800, 0, 1)
        heat = cv2.applyColorMap((geo * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        heat[~S["R"]["green"]] = 0
        cv2.imwrite(os.path.join(CALIB, "hox_regions.jpg"), cv2.resize(np.hstack([debug, heat]), (1600, 800), interpolation=cv2.INTER_AREA))
    elif "--sheet" in sys.argv:
        contact_sheet(setup, frame_light, [0.5, 1.4, 2.4, 4.0, 6.0, 7.6, 8.4, 11.0, 14.6],
                      os.path.join(CALIB, "contact_sheet_hox.jpg"))
    else:
        render_loop(setup, frame_light, OUT, LOOP, FPS, frames_per_beat=LOOP * FPS / BEATS)
