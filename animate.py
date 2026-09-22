"""Render the Life & Love animation: a seamless 16 s loop authored in artwork space and
warped to the projector.

    python animate.py [--sheet]     --sheet only writes a contact sheet of sample frames

Light on print multiplies with the ink, so the base layer is each ink's own colour (it
glows), black ink is left unlit (it deepens), and real hue changes are kept to the
cream/gold areas that behave like a screen.  Every periodic term completes a whole number
of cycles in LOOP seconds, and the loop opens/closes on darkness, so it repeats seamlessly.
"""
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

from mapping import CALIB, ROOT
from regions import INKS, build
from render import ProjectorWarp

LOOP, FPS = 16.0, 30
FRAMES = int(LOOP * FPS)
BEATS = 24  # 90 BPM
OUT = os.path.join(ROOT, "frames", "life_and_love")
TAU = 2 * np.pi

S = None  # per-process scene, built once by setup()


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


def setup():
    global S
    reference = cv2.imread(os.path.join(CALIB, "reference.png"))
    R = build(reference)
    n = reference.shape[0]
    yy, xx = np.mgrid[:n, :n].astype(np.float32)

    # base light: each ink's own colour, flattened (no paper grain), pushed to full brightness
    flat = cv2.bilateralFilter(cv2.medianBlur(reference, 5), 9, 50, 9).astype(np.float32) / 255
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.35, 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.25, 0, 1)
    base = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    base[R["black"]] = 0

    def polar(centre):
        dx, dy = xx - centre[0], yy - centre[1]
        return np.hypot(dx, dy), np.arctan2(dy, dx) / TAU

    def soft(mask, sigma=1.2):
        return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)

    grey = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    edges = soft(cv2.dilate(cv2.Canny(grey, 60, 140), np.ones((3, 3), np.uint8)) > 0, 1.0)

    sleeve = cv2.imread(os.path.join(CALIB, "sleeve.png"), 0) > 0
    covered = cv2.imread(os.path.join(CALIB, "covered.png"), 0) > 0
    # tight at the sleeve's own edges (just enough to avoid spill onto what is behind it)...
    alpha = soft(cv2.erode(sleeve.astype(np.uint8), np.ones((5, 5), np.uint8)), 2.0)
    # ...but a long feather where the beam itself runs out, so that cut-off reads as a fade.
    # Pad first: the artwork's own border is not a beam limit and must not be feathered.
    padded = cv2.copyMakeBorder(covered.astype(np.uint8), 60, 60, 60, 60, cv2.BORDER_REPLICATE)
    alpha *= smoothstep((cv2.distanceTransform(padded, cv2.DIST_L2, 5)[60:-60, 60:-60] - 2) / 38)

    letters = R["life"] | R["love"]
    inward = cv2.distanceTransform(letters.astype(np.uint8), cv2.DIST_L2, 5)
    amp_halo = np.exp(-cv2.distanceTransform((~R["amp"]).astype(np.uint8), cv2.DIST_L2, 5) / 22) * R["teal"]
    notes_halo = soft(cv2.dilate(R["notes"].astype(np.uint8), np.ones((15, 15), np.uint8)), 6) * R["panel_orange"]
    bolt_halo = soft(cv2.dilate(R["bolt"].astype(np.uint8), np.ones((21, 21), np.uint8)), 8) * R["panel_orange"]

    ink = R["ink"]
    names = list(INKS)
    screenlike = (ink == names.index("cream")) | (ink == names.index("gold"))

    burst_r, burst_a = polar(R["burst_centre"])
    medal_r, medal_a = polar(R["medal_centre"])
    sun_r, _ = polar(R["sun_centre"])
    _, panel_a = polar((990, 572))
    speaker_r, _ = polar((748, 690))
    cymbal_r, _ = polar((768, 488))
    mic_r, _ = polar((990, 600))

    S = dict(R=R, n=n, xx=xx, yy=yy, base=base, edges=edges, alpha=alpha, inward=inward,
             amp_halo=amp_halo, notes_halo=notes_halo, bolt_halo=bolt_halo, screenlike=screenlike,
             burst_r=burst_r, burst_a=burst_a, medal_r=medal_r, medal_a=medal_a, sun_r=sun_r,
             panel_a=panel_a, speaker_r=speaker_r, cymbal_r=cymbal_r, mic_r=mic_r,
             delay=burst_r / burst_r.max(), soft={k: soft(v) for k, v in R.items()
                                                  if isinstance(v, np.ndarray) and v.dtype == bool},
             warp=ProjectorWarp())


WARM = [rgb(1.0, 0.78, 0.30), rgb(1.0, 0.45, 0.55), rgb(1.0, 0.62, 0.18), rgb(1.0, 0.88, 0.62)]
SKY_TOP = [rgb(1.0, 0.80, 0.40), rgb(0.95, 0.45, 0.70), rgb(0.55, 0.38, 0.95), rgb(1.0, 0.62, 0.45)]
SKY_LOW = [rgb(1.0, 0.58, 0.20), rgb(1.0, 0.50, 0.30), rgb(0.98, 0.42, 0.55), rgb(1.0, 0.70, 0.25)]


def cycle(colours, phase):
    """Smoothly loop through a list of colours; phase in turns (array or scalar)."""
    phase = np.asarray(phase, np.float32) % 1.0 * len(colours)
    i = np.floor(phase).astype(int) % len(colours)
    f = smoothstep(phase - np.floor(phase))[..., None]
    table = np.stack(colours)
    return table[i] * (1 - f) + table[(i + 1) % len(colours)] * f


def frame_light(t):
    R, xx, yy, soft = S["R"], S["xx"], S["yy"], S["soft"]
    u = t / LOOP                      # loop phase 0..1
    beat = t / LOOP * BEATS           # beats elapsed
    thump = np.exp(-5.0 * (beat % 1.0))          # decays after every beat
    thump2 = np.exp(-3.0 * ((beat / 2) % 1.0))   # every other beat

    gain = np.full(xx.shape, 0.62, np.float32)
    tint = None                       # replaces base colour on screen-like ink where set
    add = np.zeros(xx.shape + (3,), np.float32)

    def glow(mask_name, amount, colour):
        add[...] += (soft[mask_name] * amount)[..., None] * colour

    # teal field: slow currents of light drifting through it
    current = (np.sin(xx / 170 + yy / 260 - TAU * 3 * u) + np.sin(xx / 95 - yy / 140 + TAU * 2 * u)
               + np.sin(yy / 75 + TAU * 5 * u)) / 3
    gain = np.where(R["teal"], 0.58 + 0.30 * current, gain)
    add += (S["amp_halo"] * (0.25 + 0.55 * thump2))[..., None] * rgb(0.55, 1.0, 0.95)

    # lettering: neon pulses travelling across the inline stripes, LIFE outside-in, LOVE inside-out
    life = lobe(S["inward"] / 46 - 12 * u, 2.0)
    love = lobe(S["inward"] / 46 + 12 * u + 0.5, 2.0)
    gain = np.where(R["life"], 0.30 + 0.70 * life, gain)
    gain = np.where(R["love"], 0.30 + 0.70 * love, gain)
    letter_tint = cycle(WARM, xx / 1400 * 0.8 - 2 * u)
    sweep = np.exp(-(wrapped((xx + 0.35 * yy) / 1700 - 4 * u) * 1700 / 46) ** 2)
    glow("life", 0.75 * sweep, rgb(1, 1, 1))
    glow("love", 0.75 * sweep, rgb(1, 1, 1))

    # sunburst: a three-armed beam sweeps the rays while ripples leave the eye on the beat
    arms = lobe(3 * S["burst_a"] - 4 * u, 2.5)
    ripple = np.exp(-((S["burst_r"] - (110 + 560 * ((beat / 2) % 1.0))) / 42) ** 2)
    gain = np.where(R["rays"], 0.22 + 0.78 * np.maximum(arms, ripple), gain)
    ray_tint = cycle(WARM, S["burst_r"] / 900 - 3 * u)
    rings = lobe(S["burst_r"] / 34 - 6 * u, 1.5)
    gain = np.where(R["eye"], 0.35 + 0.65 * rings, gain)
    add += (np.exp(-(S["burst_r"] / 30) ** 2) * thump * 0.9)[..., None] * rgb(1, 0.95, 0.8)

    # instrument panel
    gain = np.where(R["panel_orange"], 0.60 + 0.22 * thump, gain)
    gain = np.where(R["panel_items"], 0.85, gain)
    glow("panel_stroke", 0.95 * lobe(S["panel_a"] - 8 * u, 10), rgb(1, 0.97, 0.85))
    gain = np.where(R["panel_stroke"], 0.35, gain)
    for zone, radius, speed in (("waves_cymbal_zone", S["cymbal_r"], 1.0), ("waves_mic_zone", S["mic_r"], 1.0)):
        gain = np.where(R[zone], 0.35 + 0.65 * lobe(radius / 17 - speed * beat, 1.5), gain)
    cone = lobe(S["speaker_r"] / 13 - beat, 1.0) * (0.35 + 0.65 * thump)
    gain = np.where(R["speaker"], 0.25 + 0.75 * cone, gain)
    glow("speaker", 0.35 * thump * np.exp(-(S["speaker_r"] / 40) ** 2), rgb(1, 0.85, 0.4))
    key_u = (xx - 1105) / 225
    running = np.exp(-(wrapped(key_u - beat / 2) * 225 / 16) ** 2)
    gain = np.where(R["keys"], 0.30, gain)
    glow("keys", 0.95 * running, rgb(1, 1, 0.95))
    strike = (np.sin(TAU * 11 * beat) > 0.2) * np.exp(-4.0 * ((beat / 4 + 0.5) % 1.0))
    glow("bolt", 1.0 * strike, rgb(1, 1, 0.8))
    add += (S["bolt_halo"] * 0.6 * strike)[..., None] * rgb(1, 0.85, 0.35)
    slide = np.exp(-(((xx - 900) / 470 - ((beat / 4) % 1.0) * 1.3 + 0.15) * 470 / 34) ** 2)
    glow("bass", 0.85 * slide, rgb(0.85, 1, 1))
    add += (S["notes_halo"] * (0.20 + 0.55 * lobe(beat / 2 + xx / 600, 3)))[..., None] * rgb(1, 0.9, 0.5)

    # medallion: two lights orbit the ring, a soft spotlight drifts over the bassist
    gain = np.where(R["medal_ring"], 0.30 + 0.70 * lobe(2 * S["medal_a"] - 8 * u, 4), gain)
    spot = np.exp(-(((xx - 302) * 0.8 + (yy - 1105) * 0.6) / 150 - 1.6 * np.sin(TAU * 2 * u)) ** 2)
    gain = np.where(R["medal_inside"], 0.70 + 0.30 * spot, gain)
    add += (soft["medal_inside"] * spot * 0.22)[..., None] * rgb(1, 0.95, 0.85)

    # sky: dusk colours cycle on the cream/gold ink, light drifts through the clouds, the sun breathes
    height = np.clip((yy - 890) / 330, 0, 1)[..., None]
    sky_tint = cycle(SKY_TOP, u) * (1 - height) + cycle(SKY_LOW, u) * height
    drift = 0.5 + 0.5 * np.sin(TAU * (xx / 520 - 3 * u) + yy / 80)
    gain = np.where(R["sky"], 0.72, gain)
    gain = np.where(R["clouds"], 0.50 + 0.50 * drift, gain)
    halo = np.exp(-S["sun_r"] / 230) * lobe(S["sun_r"] / 85 - 8 * u, 1.5)
    add += (soft["sky"] * halo * 0.65)[..., None] * rgb(1, 0.62, 0.22)
    gain = np.where(R["sun"], 0.72 + 0.28 * thump2, gain)
    add += (np.exp(-(S["sun_r"] / (S["R"]["sun_radius"] * 0.9)) ** 2) * 0.25 * thump2)[..., None] * rgb(1, 0.8, 0.5)

    # ridgeline rim-light with a glint that runs the mountains; wordmark shimmer
    glint = np.exp(-(wrapped(xx / 1400 - 4 * u) * 1400 / 70) ** 2)
    glow("ridge", 0.30 + 0.70 * glint, rgb(1, 0.95, 0.85))
    word_sweep = np.exp(-(wrapped((xx - 785) / 535 - 4 * u - 0.3) * 535 / 40) ** 2)
    gain = np.where(R["wordmark"], 0.55 + 0.20 * thump, gain)
    glow("wordmark", 0.80 * word_sweep, rgb(1, 1, 1))

    # compose: own-colour base, except screen-like inks that take a projected hue
    colour = S["base"].copy()
    for mask, hue in ((R["life"] | R["love"] | R["wordmark"], letter_tint), (R["rays"], ray_tint), (R["sky"], sky_tint)):
        chosen = mask & S["screenlike"]
        colour[chosen] = hue[chosen]
    light = colour * gain[..., None] + add
    light[R["black"]] *= 0.0

    # open by drawing the line-art outward from the eye, flood in colour, reverse to close
    d = S["delay"]
    lines_on = smoothstep((t - 1.7 * d) / 0.45) * (1 - smoothstep((t - 14.1 - 1.4 * (1 - d)) / 0.45))
    colour_on = smoothstep((t - 1.1 - 1.7 * d) / 0.9) * (1 - smoothstep((t - 12.9 - 1.4 * (1 - d)) / 0.9))
    front = np.exp(-((d - t / 1.7) * 9) ** 2) + np.exp(-((d - (LOOP - 0.05 - t) / 1.4) * 9) ** 2)
    line_level = lines_on * (1 - 0.8 * colour_on) + 0.6 * front * lines_on
    line_colour = cycle(WARM, d * 1.5 - 2 * u)
    light = light * colour_on[..., None] + (S["edges"] * line_level)[..., None] * line_colour

    return np.clip(light * S["alpha"][..., None], 0, 1)


def render(index):
    light = frame_light(index / FPS)
    image = (light * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, f"f{index:04d}.jpg"), S["warp"](image), [cv2.IMWRITE_JPEG_QUALITY, 93])
    return index


def contact_sheet(times):
    setup()
    tiles = [cv2.resize((frame_light(t) * 255).astype(np.uint8), (466, 466), interpolation=cv2.INTER_AREA) for t in times]
    for tile, t in zip(tiles, times):
        cv2.putText(tile, f"{t:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(os.path.join(CALIB, "contact_sheet.jpg"), np.vstack(rows))


if __name__ == "__main__":
    if "--sheet" in sys.argv:
        contact_sheet([0.6, 1.6, 3.0, 5.0, 7.3, 9.5, 11.5, 13.8, 15.2])
    else:
        os.makedirs(OUT, exist_ok=True)
        for stale in os.listdir(OUT):
            os.remove(os.path.join(OUT, stale))
        with Pool(8, initializer=setup) as pool:
            for done in pool.imap_unordered(render, range(FRAMES), chunksize=4):
                if done % 60 == 0:
                    print(f"  frame {done}/{FRAMES}", flush=True)
        ProjectorWarp().write_meta(OUT, FPS, frames_per_beat=FRAMES / BEATS)
        print(f"rendered {FRAMES} frames -> {OUT}")
