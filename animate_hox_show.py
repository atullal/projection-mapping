"""Black Market Brass - Hox: a 57.6 s show of eight scenes, looping seamlessly (starts and ends dark).

    python animate_hox_show.py [--sheet]

Reuses the masks / distance fields from animate_hox.  Everything sits on a 100 BPM grid
(one beat = 18 frames); each scene lasts 12 beats and wipes in radially from the orb.
"""
import os
import sys

import cv2
import numpy as np

import animate_hox as hox
from animkit import TAU, contact_sheet, cycle, lobe, render_loop, rgb, smoothstep, wrapped
from mapping import CALIB, ROOT

FPS, BEAT = 30, 0.6
SCENE, WIPE = 12 * BEAT, BEAT          # seconds
OUT = os.path.join(ROOT, "frames", "hox_show")
H = None                                # per-process scene data


def setup():
    global H
    hox.setup()
    H = dict(hox.S)
    xx, yy = H["xx"], H["yy"]
    H["orb_a"] = np.arctan2(yy - hox.ORB[1], xx - hox.ORB[0]) / TAU
    H["eye_a"] = [np.arctan2(yy - ey, xx - ex) / TAU for ex, ey in hox.EYES]
    rng = np.random.default_rng(11)
    H["rain"] = [dict(width=w, speed=rng.uniform(500, 1100, 1400 // w + 1), offset=rng.uniform(0, 2200, 1400 // w + 1),
                      tail=rng.uniform(380, 800, 1400 // w + 1)) for w in (26, 41)]
    spectrum = np.linspace(1.0, 0.65, 20)[:, None]          # bass on the left is louder
    H["levels"] = np.clip(rng.uniform(0.25, 1.0, (20, 14)) * spectrum + rng.uniform(0, 0.25, (20, 14)), 0, 1)


def compose(gain_g, colour_g, gain_v, colour_v, add=None, red=0.6):
    R = H["R"]
    light = (np.asarray(colour_g, np.float32) * (gain_g * R["green"])[..., None]
             + np.asarray(colour_v, np.float32) * (gain_v * R["violet"])[..., None]
             + (R["red"] * red)[..., None] * rgb(1.0, 0.12, 0.15))
    if add is not None:
        light = light + add
    light[R["black"]] *= 0.12
    return light


def eye_flares(level, halo=0.0):
    add = np.zeros(H["xx"].shape + (3,), np.float32)
    for eye in H["eye_r"]:
        add += (np.exp(-(eye / 24) ** 2) * level)[..., None] * rgb(1.0, 0.45, 1.0)
        if halo:
            add += (np.exp(-(eye / 80) ** 2) * halo)[..., None] * rgb(0.8, 0.2, 1.0)
    return add


def ignition(s):
    d, u = H["delay"], s / SCENE
    pulse_g = lobe(H["geo_g"] / 150 - 6 * u, 3.0)
    pulse_v = lobe(H["geo_v"] / 210 - 3 * u + 0.5, 2.0)
    thump = np.exp(-5.0 * ((s / BEAT) % 1.0))
    lines_on = smoothstep((s - 1.4 * d) / 0.4)
    green_on = smoothstep((s - 0.9 - 1.4 * d) / 0.7)
    violet_on = smoothstep((s - 1.7 - 1.4 * d) / 0.7)
    add = (np.exp(-(H["orb_r"] / 58) ** 2) * (0.25 + 0.75 * thump) * green_on)[..., None] * rgb(0.9, 1.0, 0.85)
    add += eye_flares(0.35 + 0.85 * thump) * violet_on[..., None]
    light = compose((0.38 + 0.62 * pulse_g) * green_on, cycle(hox.GREENS, H["geo_g"] / 1700 - u),
                    (0.45 + 0.55 * pulse_v) * violet_on, cycle(hox.VIOLETS, H["orb_r"] / 1500 - 2 * u), add,
                    red=0.6 * violet_on)
    front = np.exp(-((d - s / 1.4) * 9) ** 2)
    level = lines_on * (1 - 0.85 * np.maximum(green_on, violet_on)) + 0.6 * front * lines_on
    return light + (H["edges"] * level)[..., None] * cycle(hox.LINES, d * 2 - u)


def radar(s):
    phase = (s / 3.6 - H["orb_a"]) % 1.0                    # 0 just behind the arm
    trail = np.exp(-4.5 * phase)
    arm = np.exp(-(np.minimum(phase, 1 - phase) / 0.004) ** 2)
    rings = lobe(H["orb_r"] / 175, 14.0)
    add = (arm * 0.8 + rings * trail * 0.35)[..., None] * rgb(0.6, 1.0, 0.7)
    add += (np.exp(-(H["orb_r"] / 50) ** 2) * 0.7)[..., None] * rgb(0.8, 1.0, 0.8)
    return compose(0.06 + 0.94 * trail, rgb(0.2, 1.0, 0.4), 0.05 + 0.55 * trail, rgb(0.45, 0.3, 1.0), add, red=0.2)


def layer_dance(s):
    beats = s / BEAT
    size = 350 if beats < 7 else 175
    ix, iy = np.floor(H["xx"] / size), np.floor(H["yy"] / size)
    green_turn = (ix + iy + np.floor(beats)) % 2 == 0
    flash = np.exp(-6.0 * (beats % 1.0))
    level = 0.72 + 0.28 * flash
    return compose(np.where(green_turn, level, 0.02), cycle(hox.GREENS, ix * 0.37 + iy * 0.21 + beats / 12),
                   np.where(green_turn, 0.02, level), cycle(hox.VIOLETS, ix * 0.29 + iy * 0.43 + beats / 12),
                   eye_flares(0.5 * flash), red=0.4)


def laser_eyes(s):
    thump = np.exp(-5.0 * ((s / BEAT) % 1.0))
    beam = np.zeros(H["xx"].shape, np.float32)
    for i, (angle, dist) in enumerate(zip(H["eye_a"], H["eye_r"])):
        # 0.25 turns = straight down, 0.5 = left: sweep the cover below and beside the skull
        aim = 0.37 + 0.14 * np.sin(TAU * s / 3.6 + i * 2.6) + 0.025 * np.sin(TAU * s / 0.9 + i)
        beam += np.exp(-(wrapped(angle - aim) / 0.010) ** 2) * np.clip(dist / 60, 0, 1)
    beam = np.clip(beam, 0, 1)
    add = beam[..., None] * rgb(1.0, 0.5, 1.0) * 0.7 + eye_flares(0.8 + 0.4 * thump, halo=0.5 + 0.4 * thump)
    gain_g = np.where(H["R"]["skull"], 0.35 + 0.15 * thump, 0.08) + 0.9 * beam
    return compose(gain_g, rgb(0.65, 1.0, 0.8), 0.10 + 0.9 * beam, rgb(1.0, 0.35, 1.0), add, red=0.2)


def rainfall(layer, s):
    column = (H["xx"] // layer["width"]).astype(int)
    tail = layer["tail"][column]
    head = (layer["offset"][column] + layer["speed"][column] * s) % (1400 + tail + 200) - 100
    behind = head - H["yy"]
    return np.where(behind >= 0, np.exp(-behind / tail), 0).astype(np.float32), np.exp(-(behind / 10) ** 2)


def data_rain(s):
    fall_g, head_g = rainfall(H["rain"][0], s)
    fall_v, head_v = rainfall(H["rain"][1], s + 3.0)
    add = (head_g * 0.55)[..., None] * rgb(0.85, 1.0, 0.9) + (head_v * 0.35)[..., None] * rgb(0.9, 0.7, 1.0)
    return compose(0.05 + 0.95 * fall_g, rgb(0.15, 1.0, 0.4), 0.05 + 0.75 * fall_v, rgb(0.6, 0.25, 1.0), add, red=0.3)


def equalizer(s):
    beats = s / BEAT
    band = np.clip(H["xx"] // 70, 0, 19).astype(int)
    now, fade = int(beats) % 14, 0.45 + 0.55 * np.exp(-2.0 * (beats % 1.0))
    level = (0.12 + 0.88 * H["levels"][:, now] * fade)[band]
    height = np.clip(1 - H["yy"] / 1340, 0, 1)
    lit = smoothstep((level - height) / 0.012) * ((H["xx"] % 70) > 6)
    cap = np.exp(-((height - level) / 0.007) ** 2) * ((H["xx"] % 70) > 6)
    h = height[..., None]
    return compose(0.04 + 0.96 * lit, (1 - h) * rgb(0.05, 1.0, 0.8) + h * rgb(0.85, 1.0, 0.1),
                   0.04 + 0.96 * lit, (1 - h) * rgb(0.25, 0.4, 1.0) + h * rgb(1.0, 0.15, 0.8),
                   cap[..., None] * rgb(1, 1, 1) * 0.8, red=0.3)


def hue_vortex(s):
    hue = (2 * H["orb_a"] + H["orb_r"] / 450 - s / 2.4) % 1.0
    hsv = np.stack([hue * 360, np.ones_like(hue), np.ones_like(hue)], -1).astype(np.float32)
    wheel = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    thump = np.exp(-5.0 * ((s / BEAT) % 1.0))
    add = (np.exp(-(H["orb_r"] / 58) ** 2) * (0.3 + 0.7 * thump))[..., None] * rgb(1, 1, 1) + eye_flares(0.6 * thump)
    return compose(0.92, wheel, 0.92, wheel, add, red=0.5)


def spotlights_finale(s):
    xx, yy, r = H["xx"], H["yy"], H["orb_r"]
    fade = 1 - smoothstep((s - 5.2) / 0.5)
    spot_a = np.exp(-(np.hypot(xx - (700 + 520 * np.sin(TAU * s / 3.6)), yy - (680 + 480 * np.sin(TAU * s / 2.4 + 1))) / 270) ** 2)
    spot_b = np.exp(-(np.hypot(xx - (700 + 520 * np.sin(TAU * s / 2.4 + 2)), yy - (680 + 480 * np.cos(TAU * s / 3.6))) / 270) ** 2)
    gain_g = (0.10 + spot_a + 0.35 * spot_b) * fade
    gain_v = (0.10 + spot_b + 0.35 * spot_a) * fade
    for start in (5.4, 6.0, 6.6):                            # three shockwaves, one per beat
        radius = (s - start) / 1.1 * 1700 if s >= start else -1e4
        gain_g = gain_g + np.exp(-((r - radius) / 95) ** 2)
        gain_v = gain_v + np.exp(-((r - radius + 190) / 95) ** 2)
    core = np.exp(-(r / 58) ** 2) * (0.5 * fade + 1.0 * (s > 5.4)) * (1 - smoothstep((s - 7.3) / 0.4))
    light = compose(np.clip(gain_g, 0, 1), rgb(0.5, 1.0, 0.65), np.clip(gain_v, 0, 1), rgb(0.8, 0.4, 1.0),
                    core[..., None] * rgb(0.9, 1.0, 0.85), red=0.5 * fade)
    # last gasp: the line-art flashes on and is pulled back into the orb
    d = H["delay"]
    pull = smoothstep((s - 6.7) / 0.3) * (1 - smoothstep((d - (1 - (s - 6.9) / 0.8)) / 0.06))
    return light + (H["edges"] * pull * 0.8)[..., None] * cycle(hox.LINES, d * 2)


SCENES = [ignition, radar, layer_dance, laser_eyes, data_rain, equalizer, hue_vortex, spotlights_finale]
TOTAL = SCENE * len(SCENES)


def frame_light(t):
    index = min(int(t // SCENE), len(SCENES) - 1)
    local = t - index * SCENE + (WIPE if index else 0)      # scenes after the first start one wipe early
    light = SCENES[index](local)
    upcoming = index + 1
    into = t - (upcoming * SCENE - WIPE)
    if upcoming < len(SCENES) and into >= 0:                 # the next scene wipes in from the orb
        mask = smoothstep((into / WIPE * 1.3 - H["delay"]) / 0.3)[..., None]
        edge = np.exp(-((into / WIPE * 1.3 - 0.15 - H["delay"]) / 0.05) ** 2)[..., None] * rgb(0.9, 1.0, 0.9) * 0.5
        light = light * (1 - mask) + SCENES[upcoming](into) * mask + edge * (H["R"]["green"] | H["R"]["violet"])[..., None]
    return np.clip(light, 0, 1) * H["alpha"][..., None]


if __name__ == "__main__":
    if "--sheet" in sys.argv:
        times = [i * SCENE + offset for i in range(len(SCENES)) for offset in (2.2, 5.0)]
        times[-1] = 7 * SCENE + 5.6                          # catch a shockwave in the finale
        contact_sheet(setup, frame_light, times, os.path.join(CALIB, "contact_sheet_hox_show.jpg"), tile=350, columns=4)
    else:
        render_loop(setup, frame_light, OUT, TOTAL, FPS, frames_per_beat=BEAT * FPS)
