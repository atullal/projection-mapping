# History and design decisions

## Origin

The project started as a ChatGPT conversation ("Find free 3D mapping software") about doing
projection mapping on a Mac with a Nebula Capsule 3 and no LiDAR. The recommended
off-the-shelf route was TouchDesigner (Kantan Mapper / camSchnappr) + Blender; the owner
instead wanted the assistant to *see* the objects through a camera and map them itself.
That session built a prototype that turned a plant pot into an animated fish bowl and
brought a vinyl cover to life, and claimed to save three reusable "skills"
(`projection-mapping`, `animate-printed-artwork`, `projected-video-scenes`). Those skills
were never found on the Mac, so this repository is a from-scratch rebuild of the same idea
in plain Python, done with Claude Code on 2026-09-18/19.

## Decisions and why

**Structured light instead of manual corner mapping.** The whole point was zero manual
alignment. Gray codes with inverse patterns are robust to ambient light and to the
sleeve's texture, and decode in ~26 s. 4 px stripes were chosen because 1–2 px stripes
blur into grey in a phone camera at this distance.

**Two processes.** Discovered the hard way: `import cv2, pygame` in one process prints 17
duplicate-class warnings and was expected to crash. `opencv-python-headless` still bundles
SDL2 via ffmpeg, so the split is structural, not a packaging fix.

**Homography + cubic.** The first sleeve leaned against a window and bowed; a homography
left a smooth 5 px error with a clear spatial pattern (checked on a 5×5 residual grid).
A cubic residual on top brought it to the quantisation floor. Simpler than a full mesh and
fits in one `lstsq`.

**Register on features, not the border.** The printed sleeve is trimmed 1–2 % tighter than
the digital file and its colours differ (mint vs yellow-green, blue-violet vs purple), so
SIFT on CLAHE greys with the reference blurred to match the camera's softness. Chaining
through the same rectified image on both sides makes corner error cancel.

**Author in artwork space.** Animations are written against `reference.png` coordinates so
the same code renders correctly after any recalibration, and so region masks can be built
from the clean digital art instead of a noisy photo.

**Own-colour light.** The first instinct — project arbitrary colours — looks bad on print
because light multiplies with ink. The rule set in `ANIMATION_GUIDE.md` came from looking
at camera photos of the first attempts.

**Pre-rendered frames.** `frame_light` in numpy is ~0.5 s per frame; a 16 s loop renders in
40 s on 8 cores and plays back trivially. Real-time would need a GPU port; not needed for
a fixed rig.

**Beat sync via a beat *clock*, not per-onset triggers.** Animations are authored on a beat
grid; the tracker provides a phase-locked clock; the player selects frames from that
clock. This makes hits land on predicted beats (no reaction lag), survives quiet bars, and
follows tempo changes. The harmonic scoring (L + 2L + 4L) was added after watching the
tracker hop between 83/104/139 BPM on syncopated afrobeat.

**Feather the beam edge, don't move the projector.** The owner preferred not to re-aim
the projector when the beam fell short of the sleeve's bottom; the `covered.png` mask with
a 40 px feather makes the cut-off read as an intentional fade.

## Timeline

- 2026-09-18 22:00 — repo created, camera/projector plumbing, first calibration.
- 22:30 — Skinshape *Life & Love* animated (regions.py, animate.py).
- 22:40 — Black Market Brass *Hox*: animkit.py extracted, geodesic pulses, layer selection.
- 22:47 — eight-scene Hox show.
- 22:58 — beat.py + synced playback; tempo-hopping fix.
- 2026-09-19 03:00 — realign after the projector moved; relative-contrast decode gate and
  hull-based sleeve mask added after a dark-room decode failure. Mac slept; projector and
  phone camera dropped off.
- 2026-09-22 — repository documented and committed for collaboration.
