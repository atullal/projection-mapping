# Guide for AI assistants (Claude, DeepSeek, Codex, anyone)

This file is the context an AI collaborator needs before changing anything. It is short on
purpose; the long explanations are in `docs/`. Read this, then `docs/ARCHITECTURE.md`.

## What this project is

A projector + webcam rig that automatically calibrates itself to a vinyl record sleeve and
projects animated light aligned to the printed artwork, optionally beat-synced to music in
the room. Python 3.12, OpenCV, numpy, pygame-ce, sounddevice. macOS today.

## Invariants — do not break these

1. **cv2 and pygame never share a process.** OpenCV's bundled ffmpeg ships its own SDL2 which
   clashes with pygame's (you get "Class SDLApplication is implemented in both…" warnings and
   eventually crashes). `projector.py` is pygame-only and is driven over a localhost socket
   (`projector_client.py`, port 47800). Everything that touches the camera or images is
   cv2-only. `beat.py` (numpy + sounddevice) is imported *into* projector.py and must stay
   free of cv2.
2. **Coordinate spaces are explicit.** Camera pixels (1920×1080 photo), projector pixels
   (1920×1080 output), artwork pixels (a `SIZE=1400` square, `REF=1400` in register.py).
   Every array/model is named for the direction it maps: `cam2proj`, `proj2cam`, `H_ac`
   (artwork→camera), `render_map` (projector→artwork). Keep that naming.
3. **Animations are authored in artwork space and warped once.** A `frame_light(t)` returns a
   1400×1400×3 float BGR image in [0,1]; `render.ProjectorWarp` remaps it into projector
   space and crops to the sleeve's bounding box (`meta.json` `offset` says where to blit).
   Never author directly in projector pixels.
4. **Frames sit on a beat grid.** Every animation declares its tempo (`BEATS` per loop or
   `BEAT` seconds) and its renderer writes `frames_per_beat` into `meta.json`. In sync mode
   `projector.py` picks the frame from the music's beat count, so hits authored on integer
   beats land on real beats. If you change loop length or tempo, `frames_per_beat` must
   follow (it is computed, not typed — keep it that way).
5. **Loops are seamless.** Periodic terms complete whole cycles per loop; loops open and close
   on darkness. Contact sheets should show t≈0 and t≈end both dark.
6. **Light multiplies with ink.** See the design rules in `docs/ANIMATION_GUIDE.md`. Black ink
   cannot be lit; don't waste effort trying. Colour light on a two-ink print *selects a layer*.

## Verification loop (always do this after a change)

- Geometry change → `python calibrate.py 1 && python mapping.py <x> <y> 1 && python register.py <ref>`
  and read the printed RMS numbers: cover fit ~1.8 px (floor is 1.63 from 4-px stripes),
  registration ~1 projector px. Then project the edge-outline test (see `docs/CALIBRATION.md`,
  "Verify") and *look at a camera photo*: white lines must sit on printed lines.
- Animation change → `python animate_<album>.py --sheet` and view the contact sheet, then
  full render, play, and photograph the live projection over one loop.
- Playback/sync change → `python projector_client.py status` for ~30 s; BPM should be
  stable and confidence > 2.
- Nothing here has unit tests; the camera *is* the test. Do not claim alignment works
  without a photo.

## Things that went wrong before (so you don't repeat them)

- Assumed the MacBook camera was watching the album; it was an iPhone Continuity Camera at
  index 1. Indices shift when devices connect/disconnect. Run `check_rig.py` and look at the
  probe images before every session.
- macOS camera permission: OpenCV requests it and fails instantly; the dialog disappears if the
  process exits. Keep a process retrying `VideoCapture` until the user clicks Allow
  (`calibrate.py`'s first `open_camera` will do this if you just rerun it).
- A record sleeve leaning against something bows. A plain homography left ~5 px error;
  homography + cubic residual (`mapping.fit_smooth`) gets to the quantisation floor. Don't gate
  plane membership with a tight homography-only RANSAC before the cubic fit — it clips the
  bowed corners.
- A quadrilateral fitted to the decoded region can be the *beam's* edge, not the sleeve's.
  Check `corners (projector)` for y≈0 or y≈1080.
- In a dark room the camera's noise reduction flattens fine stripes on dark ink; decoding
  survives only on bright ink. Contrast gating is therefore relative per pixel
  (`calibrate.decode`), and `mapping.cover_models` no longer needs a contiguous decoded blob.
- Beat tracking on syncopated music (afrobeat) hopped between 3/4/5-sixteenth groupings
  (83/104/139 BPM). Fixed by scoring autocorrelation at L + 2L + 4L and requiring a rival
  tempo to win by 20% for 4 s. Don't "simplify" that back to a single-lag argmax.
- The Mac slept mid-run once: the projector and iPhone both dropped off. A step that "hangs"
  may have finished — read the task output before debugging.
- Printed sleeves are trimmed 1–2% tighter than the digital cover, and printed colours differ
  from the file. Register on features, never on the outer border or on colour.

## Conventions

- Scripts are flat in the repo root and import each other by module name; run them from the
  repo root with the venv active (or `.venv/bin/python`).
- Per-album code is `animate_<album>.py` (+ `regions.py`-style segmentation if needed).
  Shared helpers go in `animkit.py`. Keep album-specific pixel coordinates in the album file.
- Debug images go to `calib/` (kept per rig) or `captures/` (throwaway). Both are gitignored.
- Comments explain *why* (physics, a failure that was seen), not what the code does.
- Don't commit `reference/` (copyright), `frames/` (size), `calib*/` (rig-specific), `.venv`.

## Where to look

| Question | File |
|---|---|
| How do the pieces fit, what are the data files | `docs/ARCHITECTURE.md` |
| Hardware, permissions, first run | `docs/SETUP.md` |
| Structured light, fitting, registration, verify, troubleshooting | `docs/CALIBRATION.md` |
| Authoring a new animation, animkit API, design rules | `docs/ANIMATION_GUIDE.md` |
| Beat tracking and synced playback | `docs/MUSIC_SYNC.md` |
| Known gaps, next ideas | `docs/ROADMAP.md` |
| Background and decisions | `docs/HISTORY.md` |
