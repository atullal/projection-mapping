# Roadmap and known gaps

Working today: automatic calibration, artwork registration, two albums with animations
(one single loop each, one eight-scene show), beat-synced playback. Everything below is
open.

## Robustness

- [ ] **Camera index auto-detection.** Probe every index, pick the one that sees the most
      of the projector's white pattern, instead of a hard-coded `1`.
- [ ] **Projector display auto-detection.** Choose the pygame display whose size is
      1920×1080 and is not the main display.
- [ ] **One-command recalibrate.** `recalibrate.py` that runs calibrate → mapping (auto seed
      from the previous mapping's centre) → register → re-render the active animation →
      resume playback.
- [ ] **Drift detection.** Periodically project a faint outline, photograph, and measure
      edge alignment; warn (or auto-recalibrate) when it drifts past ~3 px.
- [ ] **Sleep resilience.** After the Mac sleeps, the projector and Continuity Camera drop
      off. `projector.py` should notice its display vanished and exit cleanly.

## Pipeline

- [ ] **Coarser-then-finer stripes** or phase-shift patterns to get below the 1.63 px
      quantisation floor; currently the fit is at the floor, so real accuracy is limited
      by the 4 px stripe width.
- [ ] **Camera lens undistortion.** Not modelled; the cubic residual absorbs most of it,
      but wide-angle phone lenses would benefit from a one-time checkerboard calibration.
- [ ] **Non-planar objects.** The pipeline assumes a near-flat sleeve. The structured
      light already gives a dense map, so the animation-side could work directly in camera
      space for a plant pot or a bottle (as the original ChatGPT-era prototype did).
- [ ] **Linux/Windows.** `CAP_AVFOUNDATION` → platform default, `system_profiler` →
      something portable, pygame display indices verified.

## Animation

- [ ] **Generic "bring any cover to life" mode.** Without per-album regions: ink
      classification into N clusters, geodesic pulses from the artwork's densest point,
      own-colour glow base, edge-outline draw-in. Would give a decent first pass for any
      record in one command; hand-authored scenes on top.
- [ ] **Audio-reactive parameters**, not just tempo: onset strength → flash, low/high band
      energy → separate layer gains. `beat.py` already computes `flash` and `level`.
- [ ] **Downbeat / bar tracking** so scene changes land on the "1" (see MUSIC_SYNC.md).
- [ ] **Real-time rendering.** Frames are pre-rendered because `frame_light` is ~0.5 s per
      frame in numpy. A GPU port (shaders via moderngl, or Metal) would allow live
      parameter control and audio reactivity beyond tempo.
- [ ] **Scene editor / OSC control.** Trigger scenes, adjust gains and latency live from
      a phone.

## Housekeeping

- [ ] Move scripts into a package (`pm/`) with a CLI entry point; keep flat imports working.
- [ ] Measure and document the projector latency properly.
- [ ] Automated smoke test that runs decode + fit on a saved capture set (would need a
      small fixture of captured pattern photos, ~40 MB).
