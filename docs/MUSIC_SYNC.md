# Music sync

Lock the projection to whatever music is playing in the room, using the laptop microphone.

## Using it

```bash
python projector_client.py "play $PWD/frames/hox_show 30"
python projector_client.py "sync on"        # opens the mic, starts tracking
python projector_client.py status           # ok sync=on bpm=119.9 confidence=4.4 level=0.8 lead=120ms
python projector_client.py "latency 150"    # draw-ahead in ms (projector + HDMI + decode lag)
python projector_client.py "sync off"
```

Give it ~5 s to lock. `confidence` is the comb-filter peak over its mean: > 2 is a lock,
> 4 is a strong one. `level` is the mic RMS (0–1); if it sits near 0 the mic is wrong or
muted.

Standalone test without the projector: `python beat.py` prints BPM/confidence once a
second for 24 s.

## How it works (`beat.py`)

1. **Onset envelope.** Audio in 1024-sample windows at 100 Hz hop. Log-magnitude spectrum;
   spectral flux = sum of positive bin differences, with bins below 250 Hz weighted ×2.5
   (kick and bass carry the beat). 12 s of envelope is kept.
2. **Tempo.** Slow swell removed (minus a 0.5 s moving average). Autocorrelation of the
   envelope. Score for a candidate lag L = acf[L] + acf[2L] + acf[4L] (each taking the max
   over ±1 sample), multiplied by a log-normal prior centred at 105 BPM (σ = 0.5 octave)
   to resolve half/double-time. Parabolic interpolation on the peak. A new tempo more than
   6 % away must beat the currently held tempo's score by 20 % and persist for 4 s (1 s
   before the first lock) before it is accepted; within 6 % the period is smoothed toward
   the measurement.
3. **Phase.** A comb of pulses at the current period is slid over the envelope (recency-
   weighted over ~8 beats); the best offset says when the last beat was heard.
4. **Clock.** `beats(at)` = anchor_beats + (at − anchor_time) / period. The phase measurement
   nudges `anchor_beats` by 8 % of the error per update (~20 Hz) when confident, 1.5 %
   otherwise — so it settles in about a second and ignores single bad frames. Period
   changes re-anchor so the count stays continuous.

`projector.py` in sync mode shows frame `int((beats(now + lead) − origin) · frames_per_beat) mod N`,
ticking at 120 Hz so it never misses a frame boundary. `origin` is set to the next integer
beat when playback starts, so the show begins on a beat.

## Why the harmonic scoring

On the afrobeat record used for testing, plain single-lag autocorrelation hopped between
83, 104 and 139 BPM — the lags of 3, 4 and 5 sixteenth-note groupings, all of which are
genuinely periodic in syncopated music. Only the true beat also repeats at 2× and 4× (the
bar), which is what `acf[L] + acf[2L] + acf[4L]` rewards. After the change the tracker held
119.9 ± 0.1 BPM for the whole test.

## Limitations / next steps

- No downbeat detection: scene changes in the show fall on beats but not necessarily on
  bar lines. A bar tracker (spectral-flux periodicity at 4× the beat with a phase estimate)
  would let scenes change on the "1".
- `latency` (default 120 ms) is a guess. Measure it: play a click track, photograph the
  projected orb thump with a high-fps phone video, adjust until the flash lands on the
  click.
- Loud rooms with the projector fan near the mic reduce confidence; an external mic or an
  audio-interface line-in (`BeatTracker(device="...")`) will be cleaner.
- Tempo prior is fixed at 105 BPM ± 0.5 octave. Fast electronic music (> 150) or slow
  ballads (< 70) fall outside `BPM_RANGE` — widen it and the prior if needed.
- The tracker runs in the pygame process. It is numpy-only on purpose; keep cv2 out.
