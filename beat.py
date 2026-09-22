"""Live beat tracking from the microphone (numpy + sounddevice only, safe next to pygame).

Onset envelope (spectral flux, 100 Hz) -> tempo by autocorrelation with a prior around
105 BPM -> beat phase by comb filter -> a phase-locked beat clock.  The clock is what the
player reads: it free-runs between updates and predicts ahead, so frames can be chosen for
where the music *will be* once light actually leaves the projector.

    python beat.py            # listen for a while and print what it hears
"""
import threading
import time

import numpy as np
import sounddevice as sd

ENV_RATE = 100                      # onset-envelope samples per second
WINDOW = 1024
BPM_RANGE, BPM_CENTRE = (68, 165), 105
HISTORY = 12.0                      # seconds of envelope kept


class BeatTracker:
    def __init__(self, device="MacBook Pro Microphone"):
        info = sd.query_devices(device, "input")
        self.rate = int(info["default_samplerate"])
        self.hop = self.rate // ENV_RATE
        self.window = np.hanning(WINDOW).astype(np.float32)
        low = np.fft.rfftfreq(WINDOW, 1 / self.rate) < 250
        self.weights = np.where(low, 2.5, 1.0).astype(np.float32)   # kicks and bass carry the beat
        self.lock = threading.Lock()
        self.pending = np.zeros(0, np.float32)
        self.pending_end = time.monotonic()
        self.tail = np.zeros(WINDOW - self.hop, np.float32)
        self.previous = None
        self.env = np.zeros(int(HISTORY * ENV_RATE), np.float32)
        self.env_end = time.monotonic()         # wall time of the newest envelope sample
        self.heard = 0.0                        # seconds of audio analysed

        self.period = 60.0 / BPM_CENTRE         # seconds per beat
        self.anchor_time, self.anchor_beats = time.monotonic(), 0.0
        self.candidate, self.candidate_since, self.locked = None, 0.0, False
        self.confidence, self.level, self.flash = 0.0, 0.0, 0.0
        self.running = True
        self.stream = sd.InputStream(device=device, channels=1, samplerate=self.rate, blocksize=0,
                                     dtype="float32", callback=self._on_audio)
        self.stream.start()
        threading.Thread(target=self._analyse, daemon=True).start()

    # -- the clock the player reads ---------------------------------------------------------
    def beats(self, at=None):
        """Continuous beat count at wall time `at` (defaults to now); integers fall on beats."""
        at = time.monotonic() if at is None else at
        with self.lock:
            return self.anchor_beats + (at - self.anchor_time) / self.period

    @property
    def bpm(self):
        return 60.0 / self.period

    # -- audio in ---------------------------------------------------------------------------
    def _on_audio(self, block, frames, info, status):
        with self.lock:
            self.pending = np.concatenate([self.pending, block[:, 0]])
            self.pending_end = time.monotonic()

    def _envelope(self, samples):
        """Append spectral-flux samples for the new audio; returns how many were added."""
        audio = np.concatenate([self.tail, samples])
        count = (len(audio) - WINDOW) // self.hop + 1
        if count <= 0:
            self.tail = audio
            return 0
        index = np.arange(WINDOW)[None] + self.hop * np.arange(count)[:, None]
        spectrum = np.log1p(40 * np.abs(np.fft.rfft(audio[index] * self.window, axis=1))).astype(np.float32)
        stacked = spectrum if self.previous is None else np.vstack([self.previous[None], spectrum])
        flux = (np.maximum(np.diff(stacked, axis=0), 0) * self.weights).sum(1)
        if self.previous is None:
            flux = np.concatenate([[0], flux])
        self.previous = spectrum[-1]
        self.tail = audio[count * self.hop:]
        self.env = np.concatenate([self.env, flux.astype(np.float32)])[-len(self.env):]
        return count

    # -- analysis ---------------------------------------------------------------------------
    def _analyse(self):
        while self.running:
            time.sleep(0.05)
            with self.lock:
                samples, end = self.pending, self.pending_end
                self.pending = np.zeros(0, np.float32)
            if not len(samples):
                continue
            rms = float(np.sqrt(np.mean(samples ** 2)))
            self.level += 0.2 * (min(1.0, rms * 12) - self.level)
            added = self._envelope(samples)
            self.env_end = end - len(self.tail) / self.rate
            self.heard += len(samples) / self.rate
            if added:
                recent = self.env[-added:].max()
                typical = np.percentile(self.env[-300:], 90) + 1e-6
                self.flash = max(self.flash * 0.8, min(1.0, recent / (2.0 * typical)))
            if self.heard > 3.0:
                self._update_clock()

    def _update_clock(self):
        env = self.env[-int(min(self.heard, HISTORY) * ENV_RATE):]
        env = np.maximum(env - np.convolve(env, np.ones(50) / 50, "same"), 0)   # remove slow swell
        if env.max() < 1e-4:
            self.confidence = 0.0
            return
        # tempo: autocorrelation, harmonics reinforce, log-normal prior resolves octave errors
        spectrum = np.fft.rfft(env, 2 * len(env))
        acf = np.fft.irfft(np.abs(spectrum) ** 2)[:len(env)]
        acf /= acf[0] + 1e-9
        lags = np.arange(int(ENV_RATE * 60 / BPM_RANGE[1]), int(ENV_RATE * 60 / BPM_RANGE[0]) + 1)
        usable = lags[4 * lags + 2 < len(acf)]
        if not len(usable):
            return

        def around(centre):                                            # tolerate a sample of drift at long lags
            return np.maximum.reduce([acf[centre - 1], acf[centre], acf[centre + 1]])

        # A real beat also repeats at 2x and 4x (the bar).  Syncopated groupings of 3 or 5
        # sixteenths correlate at their own lag but not at those multiples, so this is what
        # keeps afrobeat / funk from dragging the tempo to 3/4 or 5/4 of the truth.
        score = acf[usable] + around(2 * usable) + around(4 * usable)
        score = score * np.exp(-0.5 * (np.log2(ENV_RATE * 60 / usable / BPM_CENTRE) / 0.5) ** 2)
        best = int(np.argmax(score))
        lag = float(usable[best])
        if 0 < best < len(score) - 1:                                  # parabolic refinement
            a, b, c = score[best - 1:best + 2]
            lag += 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
        measured = lag / ENV_RATE
        held = score[int(np.argmin(np.abs(usable - self.period * ENV_RATE)))]
        now = time.monotonic()
        if abs(measured - self.period) / self.period < 0.06:
            self._set_period(self.period + 0.15 * (measured - self.period), now)
            self.candidate, self.locked = None, self.locked or self.heard > 6.0
        elif self.locked and score[best] < 1.2 * held:
            self.candidate = None                                      # a rival has to win clearly
        elif self.candidate is None or abs(measured - self.candidate) / self.candidate > 0.06:
            self.candidate, self.candidate_since = measured, now
        elif now - self.candidate_since > (4.0 if self.locked else 1.0):
            self._set_period(measured, now)                            # ...and keep winning for a while
            self.candidate, self.locked = None, True

        # phase: slide a pulse train at the current period over the envelope
        period = self.period * ENV_RATE
        offsets = np.arange(int(period))
        beats_back = np.arange(int(len(env) / period) - 1)
        positions = len(env) - 1 - offsets[:, None] - np.round(beats_back[None] * period).astype(int)
        recency = np.exp(-beats_back / 8.0)
        comb = (env[np.clip(positions, 0, len(env) - 1)] * recency).sum(1)
        comb = np.convolve(np.concatenate([comb[-2:], comb, comb[:2]]), [0.15, 0.2, 0.3, 0.2, 0.15], "valid")
        offset = int(np.argmax(comb))
        self.confidence = float(comb[offset] / (comb.mean() + 1e-9))
        beat_time = self.env_end - offset / ENV_RATE                   # a beat was heard at this wall time
        error = self.beats(beat_time)
        error -= np.round(error)                                       # how far the clock is from an integer there
        gain = 0.08 if self.confidence > 1.6 else 0.015   # runs at ~20 Hz: settles in about a second
        with self.lock:
            self.anchor_beats -= gain * error

    def _set_period(self, period, now):
        with self.lock:                                                # re-anchor so the count stays continuous
            self.anchor_beats += (now - self.anchor_time) / self.period
            self.anchor_time, self.period = now, period

    def close(self):
        self.running = False
        self.stream.stop()
        self.stream.close()


if __name__ == "__main__":
    tracker = BeatTracker()
    print("listening on MacBook Pro Microphone...", flush=True)
    start, last_beat = time.monotonic(), 0
    while time.monotonic() - start < 24:
        time.sleep(1.0)
        print(f"  t={time.monotonic() - start:4.1f}s  level={tracker.level:.2f}  bpm={tracker.bpm:6.1f}"
              f"  confidence={tracker.confidence:.2f}  beats={tracker.beats():6.1f}", flush=True)
    tracker.close()
