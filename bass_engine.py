"""
Sub-bass contemplation engine.

Three staggered harmonic layers for [CONTEMPLATE] moments:
  - Layer 1: hydrogen-rotation harmonic series (17.6 Hz fundamental), left-biased
  - Layer 2: marine-diesel deceleration with cavitation texture, right-biased
  - Layer 3: 28 Hz sub drone, centred, longest release

All layers are hard lowpassed at 400 Hz and mixed to a single mono array
(the pipeline combines to stereo at the mastering stage).
"""

import numpy as np
from scipy.signal import butter, sosfilt


def pink_noise(n_samples, sr=22050):
    """Voss-McCartney pink noise, normalised to [-1, 1]."""
    max_key = 0x1f
    key = 0
    white_values = np.random.randn(5)
    out = np.zeros(n_samples)
    for i in range(n_samples):
        last_key = key
        key = (key + 1) & max_key
        diff = last_key ^ key
        for j in range(5):
            if diff & (1 << j):
                white_values[j] = np.random.randn()
        out[i] = white_values.sum()
    peak = np.max(np.abs(out))
    return out / peak if peak > 0 else out


class BassEngine:
    LAYER1_HARMONICS = [17.6, 35.2, 52.8, 70.4, 88.0, 105.6, 123.2]
    LAYER1_AMPLITUDES = [1 / n for n in range(1, 8)]
    LAYER1_AM_RATE = 0.07

    LAYER2_GLIDE_START = 7.0
    LAYER2_GLIDE_END = 3.5
    LAYER2_HARMONICS_MULT = [1, 2, 4, 8, 16, 32, 64]
    LAYER2_CAVITATION_BAND = (120, 200)
    LAYER2_CAVITATION_LEVEL = 0.20
    LAYER2_FM_RATE = 0.15
    LAYER2_FM_DEPTH = 0.08
    LAYER2_ENTRY_DELAY = 2.5

    LAYER3_PARTIALS = [(28.0, 1.0), (56.0, 0.4), (84.0, 0.2)]
    LAYER3_AM_RATE = 0.033
    LAYER3_ENTRY_DELAY = 5.0
    LAYER3_EXTRA_RELEASE = 3.0

    def __init__(self, sr=22050, duration=12.0, master_level=0.35,
                 glide_time=8.0, layer3_gain=1.0):
        """
        sr: sample rate (matches xtts_v2 output)
        duration: base contemplate duration (seconds)
        master_level: overall output level (sits under speech)
        glide_time: seconds for Layer 2's engine deceleration glide
        layer3_gain: scales the centre sub drone (Layer 3)
        """
        self.sr = sr
        self.duration = duration
        self.master_level = master_level
        self.glide_time = glide_time
        self.layer3_gain = layer3_gain

    def _envelope(self, n, attack_s, release_s=2.0):
        """Raised-cosine attack + linear release envelope."""
        env = np.ones(n)
        atk = int(attack_s * self.sr)
        rel = int(release_s * self.sr)
        if atk > 0:
            atk = min(atk, n)
            env[:atk] = 0.5 * (1 - np.cos(np.pi * np.arange(atk) / atk))
        if 0 < rel < n:
            env[-rel:] = np.linspace(1, 0, rel)
        return env

    def _lowpass(self, audio):
        """8th-order Butterworth lowpass at 400 Hz."""
        sos = butter(8, 400 / (self.sr / 2), btype='low', output='sos')
        return sosfilt(sos, audio)

    def _normalize(self, signal):
        peak = np.max(np.abs(signal))
        return signal / peak if peak > 1e-9 else signal

    def _layer1(self, n_total):
        """Hydrogen rotation harmonic series, left-biased."""
        t = np.arange(n_total) / self.sr
        signal = np.zeros(n_total)

        pn = pink_noise(n_total, self.sr)
        am_mod = 1.0 + 0.15 * pn
        slow_am = 0.85 + 0.15 * np.sin(2 * np.pi * self.LAYER1_AM_RATE * t)

        for freq, amp in zip(self.LAYER1_HARMONICS, self.LAYER1_AMPLITUDES):
            partial = amp * np.sin(2 * np.pi * freq * t)
            signal += partial * am_mod * slow_am

        signal = self._lowpass(signal)
        signal *= self._envelope(n_total, attack_s=4.0)
        return self._normalize(signal)

    def _layer2(self, n_total):
        """Marine diesel deceleration, right-biased, delayed entry."""
        delay_samples = int(self.LAYER2_ENTRY_DELAY * self.sr)
        t = np.arange(n_total) / self.sr
        signal = np.zeros(n_total)

        glide = np.where(
            t < self.glide_time,
            self.LAYER2_GLIDE_START - (self.LAYER2_GLIDE_START - self.LAYER2_GLIDE_END) * (t / self.glide_time),
            self.LAYER2_GLIDE_END
        )
        phase = 2 * np.pi * np.cumsum(glide) / self.sr

        for i, m in enumerate(self.LAYER2_HARMONICS_MULT):
            if m * self.LAYER2_GLIDE_END < 400:
                signal += (1 / (i + 1)) * np.sin(phase * m)

        fm = 1.0 + self.LAYER2_FM_DEPTH * np.sin(2 * np.pi * self.LAYER2_FM_RATE * t)
        signal *= fm

        noise = np.random.randn(n_total)
        lo, hi = self.LAYER2_CAVITATION_BAND
        cavitation = sosfilt(
            butter(4, [lo / (self.sr / 2), hi / (self.sr / 2)], btype='band', output='sos'),
            noise
        )
        signal += self.LAYER2_CAVITATION_LEVEL * cavitation

        signal = self._lowpass(signal)
        if delay_samples > 0:
            signal[:min(delay_samples, n_total)] = 0
        signal *= self._envelope(n_total, attack_s=3.0)
        return self._normalize(signal)

    def _layer3(self, n_total):
        """Sub drone, centred, delayed entry, extended release."""
        delay_samples = int(self.LAYER3_ENTRY_DELAY * self.sr)
        extra = int(self.LAYER3_EXTRA_RELEASE * self.sr)
        n_extended = n_total + extra
        t = np.arange(n_extended) / self.sr
        signal = np.zeros(n_extended)

        slow_am = 0.9 + 0.1 * np.sin(2 * np.pi * self.LAYER3_AM_RATE * t)
        for freq, amp in self.LAYER3_PARTIALS:
            signal += amp * np.sin(2 * np.pi * freq * t)
        signal *= slow_am

        signal = self._lowpass(signal)
        if delay_samples > 0:
            signal[:min(delay_samples, n_extended)] = 0
        signal *= self._envelope(n_extended, attack_s=6.0, release_s=8.0)
        return self._normalize(signal), extra

    def generate_contemplate_layer(self):
        """
        Generate the full three-layer contemplate texture as a mono
        numpy array, mixed at self.master_level.
        """
        n = int(self.duration * self.sr)
        l1 = self._layer1(n)
        l2 = self._layer2(n)
        l3_full, _extra = self._layer3(n)
        l3 = l3_full[:n]

        mixed = (0.75 * l1) + (0.75 * l2) + (self.layer3_gain * l3)
        mixed *= self.master_level

        peak = np.max(np.abs(mixed))
        if peak > 0.9:
            mixed *= 0.9 / peak
        return mixed
