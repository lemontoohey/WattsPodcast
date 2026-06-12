"""Pre-render a short looping sub-bass ambience clip for the web UI."""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bass_engine import BassEngine

SR = 22050
DURATION = 24.0
OUT_PATH = os.path.join(os.path.dirname(__file__), "static", "ambient.wav")

engine = BassEngine(sr=SR, duration=DURATION, master_level=0.10, glide_time=DURATION * 0.7)
audio = engine.generate_contemplate_layer()

# Crossfade the tail into the head so the clip loops seamlessly.
fade = int(SR * 2.0)
fade_curve = np.linspace(0.0, 1.0, fade)
audio[:fade] = audio[:fade] * fade_curve + audio[-fade:] * (1.0 - fade_curve)
audio = audio[:-fade]

sf.write(OUT_PATH, audio, SR)
print(f"Wrote {OUT_PATH} ({len(audio) / SR:.1f}s)")
