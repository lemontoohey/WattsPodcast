"""
Adaptive Arc Director.

Maps a script's emotional arc to per-zone render parameters: each zone
has a mood, intensity (0-1), and pacing factor. generate_audio() queries
ArcDirector for the current zone (by running word position) to drive the
bass engine, breath duration, emphasis strength, and dense-passage pauses
dynamically along the arc.

Falls back to fixed v2 defaults if no arc is available (--no-arc).
"""

import json
import re

import numpy as np

from llm_client import complete

ARC_SYSTEM_PROMPT = """
You are a sound-design director analysing a spoken-word script
(an Alan Watts style monologue) for dynamic audio rendering.

Divide the script into 4-8 sequential ZONES and return JSON:

<arc>
[
  {
    "start_word": 0,
    "mood": "wonder",
    "intensity": 0.3,
    "pacing": 1.0
  }
]
</arc>

Fields:
- "mood": one of wonder, tension, play, depth, release
- "intensity": 0.0-1.0, how deep/heavy this zone should feel
- "pacing": 0.85 (slower, spacious) to 1.1 (brisker)

RULES:
- intensity must form an ARC: start low-mid, build to a single peak
  in the final third (the deepest [CONTEMPLATE] should fall inside it),
  then release.
- 'play' zones are where [LAUGH] tokens live -- keep intensity <= 0.5 there.
- 'depth' zones contain [CONTEMPLATE] tokens -- intensity >= 0.6.
- start_word values must be ascending and within the script length.
Return ONLY the <arc> block.
"""


class ArcDirector:
    """Maps script word-position -> render parameters."""

    DEFAULTS = {'mood': 'depth', 'intensity': 0.5, 'pacing': 1.0}

    def __init__(self, script=None, client=None, enabled=True):
        self.zones = []
        self.enabled = enabled
        if enabled and script and client:
            self._build(script, client)

    def _build(self, script, client):
        raw = complete(
            client,
            system=ARC_SYSTEM_PROMPT,
            user=script[:24000],
            max_tokens=800,
        )
        m = re.search(r'<arc>(.*?)</arc>', raw, re.DOTALL)
        if m:
            try:
                self.zones = sorted(json.loads(m.group(1)),
                                     key=lambda z: z['start_word'])
                print(f'[Arc] {len(self.zones)} zones: ' + ' -> '.join(
                    f"{z['mood']}({z['intensity']:.1f})" for z in self.zones))
                json.dump(self.zones, open('arc_map.json', 'w'), indent=2)
            except (json.JSONDecodeError, KeyError):
                print('[Arc] Parse failed -- using fixed defaults.')
        else:
            print('[Arc] No arc returned -- using fixed defaults.')

    def zone_at(self, word_position):
        if not self.zones:
            return dict(self.DEFAULTS)
        current = self.zones[0]
        for z in self.zones:
            if z['start_word'] <= word_position:
                current = z
            else:
                break
        return current

    # ---------- Parameter mappings ----------

    def bass_params(self, word_position):
        """
        Map zone intensity -> BassEngine constructor params.
        intensity 0.0 -> short, light, fast-resolving bed
        intensity 1.0 -> long, deep, slow, loud bed
        """
        z = self.zone_at(word_position)
        i = float(np.clip(z['intensity'], 0.0, 1.0))
        return {
            'duration': 9.0 + 9.0 * i,        # 9s .. 18s
            'master_level': 0.25 + 0.20 * i,  # 0.25 .. 0.45
            'glide_time': 5.0 + 6.0 * i,      # engine decelerates slower when deep
            'layer3_gain': 0.6 + 0.4 * i,     # sub drone swells with intensity
        }

    def breath_duration(self, word_position):
        """Breaths stretch in slow zones, tighten in brisk zones."""
        z = self.zone_at(word_position)
        return 0.8 / float(z.get('pacing', 1.0))

    def emphasis_strength(self, word_position):
        """Emphasis pitch shift scales with intensity (0.2 .. 0.45 semitones)."""
        z = self.zone_at(word_position)
        return 0.2 + 0.25 * float(z['intensity'])

    def density_pause(self, word_position):
        """Inter-sentence pause for dense passages (~0.25s .. 0.45s)."""
        z = self.zone_at(word_position)
        return 0.3 / float(z.get('pacing', 1.0)) + 0.1 * float(z['intensity'])
