"""
Watts humour system.

Rips Alan Watts' laugh from a YouTube source, catalogues his humour
archetypes, and generates contextual jokes for [LAUGH:TYPE] markers.
"""

import os
import subprocess

import numpy as np
import soundfile as sf

from llm_client import complete


def rip_laugh(url, start_seconds=0):
    """
    Download source audio and extract a 10-second window for manual
    trimming. Laugh boundaries vary by source, so this is a starting
    point, not the final sample.

    Usage: python watts_podcast.py --rip-laugh URL [--laugh-start 45]
    """
    print("[Laugh] Downloading source audio...")
    subprocess.run([
        "yt-dlp", "-x", "--audio-format", "wav",
        "-o", "laugh_source.wav", url
    ], check=True)

    subprocess.run([
        "ffmpeg", "-y", "-i", "laugh_source.wav",
        "-ss", str(start_seconds), "-t", "10",
        "-ar", "22050", "-ac", "1",
        "laugh_raw_10s.wav"
    ], check=True)

    print("[Laugh] Saved laugh_raw_10s.wav")
    print("[Laugh] Open in Audacity, trim to the laugh only, save as watts_laugh.wav")
    print("[Laugh] Once saved, run: python watts_podcast.py --process-laugh")


def process_laugh():
    """
    Process the trimmed laugh sample:
      - normalise to -12 LUFS
      - subtle room reverb to match Watts' lecture acoustic
      - 80ms fade in/out to avoid clicks
    Saves watts_laugh_processed.wav.
    """
    filt = ','.join([
        'afade=t=in:d=0.08',
        'afade=t=out:st=1.4:d=0.08',
        'aecho=0.6:0.3:30:0.2',
        'loudnorm=I=-12:TP=-1:LRA=8'
    ])
    subprocess.run([
        'ffmpeg', '-y', '-i', 'watts_laugh.wav',
        '-af', filt,
        'watts_laugh_processed.wav'
    ], check=True)
    print('[Laugh] Processed laugh saved as watts_laugh_processed.wav')


# ---------------------------------------------------------------------------
# Humour taxonomy
# ---------------------------------------------------------------------------

HUMOUR_SYSTEM_PROMPT = """
You are generating a short humorous interjection in the voice of Alan Watts,
to be inserted into a technical monologue at a [LAUGH:TYPE] marker.

HUMOUR TYPE DEFINITIONS:

COSMIC_ABSURDITY:
  - Point out the absurdity of the universe (or a nanoparticle) taking itself seriously
  - Structure: cosmic fact -> the fact is aware of itself -> absurdity
  - Maximum 3 sentences
  - Must reference something specific from the preceding passage

SELF_DEPRECATING:
  - Mock the pretension of explaining complex things
  - 'Here I am, describing the behaviour of a sphere one ten-thousandth
    the width of a hair, as if I had the faintest idea what is actually happening'
  - 1-2 sentences only

ZEN_PUNCHLINE:
  - Setup follows the logic of the preceding passage
  - Punchline dissolves it in one line
  - Must be genuinely funny, not just paradoxical
  - The joke lands by making the complex suddenly obvious

BRITISH_DRY:
  - Deadpan understatement about something remarkable
  - Never exclamatory -- the humour is in the flatness
  - 1 sentence, maximum

AUDIENCE_MIRROR:
  - Voice the listener's inner confusion back at them, warmly
  - Must be affectionate, not condescending
  - 2-3 sentences maximum
  - End with a turn: their confusion is actually the right response

RULES FOR ALL TYPES:
  - Must reference something specific from the provided passage
  - Must sound exactly like Watts -- no modern idiom
  - Must be genuinely funny, not just clever
  - Must serve the learning: the laugh should make the concept
    MORE memorable, not interrupt it
  - No setup that requires prior knowledge of the joke form
  - Output only the joke text, nothing else
"""


def generate_contextual_joke(laugh_type, preceding_text, client):
    """Generate a contextual Watts-style joke for a [LAUGH:TYPE] marker."""
    text = complete(
        client,
        system=HUMOUR_SYSTEM_PROMPT,
        user=(
            f'HUMOUR TYPE: {laugh_type}\n\n'
            f'PRECEDING PASSAGE:\n{preceding_text[-400:]}\n\n'
            'Generate the joke now.'
        ),
        max_tokens=150,
    )
    return text.strip()


class WattsHumour:
    """Loads the processed laugh sample and generates laugh segments."""

    def __init__(self, client, tts, reference_wav, config):
        self.client = client
        self.tts = tts
        self.reference_wav = reference_wav
        self.config = config
        self.laugh_audio = None
        self.laugh_sr = None
        self._load_laugh()

    def _load_laugh(self):
        if os.path.exists('watts_laugh_processed.wav'):
            self.laugh_audio, self.laugh_sr = sf.read('watts_laugh_processed.wav')
            print('[Humour] Laugh sample loaded.')
        else:
            print('[Humour] watts_laugh_processed.wav not found.')
            print('[Humour] Run --rip-laugh then --process-laugh first.')
            print('[Humour] Continuing without laugh audio.')

    def get_laugh_audio(self, laugh_type, preceding_text):
        """
        Generate a contextual joke via Claude, synthesise it in Watts'
        voice, and append the laugh sample. Returns a mono numpy array.
        """
        joke_text = generate_contextual_joke(laugh_type, preceding_text, self.client)
        print(f'  [Humour:{laugh_type}] {joke_text[:60]}...')

        from watts_podcast import generate_chunk_with_fallback

        joke_path = f'joke_{laugh_type}.wav'
        generate_chunk_with_fallback(
            self.tts, joke_text, self.reference_wav, joke_path, self.config
        )
        joke_audio, sr = sf.read(joke_path)

        if self.laugh_audio is not None:
            silence = np.zeros(int(0.3 * sr))
            return np.concatenate([joke_audio, silence, self.laugh_audio])
        return joke_audio
