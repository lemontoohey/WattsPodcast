"""
Socratic Echo + Journal.

Provides:
  1. answer_space() -- a held thinking-space after each contemplation
     question, rendered as a low sub-drone tail.
  2. ContemplationJournal -- collects each contemplation question with
     its audio timestamp and writes journal_ep<N>.md.
  3. --reply mode -- parses a filled-in journal and generates a short
     follow-up episode where Watts responds to the listener's answers.
"""

import datetime
import re

import numpy as np

from llm_client import complete


# ---------------------------------------------------------------------------
# 1. Answer space
# ---------------------------------------------------------------------------

def answer_space(bass_engine_cls, seconds=20.0, sr=22050, intensity=0.5):
    """
    A held thinking-space after each contemplation question.
    Not dead silence -- a low Layer-3-only drone tail at a fraction of
    contemplate level keeps the space 'alive'. Ends with 1.5s of true
    silence before the voice returns.
    """
    if seconds <= 0:
        return np.zeros(0)

    drone_seconds = max(4.0, seconds - 1.5)
    eng = bass_engine_cls(sr=sr, duration=drone_seconds,
                           master_level=0.12 + 0.08 * intensity)
    drone, _extra = eng._layer3(int(drone_seconds * sr))
    drone = drone[:int(drone_seconds * sr)] * eng.master_level
    return np.concatenate([drone, np.zeros(int(1.5 * sr))])


# ---------------------------------------------------------------------------
# 2. Companion journal
# ---------------------------------------------------------------------------

class ContemplationJournal:
    """
    Collects each contemplation question during rendering, with running
    audio timestamps, then writes journal_ep<N>.md next to the output mp3.
    """

    def __init__(self, episode_id, episode_title, source_doc):
        self.episode_id = episode_id
        self.episode_title = episode_title
        self.source_doc = source_doc
        self.entries = []  # {'t_seconds': float, 'question': str}

    def log_question(self, t_seconds, question):
        self.entries.append({'t_seconds': t_seconds, 'question': question})

    @staticmethod
    def _ts(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f'{h:02d}:{m:02d}:{s:02d}'

    def write(self, path=None):
        path = path or f'journal_ep{self.episode_id}.md'
        lines = [
            f'# Contemplation Journal -- Episode {self.episode_id}',
            f'**{self.episode_title}**',
            f'Source: {self.source_doc} | {datetime.date.today().isoformat()}',
            '',
            '> Answer in plain text under each question, then run:',
            f'> `python watts_podcast.py --reply {path}`',
            '',
        ]
        for i, e in enumerate(self.entries, 1):
            lines += [
                f'## Question {i}  ·  [{self._ts(e["t_seconds"])}]',
                '',
                f'> {e["question"]}',
                '',
                'YOUR ANSWER:',
                '',
                '...',
                '',
                '---',
                '',
            ]
        open(path, 'w', encoding='utf-8').write('\n'.join(lines))
        print(f'[Journal] {path} written ({len(self.entries)} questions).')
        return path


# ---------------------------------------------------------------------------
# 3. Reply mode -- Watts responds to the listener's answers
# ---------------------------------------------------------------------------

REPLY_SYSTEM_PROMPT = """
You are Alan Watts, continuing a private dialogue with one listener.
After your last lecture you asked them creative questions; they wrote
their answers in a journal. You will receive question/answer pairs.

Compose a SHORT spoken reply monologue (5-8 minutes, ~800-1100 words):

- Open mid-thought, as if picking up a conversation: 'So. You wrote
  back. Good -- most people only listen...'
- For EACH answered question, in order:
  1. Restate THEIR idea better than they said it (the steelman)
  2. Find the part of their answer that is genuinely original and
     say precisely why it is
  3. Then push: one way their idea breaks, or one place it goes
     further than they dared take it
  4. Leave them one NEW question that only exists because of their
     answer -- the dialogue must spiral outward, never close
- Unanswered questions ('...'): mention at most one, lightly, without
  guilt: 'The second question is still sitting there. Let it sit.'
- Tone: warm, amused, never flattering. Treat the listener as a
  fellow philosopher, not a student.
- Use the same marker system: [BREATH], [EMPHASIS]...[/EMPHASIS],
  at most ONE [CONTEMPLATE], at most ONE [LAUGH:TYPE].
- No headers, no bullets. Flowing prose only.
"""


def parse_journal(journal_path):
    """Parse question/answer pairs from a filled journal_ep<N>.md."""
    text = open(journal_path, encoding='utf-8').read()
    episode_id = int(re.search(r'Episode (\d+)', text).group(1))

    pairs = []
    blocks = re.split(r'^## Question \d+.*$', text, flags=re.MULTILINE)[1:]
    for b in blocks:
        qm = re.search(r'^> (.+?)$', b, re.MULTILINE)
        am = re.search(r'YOUR ANSWER:\s*\n(.*?)(?:\n---|\Z)', b, re.DOTALL)
        if not qm:
            continue
        answer = am.group(1).strip() if am else ''
        answered = answer not in ('', '...')
        pairs.append({
            'question': qm.group(1).strip(),
            'answer': answer if answered else None,
        })
    return episode_id, pairs


def generate_reply_episode(journal_path, client, memory=None):
    """
    Build the reply script. Rendering reuses the normal pipeline:
    parse_script_markers -> generate_audio -> master_audio.
    Returns (script_path, episode_id).
    """
    episode_id, pairs = parse_journal(journal_path)
    answered = [p for p in pairs if p['answer']]
    if not answered:
        print('[Reply] No answers found in journal -- nothing to reply to.')
        return None, episode_id

    if memory is not None:
        for p in answered:
            memory.add_listener_thread(episode_id, p['question'], p['answer'])

    content = '\n\n'.join(
        f"QUESTION: {p['question']}\nANSWER: {p['answer'] or '...'}"
        for p in pairs
    )
    script = complete(
        client,
        system=REPLY_SYSTEM_PROMPT,
        user=content,
        max_tokens=4000,
    ).strip()

    path = f'watts_reply_ep{episode_id}.txt'
    open(path, 'w', encoding='utf-8').write(script)
    print(f'[Reply] Reply script saved: {path}')
    return path, episode_id
