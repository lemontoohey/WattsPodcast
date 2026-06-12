"""
Living Memory Thread.

Maintains a persistent JSON memory file across episodes:
  - injects a MEMORY THREAD context block into script generation,
    referencing previous episodes, open questions, and listener replies
  - extracts insights, open questions, and themes from each finished
    script and stores them for future episodes
"""

import datetime
import json
import os
import re

from llm_client import complete

MEMORY_FILE = 'watts_memory.json'

INSIGHT_EXTRACTION_PROMPT = """
You are the memory of an ongoing private lecture series in the style of
Alan Watts. You will be given the full script of the episode that was
just generated.

Extract, as JSON:
{
  "title": "the best chapter title or a fitting Watts-style episode title",
  "key_insights": [3-5 one-sentence insights, each capturing a conceptual
                   move the episode made -- not a summary of facts, but the
                   IDEA that would be worth recalling in a future lecture],
  "open_questions": [the contemplation questions the episode asked,
                     paraphrased to one sentence each],
  "themes": [1-3 single-word or two-word abstract themes,
             e.g. 'absence-as-mechanism', 'scale inversion', 'emergence']
}
Return ONLY the JSON object.
"""


class WattsMemory:
    def __init__(self, listener='Liam', path=MEMORY_FILE):
        self.path = path
        if os.path.exists(path):
            self.data = json.load(open(path, encoding='utf-8'))
        else:
            self.data = {
                'listener': listener,
                'episodes': [],
                'recurring_themes': [],
                'listener_threads': [],
            }

    def save(self):
        json.dump(self.data, open(self.path, 'w', encoding='utf-8'),
                  indent=2, ensure_ascii=False)

    # ---------- 1. Context injection (before script generation) ----------

    def memory_context(self, max_episodes=4, max_threads=3):
        """
        Build the MEMORY CONTEXT block appended to SYSTEM_PROMPT.
        Returns '' for the first episode (no memory yet).
        """
        eps = self.data['episodes'][-max_episodes:]
        if not eps:
            return ''

        lines = [
            '',
            'MEMORY THREAD -- this is episode %d of a continuing private'
            % (len(self.data['episodes']) + 1),
            'lecture series for one listener. Previous episodes:',
        ]
        for e in eps:
            lines.append(f"- Ep{e['id']} '{e['title']}' ({e['date']}): "
                          + '; '.join(e['key_insights'][:2]))

        open_qs = [q for e in eps for q in e.get('open_questions', [])][-3:]
        if open_qs:
            lines.append('Questions left open with the listener:')
            for q in open_qs:
                lines.append(f'- {q}')

        threads = self.data.get('listener_threads', [])[-max_threads:]
        if threads:
            lines.append('The listener wrote back (their own thinking -- '
                          'treat with respect, build on it):')
            for t in threads:
                lines.append(f"- Asked: {t['question']} | "
                              f"They answered: {t['listener_answer'][:200]}")

        if self.data['recurring_themes']:
            lines.append('Recurring themes across the series: '
                          + ', '.join(self.data['recurring_themes'][:6]))

        lines += [
            '',
            'CALLBACK RULES:',
            '- Weave AT MOST 2 natural callbacks to previous episodes into',
            '  the monologue. A callback sounds like: "You may remember,',
            '  when we spoke about X, I wondered aloud whether..." ',
            '- If the listener answered a previous question, acknowledge it',
            '  ONCE, warmly, and extend their idea further than they took it.',
            '- If a recurring theme reappears in the new material, name the',
            '  pattern explicitly -- the series noticing itself is the point.',
            '- Never recap. Callbacks are threads, not summaries.',
        ]
        return '\n'.join(lines)

    # ---------- 2. Insight extraction (after script generation) ----------

    def remember_episode(self, script, source_doc, client):
        """Extract and store insights from the finished script."""
        raw = complete(
            client,
            system=INSIGHT_EXTRACTION_PROMPT,
            user=script[:24000],
            max_tokens=600,
        )
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        meta = json.loads(m.group(0)) if m else {
            'title': source_doc, 'key_insights': [],
            'open_questions': [], 'themes': []
        }

        ep = {
            'id': len(self.data['episodes']) + 1,
            'date': datetime.date.today().isoformat(),
            'source_doc': source_doc,
            'title': meta.get('title', source_doc),
            'key_insights': meta.get('key_insights', []),
            'open_questions': meta.get('open_questions', []),
            'callbacks_used': [],
            'themes': meta.get('themes', []),
        }
        self.data['episodes'].append(ep)

        # Promote themes seen in >=2 episodes to recurring_themes
        for theme in ep['themes']:
            seen_before = any(theme in e.get('themes', [])
                               for e in self.data['episodes'][:-1])
            if seen_before and theme not in self.data['recurring_themes']:
                self.data['recurring_themes'].append(theme)

        self.save()
        print(f"[Memory] Episode {ep['id']} '{ep['title']}' remembered "
              f"({len(ep['key_insights'])} insights).")
        return ep

    # ---------- 3. Listener threads (written by Socratic Echo) ----------

    def add_listener_thread(self, episode_id, question, answer):
        self.data['listener_threads'].append({
            'episode_id': episode_id,
            'question': question,
            'listener_answer': answer,
            'source': 'journal',
        })
        self.save()

    def info(self):
        print(f"[Memory] {len(self.data['episodes'])} episodes, "
              f"{len(self.data['listener_threads'])} listener threads, "
              f"themes: {', '.join(self.data['recurring_themes']) or 'none yet'}")
