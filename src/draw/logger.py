"""Terminal / JSON logging of detections, for use without a visual interface."""

import json
import sys
import time

import requests

YGOPRODECK_API = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

STATUS_TEXT = {
    'no_text_box': "no text box found (blurry, face-down or not a card?)",
    'orientation_unknown': "orientation could not be determined",
    'low_confidence': "confidence below threshold",
    'not_in_deck': "no candidate found in deck list",
}


class _Style:
    def __init__(self, enabled):
        self.enabled = enabled

    def _wrap(self, code, text):
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def green(self, t): return self._wrap("32", t)
    def yellow(self, t): return self._wrap("33", t)
    def red(self, t): return self._wrap("31", t)
    def dim(self, t): return self._wrap("2", t)
    def bold(self, t): return self._wrap("1", t)


def fetch_card_info(card_id, timeout=5):
    """Fetch card details from the YGOPRODeck API. Returns None on failure."""
    try:
        response = requests.get(YGOPRODECK_API, params={'id': card_id}, timeout=timeout)
        response.raise_for_status()
        return response.json()['data'][0]
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None


def format_card_info(info):
    lines = []
    kind = info.get('humanReadableCardType') or info.get('type', '')
    header = [kind]
    if info.get('attribute'):
        header.append(info['attribute'])
    if info.get('race'):
        header.append(info['race'])
    if info.get('level'):
        header.append(f"Level {info['level']}")
    if info.get('linkval'):
        header.append(f"Link-{info['linkval']}")
    lines.append(" | ".join(header))

    if 'atk' in info:
        stats = f"ATK {info['atk']}"
        if 'def' in info and info['def'] is not None:
            stats += f" / DEF {info['def']}"
        lines.append(stats)
    if info.get('archetype'):
        lines.append(f"Archetype: {info['archetype']}")
    if info.get('desc'):
        lines.append(info['desc'].replace('\r', '').replace('\n', ' '))
    return lines


class DetectionLogger:
    """Prints detection events to the terminal and optionally writes JSON lines.

    - default: one line the first time each card is recognized, plus a final summary
    - verbose: every detected card box on every frame, including why it was rejected
    - card_info: card details from YGOPRODeck the first time a card is recognized
    - json_path: one JSON object per frame with all detections
    """

    def __init__(self, verbose=False, card_info=False, json_path=None, vid_stride=1, stream=sys.stdout):
        self.verbose = verbose
        self.card_info = card_info
        self.vid_stride = vid_stride
        self.stream = stream
        self.style = _Style(stream.isatty())
        self.json_file = open(json_path, 'w', encoding='utf-8') if json_path else None

        self.start_time = time.time()
        self.frames = 0
        self.total_boxes = 0
        self.status_counts = {}
        self.seen = {}  # card_id -> {name, frames, best_score, first_frame}
        self._info_cache = {}

    def _print(self, text=""):
        print(text, file=self.stream, flush=True)

    def log_frame(self, frame_idx, outputs, elapsed):
        detections = outputs.get('detections', [])
        source_frame = frame_idx * self.vid_stride
        self.frames += 1
        self.total_boxes += len(detections)

        recognized = [d for d in detections if d['status'] == 'recognized']
        for d in detections:
            self.status_counts[d['status']] = self.status_counts.get(d['status'], 0) + 1

        if self.verbose:
            s = self.style
            self._print(s.bold(f"[frame {source_frame}]") +
                        f" {len(detections)} cards detected, {len(recognized)} recognized " +
                        s.dim(f"({elapsed:.2f}s)"))
            for d in detections:
                self._print("  " + self._format_detection(d))

        for d in recognized:
            entry = self.seen.get(d['card_id'])
            if entry is None:
                self.seen[d['card_id']] = {
                    'name': d['name'], 'frames': 1, 'best_score': d['score'], 'first_frame': source_frame,
                }
                self._announce_new_card(d, source_frame)
            else:
                entry['frames'] += 1
                entry['best_score'] = max(entry['best_score'], d['score'])

        if self.json_file:
            record = {
                'frame': source_frame,
                'time': round(time.time(), 3),
                'processing_seconds': round(elapsed, 3),
                'detections': detections,
            }
            self.json_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.json_file.flush()

    def _format_detection(self, d):
        s = self.style
        where = f"at ({d['center'][0]},{d['center'][1]})"
        if d['status'] == 'recognized':
            line = (s.green("✔ ") + s.bold(d['name']) + f" [{d['card_id']}] " +
                    f"{d['score']:.1%} {where}")
            alternatives = [c for c in d.get('candidates', []) if c['label'] != d['label']][:2]
            if alternatives:
                line += s.dim("  alt: " + ", ".join(f"{c['name']} {c['score']:.1%}" for c in alternatives))
            return line

        reason = STATUS_TEXT.get(d['status'], d['status'])
        line = s.red("✘ ") + f"card {where}: {reason}"
        if d.get('candidates'):
            best = d['candidates'][0]
            line += s.dim(f"  (best guess: {best['name']} {best['score']:.1%})")
        return line

    def _announce_new_card(self, d, source_frame):
        s = self.style
        self._print(s.green("NEW CARD") + f" frame {source_frame}: " + s.bold(d['name']) +
                    f" [{d['card_id']}] confidence {d['score']:.1%}")
        if not self.card_info:
            return
        if d['card_id'] not in self._info_cache:
            self._info_cache[d['card_id']] = fetch_card_info(d['card_id'])
        info = self._info_cache[d['card_id']]
        if info is None:
            self._print(s.dim("    (card details unavailable)"))
            return
        for line in format_card_info(info):
            self._print(s.dim("    " + line))

    def summary(self):
        s = self.style
        duration = time.time() - self.start_time
        self._print()
        self._print(s.bold("=== Summary ==="))
        fps = self.frames / duration if duration > 0 else 0
        self._print(f"Frames processed: {self.frames} in {duration:.1f}s ({fps:.2f} frames/s)")
        self._print(f"Card boxes detected: {self.total_boxes}")
        for status, count in sorted(self.status_counts.items(), key=lambda x: -x[1]):
            label = 'recognized' if status == 'recognized' else STATUS_TEXT.get(status, status)
            self._print(f"  {count:>6}  {label}")

        if not self.seen:
            self._print(s.yellow("No card recognized."))
        else:
            self._print(f"Unique cards recognized: {len(self.seen)}")
            for card_id, e in sorted(self.seen.items(), key=lambda x: -x[1]['frames']):
                self._print(f"  {e['name']} [{card_id}]  in {e['frames']} frame(s), "
                            f"best {e['best_score']:.1%}, first seen frame {e['first_frame']}")

    def close(self):
        if self.json_file:
            self.json_file.close()
            self.json_file = None
