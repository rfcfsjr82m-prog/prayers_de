#!/usr/bin/env python3
"""Forced alignment of Prayer Stone audio to prayer text lines.

For each prayer/voice, transcribes the MP3 with faster-whisper (word
timestamps), aligns the recognized words to the known prayer text via
sequence matching, and emits per-line [start, end] timings — the data the
app's PrayerTimingAnalyzer currently computes on-device via SFSpeechRecognizer.

Output: timings_<lang>.json
  {"version": 1, "timings": {"<id>": {"m": [[s,e],...], "f": [[s,e],...]}}}

A per-file confidence report (matched-word ratio) is printed so low-quality
alignments can be spot-checked.
"""
import argparse
import difflib
import json
import os
import re
import sys
import time
import unicodedata

WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def normalize_word(w: str) -> str:
    w = unicodedata.normalize("NFKC", w).lower()
    w = w.replace("'", "'")
    return w.strip("'")


def text_words_by_line(text: str):
    """Returns (words, line_index_per_word, lines)."""
    lines = text.split("\n")
    words, line_of = [], []
    for i, line in enumerate(lines):
        for w in WORD_RE.findall(line):
            words.append(normalize_word(w))
            line_of.append(i)
    return words, line_of, lines


def align(rec_words, rec_times, txt_words, txt_line_of, n_lines, duration):
    """Map recognized words to text lines; return (timings, coverage_per_line)."""
    matcher = difflib.SequenceMatcher(a=txt_words, b=rec_words, autojunk=False)
    # For each text word index, the (start, end) of its matched recognized word.
    word_time = [None] * len(txt_words)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            word_time[block.a + k] = rec_times[block.b + k]

    line_words = [0] * n_lines
    line_matched = [0] * n_lines
    starts = [None] * n_lines
    ends = [None] * n_lines
    for wi, li in enumerate(txt_line_of):
        line_words[li] += 1
        t = word_time[wi]
        if t is None:
            continue
        line_matched[li] += 1
        if starts[li] is None:
            starts[li] = t[0]
        ends[li] = t[1]

    # Fill lines with no matched words (or empty lines) by interpolation:
    # start at the previous line's end, end at the next known start.
    timings = []
    prev_end = 0.0
    for i in range(n_lines):
        s = starts[i]
        e = ends[i]
        if s is None:
            s = prev_end
        s = max(s, prev_end)  # keep monotonic
        if e is None or e < s:
            nxt = next((starts[j] for j in range(i + 1, n_lines)
                        if starts[j] is not None and starts[j] > s), None)
            e = nxt if nxt is not None else min(s + 2.0, duration)
        timings.append([round(s, 2), round(e, 2)])
        prev_end = e

    coverage = [
        (line_matched[i] / line_words[i]) if line_words[i] else 1.0
        for i in range(n_lines)
    ]
    return timings, coverage


def transcribe(model, path, lang):
    segments, info = model.transcribe(
        path, language=lang, word_timestamps=True,
        beam_size=5, vad_filter=False, condition_on_previous_text=False,
    )
    words, times = [], []
    for seg in segments:
        for w in seg.words or []:
            token = normalize_word("".join(WORD_RE.findall(w.word)))
            if not token:
                continue
            words.append(token)
            times.append((w.start, w.end))
    return words, times, info.duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to prayers_de checkout")
    ap.add_argument("--lang", required=True, choices=["de", "en"])
    ap.add_argument("--ids", nargs="*", help="prayer ids (default: all with audio)")
    ap.add_argument("--voices", nargs="*", default=["m", "f"])
    ap.add_argument("--model", default="small")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    json_path = os.path.join(args.repo, f"prayers_{args.lang}.json")
    data = json.load(open(json_path))
    prayers = data["prayers"] if isinstance(data, dict) else data
    audio_dir = os.path.join(args.repo, "eng_audio" if args.lang == "en" else ".")

    out_path = args.out or f"timings_{args.lang}.json"
    result = {"version": 1, "timings": {}}
    if os.path.exists(out_path):  # resume support for the long batch run
        result = json.load(open(out_path))

    wanted = set(args.ids) if args.ids else None
    report = []
    for p in prayers:
        pid = str(p["id"])
        if wanted and pid not in wanted:
            continue
        txt_words, line_of, lines = text_words_by_line(p["text"])
        for voice in args.voices:
            if result["timings"].get(pid, {}).get(voice):
                continue  # already done (resume)
            path = os.path.join(audio_dir, f"{pid}_{voice}.mp3")
            if not os.path.exists(path):
                report.append((pid, voice, "MISSING", 0.0, 0.0))
                continue
            t0 = time.time()
            rec_words, rec_times, duration = transcribe(model, path, args.lang)
            timings, coverage = align(rec_words, rec_times, txt_words,
                                      line_of, len(lines), duration)
            nonempty = [c for c, ln in zip(coverage, lines) if ln.strip()]
            avg_cov = sum(nonempty) / len(nonempty) if nonempty else 0.0
            result["timings"].setdefault(pid, {})[voice] = timings
            report.append((pid, voice, "ok", avg_cov, time.time() - t0))
            json.dump(result, open(out_path, "w"), separators=(",", ":"))
            print(f"{pid}_{voice}: cov={avg_cov:.0%} dur={duration:.0f}s "
                  f"took={time.time()-t0:.0f}s", flush=True)

    print("\n== report ==")
    low = [r for r in report if r[2] == "ok" and r[3] < 0.7]
    missing = [r for r in report if r[2] == "MISSING"]
    done = [r for r in report if r[2] == "ok"]
    print(f"aligned: {len(done)}, low-confidence (<70%): {len(low)}, "
          f"missing files: {len(missing)}")
    for r in low:
        print(f"  LOW  {r[0]}_{r[1]} cov={r[3]:.0%}")
    for r in missing:
        print(f"  MISS {r[0]}_{r[1]}")


if __name__ == "__main__":
    main()
