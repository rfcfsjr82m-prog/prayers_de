#!/usr/bin/env python3
"""Chunked-transcription retry for low-confidence clips: split at silence
gaps and transcribe each speech chunk independently so Whisper can't skip
repeated phrases. Patches timings_clips.json when coverage improves."""
import json
import sys

from align_prayers import align, text_words_by_line, WORD_RE, normalize_word
from faster_whisper import WhisperModel, decode_audio
from faster_whisper.vad import get_speech_timestamps, VadOptions

REPO = "prayers_de"
PATHS = sys.argv[1:]

m = WhisperModel("small", device="cpu", compute_type="int8")
manifest = {x["path"]: x for x in json.load(open("clips_manifest.json"))}
out = json.load(open("timings_clips.json"))

for path in PATHS:
    item = manifest[path]
    audio = decode_audio(f"{REPO}/{path}")
    sr = 16000
    chunks = get_speech_timestamps(
        audio, VadOptions(min_silence_duration_ms=250, speech_pad_ms=150))
    words, times = [], []
    for c in chunks:
        off = c["start"] / sr
        segs, _ = m.transcribe(audio[c["start"]:c["end"]], language=item["lang"],
                               word_timestamps=True,
                               condition_on_previous_text=False)
        for s in segs:
            for w in s.words or []:
                tok = normalize_word("".join(WORD_RE.findall(w.word)))
                if tok:
                    words.append(tok)
                    times.append((w.start + off, w.end + off))
    text = "\n".join(item["lines"])
    tw, lo, lines = text_words_by_line(text)
    timings, cov = align(words, times, tw, lo, len(lines), len(audio) / sr)
    ne = [c for c, l in zip(cov, lines) if l.strip()]
    avg = sum(ne) / len(ne) if ne else 0
    print(f"{path}: cov={avg:.0%}")
    if avg >= 0.7:
        out["clips"][path] = timings
        print("  -> patched")
    else:
        print("  -> STILL LOW; recognized text:")
        print("     " + " ".join(words)[:400])
        print("     EXPECTED: " + " ".join(tw)[:400])

json.dump(out, open("timings_clips.json", "w"),
          ensure_ascii=False, separators=(",", ":"))
