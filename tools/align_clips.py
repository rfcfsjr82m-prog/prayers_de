#!/usr/bin/env python3
"""Force-aligns guided-prayer and rosary clips from clips_manifest.json.

Output timings_clips.json: {"version": 1,
  "clips": {"<repo-relative path>": [[start, end], ...]}}   # one pair per line
"""
import argparse
import json
import os
import time

from align_prayers import align, transcribe, text_words_by_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--manifest", default="clips_manifest.json")
    ap.add_argument("--model", default="small")
    ap.add_argument("--out", default="timings_clips.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="substring filter on path")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    manifest = json.load(open(args.manifest))
    result = {"version": 1, "clips": {}}
    if os.path.exists(args.out):  # resume
        result = json.load(open(args.out))

    done = 0
    low, missing = [], []
    for item in manifest:
        path = item["path"]
        if args.only and args.only not in path:
            continue
        if path in result["clips"]:
            continue
        if args.limit and done >= args.limit:
            break
        full = os.path.join(args.repo, path)
        if not os.path.exists(full):
            missing.append(path)
            continue
        text = "\n".join(item["lines"])
        txt_words, line_of, lines = text_words_by_line(text)
        t0 = time.time()
        rec_words, rec_times, duration = transcribe(model, full, item["lang"])
        timings, coverage = align(rec_words, rec_times, txt_words,
                                  line_of, len(lines), duration)
        nonempty = [c for c, ln in zip(coverage, lines) if ln.strip()]
        avg = sum(nonempty) / len(nonempty) if nonempty else 0.0
        result["clips"][path] = timings
        done += 1
        if avg < 0.7:
            low.append((path, avg))
        if done % 25 == 0:
            json.dump(result, open(args.out, "w"),
                      ensure_ascii=False, separators=(",", ":"))
        print(f"{path}: cov={avg:.0%} dur={duration:.0f}s "
              f"took={time.time()-t0:.1f}s", flush=True)

    json.dump(result, open(args.out, "w"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"\n== report == aligned {done}, low-confidence {len(low)}, "
          f"missing {len(missing)}")
    for p, c in low:
        print(f"  LOW  {p} cov={c:.0%}")
    for p in missing:
        print(f"  MISS {p}")


if __name__ == "__main__":
    main()
