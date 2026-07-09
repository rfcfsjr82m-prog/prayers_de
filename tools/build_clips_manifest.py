#!/usr/bin/env python3
"""Builds a manifest of every guided-prayer and rosary audio clip with the text
lines the app displays for it, so align_clips.py can force-align each clip.

Sources:
- Guided steps: parsed out of PrayerLibrary.swift's `bundledSteps` literal.
- Rosary steps: the bundled rosary_*.json files in the app repo; text is split
  into phrases exactly like RosaryView.phrases().

Output manifest.json: [{"path": "acts/183_m_1.mp3", "lang": "de",
                        "lines": ["...", ...]}, ...]
`path` is repo-relative in prayers_de — the key the app will use to look up
shipped timings.
"""
import json
import os
import re
import sys

APP = "/Users/christiankasper/Documents/Cowork/Code/PrayerApp/PrayerApp"
REPO = sys.argv[1] if len(sys.argv) > 1 else "prayers_de"

# Mirrors PrayerLibrary.clipPathTemplates / clipStartNumbers.
CLIP_TEMPLATES = {
    "183": ("acts/183_{voice}_{n}", None),
    "184": ("TAR/184{n}_{voice}", None),
    "185": ("ALTAR/185{n}_{voice}", None),
    "200": ("acts_en/{f}_{voice}", 2001),
    "201": ("TAR_en/{f}_{voice}", 2007),
    "202": ("altar_en/{f}_{voice}", 2012),
    "203": ("quiet_strength_en/{f}_{voice}", 2019),
}
GUIDED_LANG = {"183": "de", "184": "de", "185": "de",
               "200": "en", "201": "en", "202": "en", "203": "en"}


def unescape_swift(s: str) -> str:
    return s.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')


def parse_bundled_steps(swift_path):
    """Extract {prayer_id: [step_text, ...]} from the bundledSteps literal."""
    src = open(swift_path).read()
    start = src.index("static let bundledSteps")
    end = src.index("static let clipPathTemplates")
    block = src[start:end]
    steps_by_id = {}
    current_id = None
    # Walk the block: prayer-id keys look like `"183": [`, steps are
    # PrayerStep(title: "...", text: "...", ...)
    token_re = re.compile(
        r'"(\d+)":\s*\[|text:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
    for m in token_re.finditer(block):
        if m.group(1) is not None:
            current_id = m.group(1)
            steps_by_id[current_id] = []
        else:
            steps_by_id[current_id].append(unescape_swift(m.group(2)))
    return steps_by_id


def rosary_phrases(text):
    """Replicates RosaryView.phrases(): hard split on .!? , soft on ,;: after
    38 chars."""
    hard, soft = set(".!?"), set(",;:")
    result, current = [], ""
    for ch in text:
        current += ch
        trimmed = current.strip()
        if ch in hard:
            if trimmed:
                result.append(trimmed)
            current = ""
        elif ch in soft and len(trimmed) >= 38:
            result.append(trimmed)
            current = ""
    tail = current.strip()
    if tail:
        result.append(tail)
    return result or [text]


def main():
    manifest = []
    seen = set()

    # --- Guided prayers ---
    steps_by_id = parse_bundled_steps(os.path.join(APP, "Models/PrayerLibrary.swift"))
    for pid, (template, start_num) in CLIP_TEMPLATES.items():
        texts = steps_by_id.get(pid, [])
        if not texts:
            print(f"WARN: no bundled steps parsed for {pid}", file=sys.stderr)
            continue
        for i, text in enumerate(texts):
            for voice in ("m", "f"):
                path = template.replace("{voice}", voice).replace("{n}", str(i + 1))
                if start_num is not None:
                    path = path.replace("{f}", str(start_num + i))
                path += ".mp3"
                # GuidedPrayerView feeds the analyzer ALL lines of the step
                # (prompts included — the voiceover reads them).
                manifest.append({"path": path, "lang": GUIDED_LANG[pid],
                                 "lines": text.split("\n")})

    # --- Rosaries ---
    for res in ["rosary_de", "rosary_freudenreich_de", "rosary_lichtreich_de",
                "rosary_schmerzhaft_de", "rosary_glorious_en", "rosary_joyful_en",
                "rosary_luminous_en", "rosary_sorrowful_en"]:
        lang = "en" if res.endswith("_en") else "de"
        doc = json.load(open(os.path.join(APP, res + ".json")))
        for step in doc["steps"]:
            for folder_key, audio_key in [("audioFolder", "audio"),
                                          ("audioFolderFemale", "audioFemale")]:
                folder = doc.get(folder_key) or doc.get("audioFolder")
                file = step.get(audio_key) or step.get("audio")
                if not folder or not file:
                    continue
                path = f"{folder}/{file}"
                if path in seen:
                    continue  # female falling back to the male clip
                seen.add(path)
                manifest.append({"path": path, "lang": lang,
                                 "lines": rosary_phrases(step["text"])})

    missing = [m["path"] for m in manifest
               if not os.path.exists(os.path.join(REPO, m["path"]))]
    print(f"manifest: {len(manifest)} clips, {len(missing)} missing on disk")
    for p in missing[:20]:
        print("  MISS", p)
    json.dump(manifest, open("clips_manifest.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
