"""T27: narration timing and subtitle files from docs/DEMO_SCRIPT.md (Stream A's script, read from main).

    PYTHONPATH=src ../med-tsunami-ais/.venv/bin/python scripts/demo/make_narration.py

Parses each scene's **Narration:** quote, checks it against the scene length at the script's pace
(about 2 words per second) and writes, for each cut:
  outputs/demo/narration_full.srt / narration_3min.srt   subtitles timed to the joined video
  outputs/demo/narration_full.md  / narration_3min.md    presenter sheet: start time, scene, line, fit
The 3-minute cut uses the full-cut lines; scenes whose line is too long for the trimmed length are flagged.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

WPS = 2.5          # normal presenting pace (150 words/min); the script says about 2, which needs 319 s
LEAD_IN = 0.4      # seconds of silence at the start of each scene


def parse_script(path: Path) -> dict[str, str]:
    """{scene tag e.g. '08', '09b': narration text}."""
    text = path.read_text()
    out = {}
    for block in re.split(r"\n## Scene ", text)[1:]:
        m = re.match(r"(\d+)([a-z]?)\b", block)
        if not m:
            continue
        tag = f"{int(m.group(1)):02d}{m.group(2)}"
        nm = re.search(r"\*\*Narration:\*\*\s*\n((?:>.*\n?)+)", block)
        if nm:
            line = " ".join(ln.lstrip("> ").strip() for ln in nm.group(1).splitlines())
            out[tag] = line.strip().strip('"').strip("“”")
    return out


def srt_time(s: float) -> str:
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def split_sentences(line: str, max_words: int = 14) -> list[str]:
    """Subtitle-sized pieces: sentences, then clauses, then word chunks."""
    parts = []
    for sent in re.split(r"(?<=[.!?])\s+", line):
        words = sent.split()
        while len(words) > max_words:
            cut = max((i for i, w in enumerate(words[:max_words]) if w.endswith(",")), default=max_words - 1) + 1
            parts.append(" ".join(words[:cut]))
            words = words[cut:]
        if words:
            parts.append(" ".join(words))
    return parts


def build(cut: str, lines: dict[str, str]):
    srt, rows, t0, k = [], [], 0.0, 1
    for num in cm.ORDER:
        spec = cm.SCENES[num]
        dur = cm.duration(num, cut)
        if dur is None:
            continue
        line = lines.get(spec.tag, "")
        n_words = len(line.split())
        need = n_words / WPS
        fit = "ok" if need <= dur - LEAD_IN else f"TOO LONG by {need - (dur - LEAD_IN):.1f} s"
        rows.append((t0, spec, line, n_words, dur, need, fit))
        pieces = split_sentences(line) if line else []
        t = t0 + LEAD_IN
        span = min(need, dur - LEAD_IN)           # squeeze if too long, so cues never cross scenes
        for p in pieces:
            d = span * len(p.split()) / max(1, n_words)
            srt.append(f"{k}\n{srt_time(t)} --> {srt_time(t + d)}\n{p}\n")
            k, t = k + 1, t + d
        t0 += dur
    return srt, rows, t0


def main():
    if not cm.SCRIPT.exists():
        sys.exit(f"{cm.SCRIPT} not found")
    lines = parse_script(cm.SCRIPT)
    missing = [cm.SCENES[n].tag for n in cm.ORDER if cm.SCENES[n].tag not in lines]
    if missing:
        print("no narration found for scenes:", ", ".join(missing))
    cm.OUT_DEMO.mkdir(parents=True, exist_ok=True)
    for cut in ("full", "3min"):
        srt, rows, total = build(cut, lines)
        (cm.OUT_DEMO / f"narration_{cut}.srt").write_text("\n".join(srt))
        md = [f"# Narration, {cut} cut ({total:.0f} s), from docs/DEMO_SCRIPT.md at {WPS:g} words/s\n",
              "| Start | Scene | Seconds | Words | Needs | Fit | Line |", "|---|---|---|---|---|---|---|"]
        for t0, spec, line, nw, dur, need, fit in rows:
            md.append(f"| {int(t0 // 60)}:{t0 % 60:04.1f} | {spec.tag} {spec.title} | {dur:g} | {nw} | "
                      f"{need:.1f} s | {fit} | {line} |")
        (cm.OUT_DEMO / f"narration_{cut}.md").write_text("\n".join(md) + "\n")
        bad = [r for r in rows if r[6] != "ok"]
        print(f"{cut}: {total:.0f} s, {len(rows)} scenes, {len(bad)} lines too long"
              + "".join(f"\n  scene {r[1].tag}: {r[3]} words in {r[4]:g} s ({r[6]})" for r in bad))


if __name__ == "__main__":
    main()
