"""T27: render every demo scene (in parallel) and join them into the two cuts.

    PYTHONPATH=src ../med-tsunami-ais/.venv/bin/python scripts/demo/build_all.py [--jobs N] [--only 1,2,8]
                                                    [--cut full|3min|both] [--no-join] [--draft]

Every scene is split into 5-second chunks and all chunks share one process pool (longest scenes
first), so no single slow scene holds up the build. --draft renders half resolution at 15 fps into
outputs/demo/draft/ (about 4x faster) and joins outputs/demo/draft/demo_full.mp4 for a quick watch.

Writes outputs/demo/scene_XX.mp4 (+ scene_XX_3min.mp4 where the 3-minute length differs),
outputs/demo/scene_XX.png (one still per scene), outputs/demo/demo_full.mp4 and
outputs/demo/demo_3min.mp4. A scene that crashes is replaced by a rendered placeholder
card so the joined video always has the right length, and the failure is listed at the end.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def _render_chunk(num: int, cut: str, k0: int, k1: int, part: str, quality: str):
    import common as cm
    try:
        cm.render_chunk(num, cut, k0, k1, Path(part), quality)
        return num, cut, "ok", part
    except Exception:  # noqa: BLE001  (the whole scene falls back to a placeholder card)
        return num, cut, "FAILED", traceback.format_exc()


def _render_still(num: int):
    import common as cm
    try:
        cm.render_still(num, "full")
        return num, "full", "ok", "still"
    except Exception:  # noqa: BLE001
        return num, "full", "FAILED", traceback.format_exc()


def _render_fallback(num, cut, msg, quality="final"):
    import common as cm

    class _M:
        SCENE = num

        @staticmethod
        def build(cut, ctx):
            fig, chrome = cm.new_figure(num, caption="scene failed to render; see build log")
            cm.placeholder(fig, f"Scene {num}: {cm.SCENES[num].title}", needs=msg[:90],
                           rect=(0.1, 0.15, 0.8, 0.65))
            return cm.Scene(fig, lambda t: None)

    cm.render(_M, cut, verbose=False, quality=quality)


def join(cut: str, nums, quality: str = "final"):
    base = common.OUT_DEMO if quality == "final" else common.OUT_DEMO / "draft"
    lst = base / f"concat_{cut}.txt"
    files = []
    for n in nums:
        if common.duration(n, cut) is None:
            continue
        mp4 = common.video_path(n, cut, quality)
        if not mp4.exists():
            print(f"  missing {mp4.name}; {cut} cut not joined")
            return None
        files.append(mp4)
    lst.write_text("".join(f"file '{f.resolve()}'\n" for f in files))
    out = base / f"demo_{cut}.mp4"
    # Every scene uses the same codec settings, so stream copy is safe and instant.
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
           "-c", "copy", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "default=nw=1:nk=1", str(out)], capture_output=True, text=True).stdout.strip()
    print(f"  {out.relative_to(common.REPO)}: {float(dur):.1f} s from {len(files)} scenes")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=common.default_jobs())
    ap.add_argument("--only", default="", help="comma-separated scene numbers")
    ap.add_argument("--cut", default="both", choices=["full", "3min", "both"])
    ap.add_argument("--no-join", action="store_true")
    ap.add_argument("--draft", action="store_true", help="half resolution, 15 fps, into outputs/demo/draft/")
    a = ap.parse_args()
    quality = "draft" if a.draft else "final"
    nums = list(common.ORDER) if not a.only else [common.scene_id(x) for x in a.only.split(",")]
    cuts = ["full", "3min"] if a.cut == "both" else [a.cut]

    jobs = []
    for n in nums:
        stem = common.SCENES[n].stem
        if not (common.DEMO_DIR / f"{stem}.py").exists():
            print(f"{stem}.py missing; rendering a placeholder")
            for c in cuts:
                if common.duration(n, c) is not None:
                    _render_fallback(n, c, f"scripts/demo/{stem}.py not written yet", quality)
            continue
        for c in cuts:
            if common.duration(n, c) is None:
                continue
            mp4_full, _ = common.out_paths(n, "full")
            mp4_c, _ = common.out_paths(n, c)
            if c == "3min" and mp4_c == mp4_full and "full" in cuts:
                continue  # identical file, rendered by the full job
            jobs.append((n, c))

    # split every scene into chunks; longest scenes first so the pool drains evenly
    fps = common.QUALITY[quality]["fps"]
    parts = {}
    tasks = []
    for n, c in sorted(jobs, key=lambda j: -common.duration(*j)):
        out = common.video_path(n, c, quality)
        chunks = common.plan_chunks(*common.frame_range(n, c, quality), fps)
        parts[(n, c)] = (out, [common.part_path(out, i) for i in range(len(chunks))])
        tasks += [(n, c, k0, k1, str(pp), quality) for (k0, k1), pp in zip(chunks, parts[(n, c)][1])]

    t0 = time.time()
    failed = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(_render_chunk, *t) for t in tasks]
        if quality == "final":
            futs += [ex.submit(_render_still, n) for n, c in jobs if c == "full"]
        for f in as_completed(futs):
            n, c, state, info = f.result()
            if state != "ok" and (n, c) not in failed:
                failed[(n, c)] = info
    for (n, c), (out, pp) in sorted(parts.items()):
        if (n, c) in failed:
            for x in pp:
                x.unlink(missing_ok=True)
            print(f"scene {common.SCENES[n].tag} [{c}]: FAILED, placeholder rendered")
            _render_fallback(n, c, failed[(n, c)].strip().splitlines()[-1], quality)
        else:
            common.concat_videos(pp, out)
            print(f"scene {common.SCENES[n].tag} [{c}]: ok ({len(pp)} chunks)")
    print(f"rendered {len(jobs)} scenes ({len(tasks)} chunks, {a.jobs} processes, {quality}) "
          f"in {time.time() - t0:.0f} s")

    if not a.no_join and not a.only:
        for c in cuts:
            join(c, common.ORDER, quality)
    for (n, c), info in failed.items():
        print(f"\n=== scene {common.SCENES[n].tag} [{c}] FAILED ===\n{info}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
