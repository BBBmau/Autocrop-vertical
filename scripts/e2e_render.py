#!/usr/bin/env python3
"""End-to-end render check for the `autocrop` CLI, designed for CI.

Builds a synthetic 1280x720 H.264 fixture with three hard cuts:

    0-2s  wide shot, nobody detected            -> LETTERBOX
    2-4s  one person on the left                -> TRACK (zoom-in)
    4-6s  one person on the right               -> TRACK (pan)
    6-8s  wide shot, nobody detected            -> LETTERBOX (zoom-out)

then runs `main.cli()` exactly as the worker does — real scene detection,
real plan, real ffmpeg encode — with only YOLO detection stubbed (the
runner has no GPU and the fixture has no real people). It asserts that the
rendered output moves gradually through every boundary and writes
before/after artifacts for humans:

    fixture.mp4        the input ("before")
    rendered.mp4       the autocrop output ("after")
    plan.json          the scene/transition plan autocrop produced
    contact-sheet.jpg  source vs output frames around each boundary
    report.json        per-frame measurements and pass/fail per check
    summary.md         GitHub-flavoured summary (also printed to stdout)

Exit status is non-zero when any check fails; artifacts are still written so
CI can upload them for inspection.

Requires: opencv-python(-headless), numpy, scenedetect, tqdm, ffmpeg on PATH.
Does not require ultralytics/torch.
"""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import main as autocrop  # noqa: E402

W, H = 1280, 720
SCENES = [
    # (start_sec, end_sec, background BGR, person centre x or None)
    (0, 2, (44, 40, 62), None),
    (2, 4, (205, 200, 190), W // 4),
    (4, 6, (60, 120, 200), W * 3 // 4),
    (6, 8, (62, 40, 44), None),
]
PERSON_COLOR = (118, 210, 112)


def scene_at(sec):
    for start, end, bg, person_x in SCENES:
        if start <= sec < end:
            return start, end, bg, person_x
    return SCENES[-1]


def person_box(center_x):
    return [center_x - 105, H // 3 - 75, center_x + 105, H - 70]


def make_fixture(path, fps):
    raw = path.with_name("fixture_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    total = int(SCENES[-1][1] * fps)
    for n in range(total):
        _, _, bg, person_x = scene_at(n / fps)
        frame = np.full((H, W, 3), bg, dtype=np.uint8)
        # Blue channel encodes source column so output pixels reveal which
        # source columns are on screen. Background channels stay >= 40 so
        # letterbox bars (pure black) are distinguishable.
        frame[:, :, 0] = (np.arange(W) * 255 // (W - 1)).astype(np.uint8)[None, :]
        for x in range(0, W, 80):
            cv2.line(frame, (x, 0), (x, H), (80, 80, 80), 1)
        if person_x is not None:
            box = person_box(person_x)
            cv2.circle(frame, (person_x, H // 3), 75, PERSON_COLOR, -1)
            cv2.rectangle(frame, (box[0], box[1] + 150), (box[2], box[3]), PERSON_COLOR, -1)
        cv2.putText(frame, f"autocrop e2e fixture  t={n / fps:05.2f}s", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235, 235, 235), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(path)],
        check=True)
    raw.unlink()
    return total


def fake_analyze_scene_content(video_path, start, end, samples=1):
    """Stand-in for YOLO: report the fixture's scripted person, if any."""
    _, _, _, person_x = scene_at(start.get_seconds() + 0.05)
    if person_x is None:
        return []
    return [{"person_box": person_box(person_x), "face_box": None, "motion": 0.0}]


def measure_output(path):
    """Per output frame: (letterbox bar height, approx source x of left edge)."""
    cap = cv2.VideoCapture(str(path))
    bars, lefts = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        column = frame[:, w // 2, :].max(axis=1)
        rows = np.where(column > 20)[0]
        bar = int(rows[0]) if len(rows) else h // 2
        bars.append(bar)
        # Sample inside the picture (below the top bar) to read the blue code.
        y = min(h - 1, bar + (h - 2 * bar) // 2)
        lefts.append(int(frame[y, 0:4, 0].mean() * (W - 1) / 255))
    cap.release()
    return bars, lefts


def gradual(values, expect_increasing, min_distinct, max_step):
    distinct = len(set(values))
    steps = [b - a for a, b in zip(values, values[1:])]
    if not expect_increasing:
        steps = [-s for s in steps]
    monotonic = all(s >= -2 for s in steps)  # tolerate 1-2px codec noise
    return {
        "distinct": distinct,
        "maxStep": max(abs(s) for s in steps) if steps else 0,
        "monotonic": monotonic,
        "ok": distinct >= min_distinct and monotonic and max(abs(s) for s in steps) <= max_step,
    }


def contact_sheet(fixture, rendered, boundaries, fps, out_path):
    src = cv2.VideoCapture(str(fixture))
    out = cv2.VideoCapture(str(rendered))
    out_w = int(out.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(out.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cell_h = 270
    src_cell_w = int(round(cell_h * W / H))
    out_cell_w = int(round(cell_h * out_w / out_h))
    offsets = [-3, 0, 3, 6, 9, 15]
    rows = []
    for name, boundary in boundaries:
        top, bottom = [], []
        for off in offsets:
            n = boundary + off
            src.set(cv2.CAP_PROP_POS_FRAMES, n)
            out.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok_s, fs = src.read()
            ok_o, fo = out.read()
            if not (ok_s and ok_o):
                fs = np.zeros((H, W, 3), np.uint8)
                fo = np.zeros((out_h, out_w, 3), np.uint8)
            fs = cv2.resize(fs, (src_cell_w, cell_h))
            fo = cv2.resize(fo, (out_cell_w, cell_h))
            pad = np.zeros((cell_h, src_cell_w - out_cell_w, 3), np.uint8)
            fo = np.hstack([fo, pad])
            label = f"{name} {off:+d}f"
            for img in (fs, fo):
                cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)
            top.append(fs)
            bottom.append(fo)
        rows.append(np.hstack(top))
        rows.append(np.hstack(bottom))
        rows.append(np.full((12, src_cell_w * len(offsets), 3), 90, np.uint8))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    src.release()
    out.release()


def run(args):
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture = out_dir / "fixture.mp4"
    rendered = out_dir / "rendered.mp4"
    plan_path = out_dir / "plan.json"
    fps = args.fps

    total_frames = make_fixture(fixture, fps)
    autocrop.analyze_scene_content = fake_analyze_scene_content
    sys.argv = [
        "autocrop", "-i", str(fixture), "-o", str(rendered),
        "--plan-json", str(plan_path),
        "--pan-duration", str(args.pan_duration),
        "--zoom-duration", str(args.zoom_duration),
        "--scene-threshold", str(args.scene_threshold),
        "--quality", "fast",
    ]
    try:
        autocrop.cli()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise RuntimeError(f"autocrop exited with {exc.code}")

    plan = json.load(open(plan_path))
    summary = plan["summary"]
    bars, lefts = measure_output(rendered)

    boundaries = [(s[0], int(s[0] * fps)) for s in SCENES[1:]]
    zoom_in_b = int(SCENES[1][0] * fps)
    pan_b = int(SCENES[2][0] * fps)
    zoom_out_b = int(SCENES[3][0] * fps)
    zoom_frames = max(2, int(round(args.zoom_duration * fps)))
    pan_frames = max(2, int(round(args.pan_duration * fps)))

    checks = {}
    checks["frameCount"] = {"expected": total_frames, "actual": len(bars),
                            "ok": len(bars) == total_frames}
    checks["planSummary"] = {
        "summary": summary,
        "ok": summary.get("zoom") == 2 and summary.get("pan") == 1
        and summary.get("layout_switch") == 0,
    }
    zin = bars[zoom_in_b - 1: zoom_in_b + zoom_frames + 1]
    zout = bars[zoom_out_b - 1: zoom_out_b + zoom_frames + 1]
    pan = lefts[pan_b - 1: pan_b + pan_frames + 1]
    checks["zoomIn"] = dict(values=zin, **gradual(zin, False, 6, 60))
    checks["zoomIn"]["ok"] = checks["zoomIn"]["ok"] and zin[0] > 100 and zin[-1] == 0
    checks["zoomOut"] = dict(values=zout, **gradual(zout, True, 6, 60))
    checks["zoomOut"]["ok"] = checks["zoomOut"]["ok"] and zout[0] == 0 and zout[-1] > 100
    checks["pan"] = dict(values=pan, **gradual(pan, True, 6, 200))
    ok = all(c["ok"] for c in checks.values())

    contact_sheet(fixture, rendered,
                  [("zoom-in", zoom_in_b), ("pan", pan_b), ("zoom-out", zoom_out_b)],
                  fps, out_dir / "contact-sheet.jpg")

    report = {
        "ok": ok,
        "fixture": str(fixture),
        "rendered": str(rendered),
        "fps": fps,
        "panDuration": args.pan_duration,
        "zoomDuration": args.zoom_duration,
        "checks": checks,
        "barHeightByFrame": bars,
        "leftSourceXByFrame": lefts,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"## Autocrop render E2E — {'PASS' if ok else 'FAIL'}",
        "",
        f"Fixture: LETTERBOX → TRACK(left) → TRACK(right) → LETTERBOX, "
        f"{W}x{H} @ {fps}fps, pan {args.pan_duration}s, zoom {args.zoom_duration}s.",
        "",
        f"Plan: `{summary.get('pan')} pan / {summary.get('zoom')} zoom / "
        f"{summary.get('hold')} hold / {summary.get('layout_switch')} layout-switch` "
        f"over {summary.get('track_to_track')} TRACK->TRACK + "
        f"{summary.get('layout_boundaries')} layout boundaries.",
        "",
        "| check | result | detail |",
        "|---|---|---|",
        f"| frame count | {'✅' if checks['frameCount']['ok'] else '❌'} | "
        f"{checks['frameCount']['actual']} / {checks['frameCount']['expected']} |",
        f"| plan | {'✅' if checks['planSummary']['ok'] else '❌'} | "
        f"expected 1 pan, 2 zoom, 0 layout-switch |",
        f"| zoom-in bars | {'✅' if checks['zoomIn']['ok'] else '❌'} | "
        f"{checks['zoomIn']['distinct']} distinct heights, max step "
        f"{checks['zoomIn']['maxStep']}px: `{zin}` |",
        f"| pan crop x | {'✅' if checks['pan']['ok'] else '❌'} | "
        f"{checks['pan']['distinct']} distinct positions, max step "
        f"{checks['pan']['maxStep']}px: `{pan}` |",
        f"| zoom-out bars | {'✅' if checks['zoomOut']['ok'] else '❌'} | "
        f"{checks['zoomOut']['distinct']} distinct heights, max step "
        f"{checks['zoomOut']['maxStep']}px: `{zout}` |",
        "",
        "Artifacts: `fixture.mp4` (before), `rendered.mp4` (after), "
        "`contact-sheet.jpg`, `plan.json`, `report.json`.",
    ]
    summary_md = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary_md)
    print(summary_md)
    return ok


def main_():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default="autocrop-e2e")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pan-duration", type=float, default=0.4)
    parser.add_argument("--zoom-duration", type=float, default=0.5)
    parser.add_argument("--scene-threshold", type=float, default=27)
    args = parser.parse_args()
    try:
        ok = run(args)
    except Exception:
        traceback.print_exc()
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "summary.md").write_text(
            "## Autocrop render E2E — ERROR\n\n```\n" + traceback.format_exc() + "```\n")
        sys.exit(2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main_()
