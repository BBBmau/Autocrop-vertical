#!/usr/bin/env python3
"""End-to-end render checks for the `autocrop` CLI, designed for CI.

Two cases, each rendered through `main.cli()` exactly as the worker runs it
(real scene detection, real plan, real ffmpeg encode) with only the ML
detectors stubbed, because the runner has no GPU and the fixtures contain
drawn stand-ins rather than real people:

  transitions/  synthetic 1280x720 H.264 fixture with three hard cuts:
                  0-2s  wide shot, nobody detected      -> LETTERBOX
                  2-4s  one person on the left          -> TRACK (zoom-in)
                  4-6s  one person on the right         -> TRACK (pan)
                  6-8s  wide shot, nobody detected      -> LETTERBOX (zoom-out)
                Asserts every boundary eases instead of snapping.

  speaker/      one 8s scene with two people on screen the whole time and a
                scripted speaker score series (A talks, B interjects too
                briefly to matter, B takes the floor, both talk over each
                other, A returns). Runs with --speaker-focus auto and asserts
                the scene is split at speaker turns only, that turns pan and
                crosstalk widens to the group, and writes the debug overlay.
                Also loads the real YuNet face detector once as a smoke test.

Per case: fixture.mp4 (before), rendered.mp4 (after), plan.json,
contact-sheet.jpg (source vs output around each boundary), report.json and
(speaker) overlay.mp4. The root gets summary.md and report.json.

Exit status is non-zero when any check fails; artifacts are still written.
Requires: opencv-python(-headless), numpy, scenedetect, tqdm, ffmpeg on PATH.
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

import asd  # noqa: E402
import main as autocrop  # noqa: E402
import speaker  # noqa: E402

W, H = 1280, 720
PERSON_COLOR = (118, 210, 112)
HEAD_R = 75
HEAD_Y = H // 3

# (start_sec, end_sec, background BGR, [person centre x, ...])
TRANSITION_SCENES = [
    (0, 2, (44, 40, 62), []),
    (2, 4, (205, 200, 190), [W // 4]),
    (4, 6, (60, 120, 200), [W * 3 // 4]),
    (6, 8, (62, 40, 44), []),
]
SPEAKER_SCENES = [
    (0, 8, (70, 110, 150), [W // 4, W * 3 // 4]),
]
# Speaker script for the speaker case, in seconds: (start, end, track ids
# speaking). Track 0 is the left person, track 1 the right one.
SPEAKER_SCRIPT = [
    (0.0, 3.0, [0]),
    (1.5, 1.8, [1]),        # interjection, shorter than dwell: must not switch
    (3.0, 5.5, [1]),
    (5.5, 6.8, [0, 1]),     # crosstalk >= dwell: group (LETTERBOX)
    (6.8, 8.0, [0]),
]
SPEAKER_DWELL_SEC = 1.0
# Face-zoom case: same two people, but drawn with small heads (56px faces =
# 8% of frame height, like a wide shot) and a simple A-then-B script.
FACEZOOM_HEAD_R = 28
FACEZOOM_SCENES = [
    (0, 8, (90, 100, 120), [W // 4, W * 3 // 4]),
]
FACEZOOM_SCRIPT = [
    (0.0, 4.0, [0]),
    (4.0, 8.0, [1]),
]


def scene_at(scenes, sec):
    for start, end, bg, people in scenes:
        if start <= sec < end:
            return start, end, bg, people
    return scenes[-1]


def person_box(center_x, head_r=HEAD_R):
    return [center_x - int(1.4 * head_r), HEAD_Y - head_r, center_x + int(1.4 * head_r), H - 70]


def face_box(center_x, head_r=HEAD_R):
    return [center_x - head_r, HEAD_Y - head_r, center_x + head_r, HEAD_Y + head_r]


def make_fixture(path, fps, scenes, label, head_r=HEAD_R, row_gradient=False):
    raw = path.with_name(path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    total = int(scenes[-1][1] * fps)
    for n in range(total):
        _, _, bg, people = scene_at(scenes, n / fps)
        frame = np.full((H, W, 3), bg, dtype=np.uint8)
        # Blue channel encodes the source column so output pixels reveal which
        # source columns are on screen. Other channels stay >= 40 so letterbox
        # bars (pure black) are distinguishable.
        frame[:, :, 0] = (np.arange(W) * 255 // (W - 1)).astype(np.uint8)[None, :]
        if row_gradient:
            # Red channel encodes the source row (40..255) so output pixels
            # reveal how much source height is on screen (face zoom).
            frame[:, :, 2] = (40 + np.arange(H) * 215 // (H - 1)).astype(np.uint8)[:, None]
        for x in range(0, W, 80):
            cv2.line(frame, (x, 0), (x, H), (80, 80, 80), 1)
        for cx in people:
            box = person_box(cx, head_r)
            cv2.circle(frame, (cx, HEAD_Y), head_r, PERSON_COLOR, -1)
            cv2.rectangle(frame, (box[0], box[1] + 2 * head_r), (box[2], box[3]),
                          PERSON_COLOR, -1)
        cv2.putText(frame, f"autocrop e2e {label}  t={n / fps:05.2f}s", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235, 235, 235), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    # Add a synthetic voice-like audio track (tone bursts) so the audio mux path
    # and the Light-ASD feature extraction run on real streams in CI.
    duration = scenes[-1][1]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
         "-f", "lavfi", "-t", str(duration),
         "-i", "sine=frequency=220:sample_rate=16000,tremolo=f=4:d=0.9,volume=0.4",
         "-map", "0:v", "-map", "1:a", "-shortest",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
         "-c:a", "aac", "-b:a", "64k", str(path)],
        check=True)
    raw.unlink()
    return total


def fake_analyze_for(scenes, head_r=HEAD_R):
    """Stand-in for YOLO: report the fixture's scripted people, if any."""
    def fake_analyze_scene_content(video_path, start, end, samples=1):
        _, _, _, people = scene_at(scenes, start.get_seconds() + 0.05)
        return [{"person_box": person_box(cx, head_r), "face_box": None, "motion": 0.0}
                for cx in people]
    return fake_analyze_scene_content


def fake_track_faces_for(scenes, head_r=HEAD_R):
    """Stand-in for YuNet tracking: one steady face track per drawn person."""
    def fake_track_faces(video_path, start_frame, end_frame, fps, **kwargs):
        _, _, _, people = scene_at(scenes, start_frame / fps + 0.05)
        return [{"id": i, "first": start_frame, "last": end_frame - 1,
                 "boxes": {n: face_box(cx, head_r) for n in range(start_frame, end_frame)}}
                for i, cx in enumerate(people)]
    return fake_track_faces


def fake_score_for(script):
    def fake_score_speaking(video_path, scene, tracks, fps, log=None):
        length = scene["end_frame"] - scene["start_frame"]
        scores = {t["id"]: np.zeros(length) for t in tracks}
        for start, end, ids in script:
            a = int(round(start * fps)) - scene["start_frame"]
            b = int(round(end * fps)) - scene["start_frame"]
            for i in ids:
                if i in scores:
                    scores[i][max(0, a):max(0, b)] = 0.9
        return scores
    return fake_score_speaking


fake_score_speaking = fake_score_for(SPEAKER_SCRIPT)


def measure_visible_height(path):
    """Per output frame: source rows visible (from the red row gradient)."""
    cap = cv2.VideoCapture(str(path))
    heights = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        # Sample a background band near the left edge (people are centred on
        # W/4 and 3W/4, far from the crop's left edge), a few rows in from the
        # top/bottom to dodge codec ringing. The median over a 40px-wide band
        # ignores the fixture's 1px grid lines when a pan edge crosses one.
        top = float(np.median(frame[4:10, 2:42, 2]))
        bottom = float(np.median(frame[h - 10:h - 4, 2:42, 2]))
        to_row = lambda code: (code - 40) * (H - 1) / 215  # noqa: E731
        heights.append(int(round(to_row(bottom) - to_row(top))))
    cap.release()
    return heights


def run_cli(fixture, rendered, plan_path, args, extra):
    sys.argv = [
        "autocrop", "-i", str(fixture), "-o", str(rendered),
        "--plan-json", str(plan_path),
        "--pan-duration", str(args.pan_duration),
        "--zoom-duration", str(args.zoom_duration),
        "--scene-threshold", str(args.scene_threshold),
        "--quality", "fast",
    ] + extra
    try:
        autocrop.cli()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise RuntimeError(f"autocrop exited with {exc.code}")
    return json.load(open(plan_path))


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
        # Sample a row that maps to source y ~ 100: background in both
        # fixtures (heads start at y=165), below the title text.
        y = min(h - 1, bar + int(0.14 * (h - 2 * bar)))
        # Median of a small patch: 4:2:0 chroma subsampling and the codec's
        # deblocking make single blue samples noisy by several source pixels.
        y0, y1 = max(0, y - 3), min(h, y + 4)
        code = float(np.median(frame[y0:y1, 0:6, 0]))
        lefts.append(int(round(code * (W - 1) / 255)))
    cap.release()
    return bars, lefts


def gradual(values, expect_increasing, min_distinct, max_step_ratio, noise=2):
    """Judge whether a measured trajectory eased rather than snapped.

    A snap is one step covering (nearly) the whole travel; an eased
    transition spreads it over several frames, so the largest step must stay
    below max_step_ratio of the total travel. Measurements come from decoded
    H.264, so backwards wiggles up to `noise` pixels are tolerated.
    """
    distinct = len(set(values))
    steps = [b - a for a, b in zip(values, values[1:])]
    if not expect_increasing:
        steps = [-s for s in steps]
    travel = abs(values[-1] - values[0]) if values else 0
    max_step = max((abs(s) for s in steps), default=0)
    monotonic = all(s >= -noise for s in steps)
    return {
        "distinct": distinct,
        "travel": travel,
        "maxStep": max_step,
        "maxStepRatio": round(max_step / travel, 3) if travel else None,
        "monotonic": monotonic,
        "ok": distinct >= min_distinct and monotonic and travel > 0
        and max_step <= max_step_ratio * travel,
    }


def contact_sheet(fixture, rendered, boundaries, out_path):
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


def mark(ok):
    return "✅" if ok else "❌"


def describe(check, values):
    return (f"{check['distinct']} distinct, max step {check['maxStep']}px "
            f"({check['maxStepRatio']} of travel): `{values}`")


def window(series, boundary, frames):
    return series[max(0, boundary - 1): boundary + frames + 1]


# --------------------------------------------------------------------------
# Case 1: cuts -> zoom-in, pan, zoom-out
# --------------------------------------------------------------------------
def run_transitions(out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture, rendered, plan_path = (out_dir / "fixture.mp4", out_dir / "rendered.mp4",
                                    out_dir / "plan.json")
    fps = args.fps
    total_frames = make_fixture(fixture, fps, TRANSITION_SCENES, "transitions")
    autocrop.analyze_scene_content = fake_analyze_for(TRANSITION_SCENES)
    plan = run_cli(fixture, rendered, plan_path, args, [])
    summary = plan["summary"]
    bars, lefts = measure_output(rendered)

    zoom_in_b = int(TRANSITION_SCENES[1][0] * fps)
    pan_b = int(TRANSITION_SCENES[2][0] * fps)
    zoom_out_b = int(TRANSITION_SCENES[3][0] * fps)
    zoom_frames = max(2, int(round(args.zoom_duration * fps)))
    pan_frames = max(2, int(round(args.pan_duration * fps)))

    checks = {}
    checks["frameCount"] = {"expected": total_frames, "actual": len(bars),
                            "ok": len(bars) == total_frames}
    checks["planSummary"] = {
        "summary": summary,
        "ok": summary.get("zoom") == 2 and summary.get("pan") == 1
        and summary.get("layout_switch") == 0 and summary.get("speaker_turns") == 0,
    }
    zin = window(bars, zoom_in_b, zoom_frames)
    zout = window(bars, zoom_out_b, zoom_frames)
    pan = window(lefts, pan_b, pan_frames)
    # Eased smoothstep peaks at 1.5x the average per-frame step, i.e. about
    # 0.1-0.15 of the travel for these durations; a snap is ~1.0. Bar heights
    # are exact pixel rows, the pan readback is decoded chroma (noisier).
    checks["zoomIn"] = dict(values=zin, **gradual(zin, False, 6, 0.3))
    checks["zoomIn"]["ok"] = checks["zoomIn"]["ok"] and zin[0] > 100 and zin[-1] == 0
    checks["zoomOut"] = dict(values=zout, **gradual(zout, True, 6, 0.3))
    checks["zoomOut"]["ok"] = checks["zoomOut"]["ok"] and zout[0] == 0 and zout[-1] > 100
    checks["pan"] = dict(values=pan, **gradual(pan, True, 6, 0.4, noise=12))
    ok = all(c["ok"] for c in checks.values())

    contact_sheet(fixture, rendered,
                  [("zoom-in", zoom_in_b), ("pan", pan_b), ("zoom-out", zoom_out_b)],
                  out_dir / "contact-sheet.jpg")
    report = {"ok": ok, "checks": checks, "barHeightByFrame": bars,
              "leftSourceXByFrame": lefts}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"### transitions — {'PASS' if ok else 'FAIL'}",
        "",
        f"LETTERBOX → TRACK(left) → TRACK(right) → LETTERBOX, {W}x{H} @ {fps}fps, "
        f"pan {args.pan_duration}s, zoom {args.zoom_duration}s. "
        f"Plan: `{summary.get('pan')} pan / {summary.get('zoom')} zoom / "
        f"{summary.get('hold')} hold / {summary.get('layout_switch')} layout-switch`.",
        "",
        "| check | result | detail |",
        "|---|---|---|",
        f"| frame count | {mark(checks['frameCount']['ok'])} | "
        f"{checks['frameCount']['actual']} / {checks['frameCount']['expected']} |",
        f"| plan | {mark(checks['planSummary']['ok'])} | expected 1 pan, 2 zoom, 0 layout-switch |",
        f"| zoom-in bars | {mark(checks['zoomIn']['ok'])} | {describe(checks['zoomIn'], zin)} |",
        f"| pan crop x | {mark(checks['pan']['ok'])} | {describe(checks['pan'], pan)} |",
        f"| zoom-out bars | {mark(checks['zoomOut']['ok'])} | {describe(checks['zoomOut'], zout)} |",
    ]
    return ok, report, lines


# --------------------------------------------------------------------------
# Case 2: one scene, two people, scripted speaker turns
# --------------------------------------------------------------------------
def run_speaker(out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture, rendered, plan_path = (out_dir / "fixture.mp4", out_dir / "rendered.mp4",
                                    out_dir / "plan.json")
    overlay = out_dir / "overlay.mp4"
    fps = args.fps
    total_frames = make_fixture(fixture, fps, SPEAKER_SCENES, "speaker")

    checks = {}
    # Smoke-test the real face detector (downloads YuNet on first use) so a
    # broken model URL or OpenCV build fails here, not in production.
    try:
        faces = speaker.detect_faces(np.zeros((H, W, 3), np.uint8))
        checks["faceDetector"] = {"ok": faces == [], "facesOnBlank": len(faces),
                                  "model": speaker.face_model_path()}
    except Exception as exc:  # noqa: BLE001
        checks["faceDetector"] = {"ok": False, "error": repr(exc)}

    # Smoke-test the real Light-ASD path on the fixture: weights download, MFCC,
    # crops, resampling and inference. Drawn faces carry no lip motion, so we
    # only check shapes, range and runtime — accuracy is measured on real
    # footage by scripts/speaker_eval.py.
    import time
    scene = {"start_frame": 0, "end_frame": total_frames,
             "start_seconds": 0.0, "end_seconds": total_frames / fps}
    tracks = fake_track_faces_for(SPEAKER_SCENES)(str(fixture), 0, total_frames, fps)
    try:
        t0 = time.time()
        real = speaker.score_speaking(str(fixture), scene, tracks, fps)
        elapsed = time.time() - t0
        if real is None:
            checks["asdModel"] = {"ok": False, "error": "score_speaking returned None "
                                  "(torch/python_speech_features missing or no audio)"}
        else:
            arrays = [np.asarray(real[t["id"]]) for t in tracks]
            checks["asdModel"] = {
                "ok": all(a.shape == (total_frames,) for a in arrays)
                and all(np.all((a >= 0) & (a <= 1)) for a in arrays),
                "seconds": round(elapsed, 2),
                "meanScores": [round(float(a.mean()), 3) for a in arrays],
                "model": asd.model_path(),
            }
    except Exception as exc:  # noqa: BLE001
        checks["asdModel"] = {"ok": False, "error": repr(exc)}

    autocrop.analyze_scene_content = fake_analyze_for(SPEAKER_SCENES)
    speaker.track_faces = fake_track_faces_for(SPEAKER_SCENES)
    speaker.score_speaking = fake_score_speaking
    plan = run_cli(fixture, rendered, plan_path, args, [
        "--speaker-focus", "auto",
        "--speaker-min-dwell", str(SPEAKER_DWELL_SEC),
        # 'group' so the fixture exercises the widen/narrow (zoom) path too;
        # the production default 'loudest' is covered by unit tests.
        "--speaker-overlap", "group",
        "--debug-overlay", str(overlay),
    ])
    summary = plan["summary"]
    scenes = plan["scenes"]
    bars, lefts = measure_output(rendered)

    checks["frameCount"] = {"expected": total_frames, "actual": len(bars),
                            "ok": len(bars) == total_frames}
    turns = [s for s in scenes if s.get("boundary_source") == "speaker-turn"]
    kinds = [s["boundary_kind"] for s in turns]
    speakers = [s["speaker"]["track_id"] if (s.get("speaker") or {}).get("kind") == "track"
                else (s.get("speaker") or {}).get("kind") for s in scenes]
    expected_speakers = [0, 1, "group", 0]
    expected_kinds = ["pan", "zoom-out", "zoom-in"]
    checks["segments"] = {
        "speakers": speakers, "expected": expected_speakers,
        "starts": [s["start_frame"] for s in scenes],
        "ok": speakers == expected_speakers and summary.get("speaker_turns") == 3,
    }
    checks["turnKinds"] = {"kinds": kinds, "expected": expected_kinds,
                           "ok": kinds == expected_kinds}
    # Turn boundaries should sit within a few frames of the script (smoothing
    # shifts them slightly); the 0.3s interjection must not appear at all.
    script_starts = [int(round(3.0 * fps)), int(round(5.5 * fps)), int(round(6.8 * fps))]
    starts = [s["start_frame"] for s in turns]
    checks["turnTiming"] = {
        "starts": starts, "script": script_starts,
        "ok": len(starts) == 3 and all(abs(a - b) <= 6 for a, b in zip(starts, script_starts)),
    }
    checks["overlay"] = {"ok": overlay.exists() and overlay.stat().st_size > 0,
                         "bytes": overlay.stat().st_size if overlay.exists() else 0}

    zoom_frames = max(2, int(round(args.zoom_duration * fps)))
    pan_frames = max(2, int(round(args.pan_duration * fps)))
    if len(starts) == 3:
        pan = window(lefts, starts[0], pan_frames)
        zout = window(bars, starts[1], zoom_frames)
        zin = window(bars, starts[2], zoom_frames)
        checks["pan"] = dict(values=pan, **gradual(pan, True, 6, 0.4, noise=12))
        checks["zoomOut"] = dict(values=zout, **gradual(zout, True, 6, 0.3))
        checks["zoomOut"]["ok"] = checks["zoomOut"]["ok"] and zout[0] == 0 and zout[-1] > 100
        checks["zoomIn"] = dict(values=zin, **gradual(zin, False, 6, 0.3))
        checks["zoomIn"]["ok"] = checks["zoomIn"]["ok"] and zin[0] > 100 and zin[-1] == 0
    else:
        pan = zout = zin = []
        for key in ("pan", "zoomOut", "zoomIn"):
            checks[key] = {"ok": False, "distinct": 0, "maxStep": 0, "maxStepRatio": None}
    ok = all(c["ok"] for c in checks.values())

    boundaries = list(zip(["A->B pan", "crosstalk zoom-out", "B->A zoom-in"], starts))
    if boundaries:
        contact_sheet(fixture, rendered, boundaries, out_dir / "contact-sheet.jpg")
    report = {"ok": ok, "checks": checks, "script": SPEAKER_SCRIPT,
              "barHeightByFrame": bars, "leftSourceXByFrame": lefts}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"### speaker focus — {'PASS' if ok else 'FAIL'}",
        "",
        f"One scene, two people, scripted speaking: A 0-3s (B interjects 1.5-1.8s), "
        f"B 3-5.5s, crosstalk 5.5-6.8s, A 6.8-8s; dwell {SPEAKER_DWELL_SEC}s, "
        f"overlap policy `group` (scripted scores; real Light-ASD smoke-tested separately). "
        f"Plan: `{summary.get('speaker_turns')} speaker-turns, {summary.get('pan')} pan / "
        f"{summary.get('zoom')} zoom`.",
        "",
        "| check | result | detail |",
        "|---|---|---|",
        f"| face detector | {mark(checks['faceDetector']['ok'])} | "
        f"{checks['faceDetector'].get('error') or 'YuNet loaded, 0 faces on blank frame'} |",
        f"| Light-ASD | {mark(checks['asdModel']['ok'])} | "
        f"{checks['asdModel'].get('error') or ('scored ' + str(len(checks['asdModel']['meanScores'])) + ' tracks x ' + str(total_frames) + ' frames in ' + str(checks['asdModel']['seconds']) + 's; mean P(speaking) ' + str(checks['asdModel']['meanScores']))} |",
        f"| frame count | {mark(checks['frameCount']['ok'])} | "
        f"{checks['frameCount']['actual']} / {checks['frameCount']['expected']} |",
        f"| segments | {mark(checks['segments']['ok'])} | {speakers} (expected {expected_speakers}) |",
        f"| turn kinds | {mark(checks['turnKinds']['ok'])} | {kinds} (expected {expected_kinds}) |",
        f"| turn timing | {mark(checks['turnTiming']['ok'])} | starts {starts}, script {script_starts} |",
        f"| A->B pan | {mark(checks['pan']['ok'])} | {describe(checks['pan'], pan) if pan else 'n/a'} |",
        f"| crosstalk zoom-out | {mark(checks['zoomOut']['ok'])} | "
        f"{describe(checks['zoomOut'], zout) if zout else 'n/a'} |",
        f"| B->A zoom-in | {mark(checks['zoomIn']['ok'])} | "
        f"{describe(checks['zoomIn'], zin) if zin else 'n/a'} |",
        f"| debug overlay | {mark(checks['overlay']['ok'])} | {checks['overlay']['bytes']} bytes |",
    ]
    return ok, report, lines


def run_facezoom(out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture, rendered, plan_path = (out_dir / "fixture.mp4", out_dir / "rendered.mp4",
                                    out_dir / "plan.json")
    overlay = out_dir / "overlay.mp4"
    fps = args.fps
    head_r = FACEZOOM_HEAD_R
    total_frames = make_fixture(fixture, fps, FACEZOOM_SCENES, "facezoom",
                                head_r=head_r, row_gradient=True)
    autocrop.analyze_scene_content = fake_analyze_for(FACEZOOM_SCENES, head_r)
    speaker.track_faces = fake_track_faces_for(FACEZOOM_SCENES, head_r)
    speaker.score_speaking = fake_score_for(FACEZOOM_SCRIPT)
    plan = run_cli(fixture, rendered, plan_path, args, [
        "--speaker-focus", "auto",
        "--speaker-min-dwell", str(SPEAKER_DWELL_SEC),
        "--debug-overlay", str(overlay),
    ])
    summary = plan["summary"]
    scenes = plan["scenes"]
    _, lefts = measure_output(rendered)
    heights = measure_visible_height(rendered)

    checks = {}
    checks["frameCount"] = {"expected": total_frames, "actual": len(heights),
                            "ok": len(heights) == total_frames}
    # 56px face / 0.18 = 311px wanted; max upscale 2.0 caps the crop at 360.
    expected_h = H // 2
    zooms = [s.get("zoom_region") for s in scenes]
    checks["plan"] = {
        "zoomRegions": zooms, "faceZoomScenes": summary.get("face_zoom"),
        "ok": len(scenes) == 2 and summary.get("face_zoom") == 2
        and all(z and z[3] == expected_h for z in zooms),
    }
    # Face centre sits in the upper third of every zoomed crop.
    face_cy = HEAD_Y
    placements = [round((face_cy - z[1]) / z[3], 2) for z in zooms if z]
    checks["facePlacement"] = {"fractions": placements,
                               "ok": all(abs(p - 0.38) < 0.05 for p in placements)}
    turns = [s for s in scenes if s.get("boundary_source") == "speaker-turn"]
    t = turns[0]["transition"] if turns and turns[0].get("transition") else None
    checks["turn"] = {
        "kind": turns[0]["boundary_kind"] if turns else None,
        "heights": (t["from_h"], t["to_h"]) if t else None,
        "ok": bool(t) and turns[0]["boundary_kind"] == "pan"
        and t["from_h"] == t["to_h"] == expected_h,
    }
    start = turns[0]["start_frame"] if turns else None
    pan_frames = max(2, int(round(args.pan_duration * fps)))
    # Rendered frames show ~half the source height throughout (2x zoom).
    # Pan frames are skipped: while the crop edge sweeps across person A the
    # edge probe reads the drawn body, not the background gradient.
    steady = [v for i, v in enumerate(heights[5:-5], start=5)
              if start is None or not (start - 1 <= i <= start + pan_frames + 1)]
    checks["visibleHeight"] = {
        "min": min(steady) if steady else None, "max": max(steady) if steady else None,
        "expected": expected_h, "framesChecked": len(steady),
        "ok": bool(steady) and all(abs(v - expected_h) <= 12 for v in steady),
    }
    pan = window(lefts, start, pan_frames) if start is not None else []
    checks["pan"] = dict(values=pan, **gradual(pan, True, 6, 0.4, noise=12)) if pan else \
        {"ok": False, "distinct": 0, "maxStep": 0, "maxStepRatio": None}
    checks["overlay"] = {"ok": overlay.exists() and overlay.stat().st_size > 0,
                         "bytes": overlay.stat().st_size if overlay.exists() else 0}
    ok = all(c["ok"] for c in checks.values())

    if start is not None:
        contact_sheet(fixture, rendered, [("A->B pan (zoomed)", start)],
                      out_dir / "contact-sheet.jpg")
    report = {"ok": ok, "checks": checks, "script": FACEZOOM_SCRIPT,
              "visibleSourceHeightByFrame": heights, "leftSourceXByFrame": lefts}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"### face zoom — {'PASS' if ok else 'FAIL'}",
        "",
        f"Two people with small ({2 * head_r}px, {100 * 2 * head_r // H}% of height) faces, "
        f"A talks 0-4s then B 4-8s. The crop should tighten to {expected_h}px of source "
        f"height (2x, capped by --face-zoom-max-upscale) around the talker's face and pan "
        f"between the two zoomed crops. Plan: `{summary.get('face_zoom')} face-zoom scenes, "
        f"{summary.get('speaker_turns')} speaker-turns`.",
        "",
        "| check | result | detail |",
        "|---|---|---|",
        f"| frame count | {mark(checks['frameCount']['ok'])} | "
        f"{checks['frameCount']['actual']} / {checks['frameCount']['expected']} |",
        f"| plan | {mark(checks['plan']['ok'])} | zoom regions {zooms} |",
        f"| face placement | {mark(checks['facePlacement']['ok'])} | "
        f"face centre at {placements} of crop height (want 0.38) |",
        f"| A->B turn | {mark(checks['turn']['ok'])} | {checks['turn']['kind']}, "
        f"h {checks['turn']['heights']} |",
        f"| visible source height | {mark(checks['visibleHeight']['ok'])} | "
        f"{checks['visibleHeight']['min']}-{checks['visibleHeight']['max']}px of {H} "
        f"(want ~{expected_h}) |",
        f"| A->B pan | {mark(checks['pan']['ok'])} | {describe(checks['pan'], pan) if pan else 'n/a'} |",
        f"| debug overlay | {mark(checks['overlay']['ok'])} | {checks['overlay']['bytes']} bytes |",
    ]
    return ok, report, lines


def run(args):
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    lines = []
    all_ok = True
    for name, fn in (("transitions", run_transitions), ("speaker", run_speaker),
                     ("facezoom", run_facezoom)):
        if args.case not in ("all", name):
            continue
        try:
            ok, report, case_lines = fn(out_dir / name, args)
        except Exception:  # noqa: BLE001
            ok, report = False, {"ok": False, "error": traceback.format_exc()}
            case_lines = [f"### {name} — ERROR", "", "```", traceback.format_exc(), "```"]
        results[name] = report
        lines += case_lines + [""]
        all_ok = all_ok and ok

    header = [
        f"## Autocrop render E2E — {'PASS' if all_ok else 'FAIL'}",
        "",
        f"pan {args.pan_duration}s, zoom {args.zoom_duration}s, {args.fps}fps. "
        "Per case: `fixture.mp4` (before), `rendered.mp4` (after), `contact-sheet.jpg`, "
        "`plan.json`, `report.json`; speaker and facezoom cases also `overlay.mp4`.",
        "",
    ]
    summary_md = "\n".join(header + lines)
    (out_dir / "summary.md").write_text(summary_md)
    (out_dir / "report.json").write_text(
        json.dumps({"ok": all_ok, "cases": results}, indent=2) + "\n")
    print(summary_md)
    return all_ok


def main_():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default="autocrop-e2e")
    parser.add_argument("--case", default="all", choices=["all", "transitions", "speaker", "facezoom"])
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
