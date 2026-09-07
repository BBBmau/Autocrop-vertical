#!/usr/bin/env python3
"""Speaker-focus accuracy on real footage: prefill labels, review, evaluate.

Workflow
  1. prefill   run scene detection + face tracking + Light-ASD on a clip and
               write a DRAFT label file with the predicted speaker timeline,
               plus a review video (tracks, scores, predicted speaker).
  2. review    watch the review video, fix the draft's `speaker` entries
               (x/y are fractions of the frame where the speaker's face is;
               null means nobody on screen is speaking).
  3. evaluate  re-run the pipeline and score predictions against the labels:
               speaker accuracy over labelled speech frames, switch counts,
               and per-segment detail. Non-zero exit when below --min-accuracy.

Label file (JSON):
  {"clip": "name.mp4",
   "segments": [{"start": 0.0, "end": 3.4, "speaker": {"x": 0.31, "y": 0.22}},
                {"start": 3.4, "end": 5.0, "speaker": null}, ...]}

Usage
  speaker_eval.py prefill  CLIP.mp4 --labels LABELS.json --review REVIEW.mp4
  speaker_eval.py evaluate CLIP.mp4 --labels LABELS.json [--report REPORT.json]
  speaker_eval.py evaluate-set DIR   # every DIR/*.mp4 with DIR/*.json labels

Requires the speaker-focus extras: opencv, scenedetect, torch,
python_speech_features, ffmpeg. YOLO is not needed: a scene counts as
multi-person when it has >= 2 face tracks.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import main as autocrop  # noqa: E402
import speaker  # noqa: E402


def analyze_clip(clip, min_dwell=1.0, overlap="loudest", face_stride=2,
                 scene_threshold=27, log=print):
    """Scenes -> face tracks -> speaker scores -> segments, without YOLO."""
    width, height, fps = autocrop.get_video_properties(str(clip))
    scenes, _ = autocrop.detect_scenes(str(clip), threshold=scene_threshold)
    scenes_analysis = []
    for start, end in scenes:
        scenes_analysis.append({
            "start_frame": start.get_frames(), "end_frame": end.get_frames(),
            "start_seconds": start.get_seconds(), "end_seconds": end.get_seconds(),
            "analysis": [], "strategy": "LETTERBOX", "target_box": None,
        })
    # Stand in for the YOLO person count with face tracks so apply_speaker_focus
    # treats scenes with >= 2 faces as multi-person.
    tracks_by_scene = {}
    for idx, scene in enumerate(scenes_analysis):
        tracks = speaker.track_faces(str(clip), scene["start_frame"], scene["end_frame"],
                                     fps, face_stride=face_stride)
        tracks_by_scene[idx] = tracks
        scene["analysis"] = [
            {"person_box": _person_from_face(speaker.median_box(t, scene["start_frame"],
                                                                scene["end_frame"]), height),
             "face_box": None, "motion": 0.0}
            for t in tracks]
        if len(tracks) == 1:
            scene["strategy"], scene["target_box"] = "TRACK", scene["analysis"][0]["person_box"]
        elif len(tracks) > 1:
            scene["strategy"], scene["target_box"] = autocrop.decide_cropping_strategy(
                scene["analysis"], height)
    original_track_faces = speaker.track_faces
    speaker.track_faces = lambda path, s, e, f, **kw: tracks_by_scene[
        next(i for i, sc in enumerate(scenes_analysis) if sc["start_frame"] == s)]
    try:
        split, debug = speaker.apply_speaker_focus(
            str(clip), scenes_analysis, fps, height,
            lambda analysis, h: autocrop.decide_cropping_strategy(analysis, h),
            min_dwell_sec=min_dwell, face_stride=face_stride, overlap=overlap, log=log)
    finally:
        speaker.track_faces = original_track_faces
    autocrop.plan_face_zoom(split, width, height)
    autocrop.plan_pan_transitions(str(clip), split, width, height, fps)
    return {"width": width, "height": height, "fps": fps, "scenes": split,
            "debug": debug, "tracks_by_scene": tracks_by_scene}


def _person_from_face(face, height):
    x1, y1, x2, y2 = face
    w = x2 - x1
    return [int(x1 - 1.5 * w), int(y1 - 0.5 * w), int(x2 + 1.5 * w), min(height, int(y2 + 6 * w))]


def predicted_timeline(result):
    """[(start_sec, end_sec, (x_frac, y_frac) | None)] for framed speakers."""
    fps, w, h = result["fps"], result["width"], result["height"]
    out = []
    for scene in result["scenes"]:
        spk = scene.get("speaker") or {}
        if spk.get("kind") == "track" and scene.get("target_box"):
            b = scene["target_box"]
            pos = (((b[0] + b[2]) / 2) / w, ((b[1] + b[3]) / 2) / h)
        else:
            pos = None
        out.append((scene["start_frame"] / fps, scene["end_frame"] / fps, pos))
    return out


def write_draft_labels(clip, result, path):
    segments = []
    for start, end, pos in predicted_timeline(result):
        segments.append({"start": round(start, 3), "end": round(end, 3),
                         "speaker": None if pos is None else
                         {"x": round(pos[0], 3), "y": round(pos[1], 3)}})
    data = {"clip": Path(clip).name, "draft": True,
            "note": "Review against the review video. x/y = speaker face centre as frame "
                    "fractions; speaker null = nobody on screen is speaking. Delete 'draft' "
                    "when verified.",
            "segments": segments}
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
    return data


def frame_truth(labels, n_frames, fps):
    """Per frame: (x, y) of labelled speaker, None for no speaker, 'unlabelled'."""
    truth = ["unlabelled"] * n_frames
    for seg in labels["segments"]:
        a, b = int(round(seg["start"] * fps)), int(round(seg["end"] * fps))
        spk = seg.get("speaker")
        value = None if spk is None else (spk["x"], spk["y"])
        for n in range(max(0, a), min(n_frames, b)):
            truth[n] = value
    return truth


def evaluate(clip, labels, result, tolerance=0.08):
    fps, w, h = result["fps"], result["width"], result["height"]
    n_frames = max(s["end_frame"] for s in result["scenes"])
    truth = frame_truth(labels, n_frames, fps)
    pred = [None] * n_frames
    for start, end, pos in predicted_timeline(result):
        for n in range(int(round(start * fps)), min(n_frames, int(round(end * fps)))):
            pred[n] = pos
    speech = correct = wrong_face = missed = 0
    false_focus = 0
    for n in range(n_frames):
        t = truth[n]
        if t == "unlabelled":
            continue
        p = pred[n]
        if t is None:
            # Nobody speaking: framing anyone specific is not wrong per se
            # (we hold the last speaker), so only count for information.
            if p is not None:
                false_focus += 1
            continue
        speech += 1
        if p is None:
            missed += 1
        elif abs(p[0] - t[0]) <= tolerance and abs(p[1] - t[1]) <= tolerance * (w / h):
            correct += 1
        else:
            wrong_face += 1
    pred_switches = sum(1 for s in result["scenes"] if s.get("boundary_source") == "speaker-turn")
    label_switches = 0
    last = "unset"
    for seg in labels["segments"]:
        spk = seg.get("speaker")
        key = None if spk is None else (round(spk["x"], 1), round(spk["y"], 1))
        if key is not None and last != "unset" and key != last:
            label_switches += 1
        if key is not None:
            last = key
    return {
        "clip": Path(clip).name,
        "frames": n_frames,
        "speechFrames": speech,
        "correct": correct,
        "wrongFace": wrong_face,
        "missed": missed,
        "holdWhileSilent": false_focus,
        "accuracy": round(correct / speech, 3) if speech else None,
        "predictedSwitches": pred_switches,
        "labelledSwitches": label_switches,
        "predicted": [{"start": round(a, 2), "end": round(b, 2),
                       "speaker": None if p is None else {"x": round(p[0], 3), "y": round(p[1], 3)}}
                      for a, b, p in predicted_timeline(result)],
    }


def cmd_prefill(args):
    result = analyze_clip(args.clip, min_dwell=args.min_dwell, overlap=args.overlap)
    labels = write_draft_labels(args.clip, result, args.labels)
    if args.review:
        speaker.render_debug_overlay(str(args.clip), str(args.review), result["scenes"],
                                     result["debug"], result["width"], result["height"],
                                     result["fps"], autocrop.resolve_frame_region)
    print(f"draft labels -> {args.labels} ({len(labels['segments'])} segments)")
    for seg in labels["segments"]:
        print(f"  {seg['start']:7.2f}-{seg['end']:7.2f}s  {seg['speaker']}")
    if args.review:
        print(f"review video -> {args.review}")


def cmd_evaluate(args):
    labels = json.loads(Path(args.labels).read_text())
    result = analyze_clip(args.clip, min_dwell=args.min_dwell, overlap=args.overlap,
                          log=lambda *_: None)
    report = evaluate(args.clip, labels, result, tolerance=args.tolerance)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print_report([report], labels_draft=[labels.get("draft", False)])
    ok = report["accuracy"] is not None and report["accuracy"] >= args.min_accuracy
    sys.exit(0 if ok else 1)


def cmd_evaluate_set(args):
    clips = sorted(p for p in Path(args.dir).glob("*.mp4") if not p.name.startswith("."))
    reports, drafts = [], []
    for clip in clips:
        label_path = clip.with_suffix(".json")
        if not label_path.exists():
            print(f"skip {clip.name}: no labels")
            continue
        labels = json.loads(label_path.read_text())
        result = analyze_clip(clip, min_dwell=args.min_dwell, overlap=args.overlap,
                              log=lambda *_: None)
        reports.append(evaluate(clip, labels, result, tolerance=args.tolerance))
        drafts.append(labels.get("draft", False))
    if args.report:
        Path(args.report).write_text(json.dumps(reports, indent=2) + "\n")
    print_report(reports, drafts)
    speech = sum(r["speechFrames"] for r in reports)
    correct = sum(r["correct"] for r in reports)
    overall = correct / speech if speech else 0.0
    verified = [r for r, d in zip(reports, drafts) if not d]
    v_speech = sum(r["speechFrames"] for r in verified)
    v_correct = sum(r["correct"] for r in verified)
    print(f"\noverall agreement {overall:.3f} over {speech} labelled speech frames "
          f"in {len(reports)} clips ({len(verified)} verified)")
    if not verified:
        print("no verified labels yet: reporting only, not gating "
              "(remove 'draft' from a label file once reviewed)")
        sys.exit(0 if reports else 1)
    v_acc = v_correct / v_speech if v_speech else 0.0
    print(f"verified accuracy {v_acc:.3f} over {v_speech} speech frames (min {args.min_accuracy})")
    sys.exit(0 if v_acc >= args.min_accuracy else 1)


def print_report(reports, labels_draft):
    print("| clip | accuracy | correct | wrong face | missed | switches pred/label | labels |")
    print("|---|---|---|---|---|---|---|")
    for r, draft in zip(reports, labels_draft):
        acc = "n/a" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
        print(f"| {r['clip']} | {acc} | {r['correct']} | {r['wrongFace']} | {r['missed']} | "
              f"{r['predictedSwitches']}/{r['labelledSwitches']} | "
              f"{'DRAFT' if draft else 'verified'} |")


def main_():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--min-dwell", type=float, default=1.0)
    common.add_argument("--overlap", default="loudest", choices=["loudest", "group"])
    common.add_argument("--tolerance", type=float, default=0.08,
                        help="max |x| distance (frame fraction) between predicted and "
                             "labelled face centre to count as correct")
    common.add_argument("--min-accuracy", type=float, default=0.85)
    p = sub.add_parser("prefill", parents=[common])
    p.add_argument("clip")
    p.add_argument("--labels", required=True)
    p.add_argument("--review")
    p.set_defaults(fn=cmd_prefill)
    e = sub.add_parser("evaluate", parents=[common])
    e.add_argument("clip")
    e.add_argument("--labels", required=True)
    e.add_argument("--report")
    e.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("evaluate-set", parents=[common])
    s.add_argument("dir")
    s.add_argument("--report")
    s.set_defaults(fn=cmd_evaluate_set)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main_()
