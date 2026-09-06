"""Speaker focus: face tracking, speaker-turn segmentation and debug overlay.

Phase 1 (this module as shipped) provides the plumbing that lets a scene be
split into per-speaker sub-scenes which the existing pan/zoom planner then
eases between:

    track_faces()            per-frame face tracks inside one scene (YuNet)
    score_speaking()         hook: per-track "is speaking" scores. Returns
                             None until an audio-visual scorer lands (phase 2),
                             in which case speaker focus leaves scenes as-is.
    segment_speaker_turns()  scores -> speaker turns with dwell/hysteresis
    split_scene_by_speaker() turns -> sub-scenes with TRACK targets
    render_debug_overlay()   source video with tracks, scores and framing

Everything here is CPU-only and sized for a 2 vCPU worker: face detection
runs on frames downscaled to <=640px wide, every `face_stride` frames, and
only for scenes where the scene-level analysis already found >= 2 people.
"""
import os
import urllib.request

FACE_MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                  "face_detection_yunet/face_detection_yunet_2023mar.onnx")
FACE_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
FACE_MODEL_ENV = "AUTOCROP_FACE_MODEL"
DETECT_MAX_WIDTH = 640

_face_detector = None
_face_detector_size = None


def face_model_path():
    """Locate the YuNet weights: $AUTOCROP_FACE_MODEL, else a cached download."""
    explicit = os.environ.get(FACE_MODEL_ENV)
    if explicit:
        return explicit
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "autocrop")
    path = os.path.join(cache_dir, os.path.basename(FACE_MODEL_URL))
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(FACE_MODEL_URL, tmp)
        _verify_sha256(tmp)
        os.replace(tmp, path)
    return path


def _verify_sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != FACE_MODEL_SHA256:
        os.remove(path)
        raise RuntimeError(f"face model checksum mismatch for {path}")


def get_face_detector(width, height):
    """YuNet detector sized for (width, height) input frames."""
    global _face_detector, _face_detector_size
    import cv2
    if _face_detector is None:
        _face_detector = cv2.FaceDetectorYN.create(
            face_model_path(), "", (width, height), 0.7, 0.3, 50)
        _face_detector_size = (width, height)
    if _face_detector_size != (width, height):
        _face_detector.setInputSize((width, height))
        _face_detector_size = (width, height)
    return _face_detector


def detect_faces(frame):
    """Return [x1, y1, x2, y2] face boxes in full-frame coordinates."""
    import cv2
    h, w = frame.shape[:2]
    scale = 1.0
    if w > DETECT_MAX_WIDTH:
        scale = DETECT_MAX_WIDTH / w
        frame = cv2.resize(frame, (DETECT_MAX_WIDTH, max(1, int(round(h * scale)))))
    dh, dw = frame.shape[:2]
    _, faces = get_face_detector(dw, dh).detect(frame)
    boxes = []
    if faces is not None:
        for row in faces:
            x, y, bw, bh = row[:4]
            boxes.append([int(x / scale), int(y / scale),
                          int((x + bw) / scale), int((y + bh) / scale)])
    return boxes


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _center_distance(a, b):
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def associate_tracks(tracks, boxes, frame_number, max_gap_frames,
                     iou_min=0.2, center_ratio=1.0):
    """Greedy match of detections to open tracks; unmatched boxes open tracks.

    A box matches a track when IoU >= iou_min or its centre lies within
    `center_ratio` face-widths of the track's last box (handles fast head
    movement between strided detections). Tracks unmatched for longer than
    max_gap_frames are closed.
    """
    open_tracks = [t for t in tracks
                   if frame_number - t["last"] <= max_gap_frames]
    pairs = []
    for ti, track in enumerate(open_tracks):
        last_box = track["boxes"][track["last"]]
        width = max(1, last_box[2] - last_box[0])
        for bi, box in enumerate(boxes):
            iou = _iou(last_box, box)
            dist = _center_distance(last_box, box) / width
            if iou >= iou_min or dist <= center_ratio:
                pairs.append((-iou + dist * 0.01, ti, bi))
    pairs.sort()
    used_tracks, used_boxes = set(), set()
    for _, ti, bi in pairs:
        if ti in used_tracks or bi in used_boxes:
            continue
        track = open_tracks[ti]
        track["boxes"][frame_number] = list(boxes[bi])
        track["last"] = frame_number
        used_tracks.add(ti)
        used_boxes.add(bi)
    for bi, box in enumerate(boxes):
        if bi in used_boxes:
            continue
        tracks.append({
            "id": len(tracks),
            "first": frame_number,
            "last": frame_number,
            "boxes": {frame_number: list(box)},
        })
    return tracks


def interpolate_track(track, start_frame, end_frame):
    """Fill every frame in [first, last] by linear interpolation of the box."""
    frames = sorted(track["boxes"])
    filled = {}
    for a, b in zip(frames, frames[1:]):
        box_a, box_b = track["boxes"][a], track["boxes"][b]
        for n in range(a, b):
            t = (n - a) / float(b - a)
            filled[n] = [int(round(box_a[k] + (box_b[k] - box_a[k]) * t))
                         for k in range(4)]
    filled[frames[-1]] = list(track["boxes"][frames[-1]])
    track["boxes"] = {n: box for n, box in filled.items()
                      if start_frame <= n < end_frame}
    return track


def track_faces(video_path, start_frame, end_frame, fps, face_stride=2,
                max_gap_sec=0.5, min_track_sec=0.6, detect=None):
    """Detect and track faces across one scene.

    Returns a list of tracks: {id, first, last, boxes: {frame: [x1,y1,x2,y2]}}
    with boxes filled for every frame of the track. Short-lived tracks
    (< min_track_sec) are dropped as false positives.
    """
    import cv2
    detect = detect or detect_faces
    max_gap = max(1, int(round(max_gap_sec * fps)))
    min_len = max(1, int(round(min_track_sec * fps)))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    tracks = []
    n = start_frame
    while n < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (n - start_frame) % max(1, face_stride) == 0:
            associate_tracks(tracks, detect(frame), n, max_gap)
        n += 1
    cap.release()
    kept = []
    for track in tracks:
        if track["last"] - track["first"] + 1 < min_len:
            continue
        interpolate_track(track, start_frame, end_frame)
        if track["boxes"]:
            kept.append(track)
    for i, track in enumerate(kept):
        track["id"] = i
    return kept


def track_box_at(track, frame_number):
    """Box for a frame, clamped to the track's extent (holds at the ends)."""
    boxes = track["boxes"]
    if frame_number in boxes:
        return boxes[frame_number]
    if frame_number < track["first"]:
        return boxes[min(boxes)]
    return boxes[max(boxes)]


def median_box(track, start_frame, end_frame):
    import numpy as np
    rows = [boxes for n, boxes in track["boxes"].items()
            if start_frame <= n < end_frame]
    if not rows:
        rows = [track_box_at(track, start_frame)]
    arr = np.array(rows, dtype=float)
    return [int(round(v)) for v in np.median(arr, axis=0)]


def score_speaking(video_path, scene, tracks, fps):
    """Hook for the audio-visual active speaker model (phase 2).

    Must return {track_id: sequence of per-frame scores in [0, 1]} aligned to
    scene['start_frame']..scene['end_frame'], or None when no scorer is
    available. Phase 1 has no scorer, so speaker focus keeps every scene as
    the scene-level analysis framed it.
    """
    return None


def _smooth(values, window):
    import numpy as np
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size == 0:
        return arr
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(arr, (pad, window - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:arr.size]


def segment_speaker_turns(scores, start_frame, end_frame, fps,
                          min_dwell_sec=1.2, on_threshold=0.5, margin=0.15,
                          smooth_sec=0.3):
    """Turn per-track speaking scores into speaker segments.

    Per frame the candidate is:
      - the top-scoring track when it is >= on_threshold and beats the
        runner-up by `margin` ("one clear speaker"),
      - 'group' when two or more tracks are >= on_threshold without a clear
        winner (crosstalk),
      - None when nobody is speaking (hold whatever was framed).
    A candidate has to persist for min_dwell_sec before the framing follows
    it, and a new segment lasts at least min_dwell_sec; interjections shorter
    than that never move the frame.

    Returns [{start_frame, end_frame, speaker: track_id|'group', confidence}]
    covering [start_frame, end_frame). Leading frames before the first
    decision inherit it.
    """
    import numpy as np
    length = end_frame - start_frame
    if length <= 0 or not scores:
        return [{"start_frame": start_frame, "end_frame": end_frame,
                 "speaker": "group", "confidence": 0.0}]
    ids = sorted(scores)
    window = max(1, int(round(smooth_sec * fps)))
    matrix = np.vstack([_smooth(np.asarray(scores[i], dtype=float)[:length], window)
                        if len(scores[i]) >= length else
                        _smooth(np.pad(np.asarray(scores[i], dtype=float),
                                       (0, length - len(scores[i]))), window)
                        for i in ids])
    dwell = max(1, int(round(min_dwell_sec * fps)))

    candidates = []
    for n in range(length):
        col = matrix[:, n]
        order = np.argsort(-col)
        top, top_score = order[0], col[order[0]]
        second = col[order[1]] if len(order) > 1 else 0.0
        if top_score < on_threshold:
            candidates.append((None, 0.0))
        elif second >= on_threshold and top_score - second < margin:
            candidates.append(("group", float(top_score)))
        else:
            candidates.append((ids[top], float(top_score - second)))

    segments = []
    current = None
    current_start = 0
    pending = None
    pending_since = 0
    for n, (cand, conf) in enumerate(candidates):
        if cand is None or cand == current:
            pending = None
            continue
        if cand != pending:
            pending, pending_since = cand, n
        held_long_enough = current is None or (n - current_start) >= dwell
        if n - pending_since + 1 >= dwell and held_long_enough:
            switch_at = pending_since
            if current is not None:
                segments.append({"start_frame": start_frame + current_start,
                                 "end_frame": start_frame + switch_at,
                                 "speaker": current, "confidence": 0.0})
            current, current_start = cand, switch_at
            pending = None
    if current is None:
        current = "group"
        current_start = 0
    segments.append({"start_frame": start_frame + current_start,
                     "end_frame": end_frame, "speaker": current,
                     "confidence": 0.0})
    segments[0]["start_frame"] = start_frame

    for seg in segments:
        a, b = seg["start_frame"] - start_frame, seg["end_frame"] - start_frame
        confs = [c for (s, c) in candidates[a:b] if s == seg["speaker"]]
        seg["confidence"] = round(float(np.mean(confs)), 3) if confs else 0.0
    return segments


def split_scene_by_speaker(scene, segments, tracks, frame_height,
                           decide_strategy):
    """Expand one scene into per-segment sub-scenes for the planner.

    Speaker segments become TRACK scenes targeting the median face box of
    that track over the segment; 'group' segments keep the scene-level
    strategy/target. Each sub-scene carries `speaker` metadata and, for
    all but the first, boundary_source='speaker-turn' so the planner and the
    summary can tell these visually-continuous boundaries from real cuts.
    """
    by_id = {t["id"]: t for t in tracks}
    out = []
    for k, seg in enumerate(segments):
        sub = dict(scene)
        sub["start_frame"] = seg["start_frame"]
        sub["end_frame"] = seg["end_frame"]
        sub["start_seconds"] = scene["start_seconds"] + (
            (seg["start_frame"] - scene["start_frame"])
            * (scene["end_seconds"] - scene["start_seconds"])
            / max(1, scene["end_frame"] - scene["start_frame"]))
        sub["end_seconds"] = scene["start_seconds"] + (
            (seg["end_frame"] - scene["start_frame"])
            * (scene["end_seconds"] - scene["start_seconds"])
            / max(1, scene["end_frame"] - scene["start_frame"]))
        if seg["speaker"] == "group" or seg["speaker"] not in by_id:
            strategy, target = decide_strategy(scene["analysis"], frame_height)
            sub["strategy"], sub["target_box"] = strategy, target
            sub["speaker"] = {"kind": "group", "confidence": seg["confidence"]}
        else:
            track = by_id[seg["speaker"]]
            sub["strategy"] = "TRACK"
            sub["target_box"] = median_box(track, seg["start_frame"], seg["end_frame"])
            sub["speaker"] = {"kind": "track", "track_id": track["id"],
                              "confidence": seg["confidence"]}
        sub["boundary_source"] = "speaker-turn" if k > 0 else scene.get("boundary_source")
        out.append(sub)
    return out


def apply_speaker_focus(video_path, scenes_analysis, fps, frame_height,
                        decide_strategy, min_dwell_sec=1.2, face_stride=2,
                        log=print):
    """Run tracking + scoring on multi-person scenes and split them by speaker.

    Returns (new_scenes_analysis, debug) where debug maps original scene
    index -> {tracks, scores, segments} for the overlay renderer.
    """
    out, debug = [], {}
    for idx, scene in enumerate(scenes_analysis):
        people = len(scene.get("analysis") or [])
        if people < 2:
            out.append(scene)
            continue
        tracks = track_faces(video_path, scene["start_frame"], scene["end_frame"],
                             fps, face_stride=face_stride)
        if len(tracks) < 2:
            log(f"   speaker-focus: scene {idx + 1} has {people} people but "
                f"{len(tracks)} face track(s); keeping scene-level framing")
            scene["speaker"] = {"kind": "unsplit", "reason": "faces", "tracks": len(tracks)}
            out.append(scene)
            debug[idx] = {"tracks": tracks, "scores": None, "segments": None,
                          "start_frame": scene["start_frame"]}
            continue
        scores = score_speaking(video_path, scene, tracks, fps)
        if not scores:
            log(f"   speaker-focus: scene {idx + 1} tracked {len(tracks)} faces "
                f"but no speaker scorer is available; keeping scene-level framing")
            scene["speaker"] = {"kind": "unsplit", "reason": "no-scorer", "tracks": len(tracks)}
            out.append(scene)
            debug[idx] = {"tracks": tracks, "scores": None, "segments": None,
                          "start_frame": scene["start_frame"]}
            continue
        segments = segment_speaker_turns(scores, scene["start_frame"],
                                         scene["end_frame"], fps,
                                         min_dwell_sec=min_dwell_sec)
        subs = split_scene_by_speaker(scene, segments, tracks, frame_height,
                                      decide_strategy)
        log(f"   speaker-focus: scene {idx + 1} -> {len(subs)} segment(s) from "
            f"{len(tracks)} face track(s)")
        out.extend(subs)
        debug[idx] = {"tracks": tracks, "scores": scores, "segments": segments,
                      "start_frame": scene["start_frame"]}
    return out, debug


def render_debug_overlay(video_path, out_path, scenes_analysis, debug,
                         frame_width, frame_height, fps, resolve_region):
    """Write the source video with tracks, speaker scores and framing drawn on.

    Boxes: face tracks (yellow; green when the current sub-scene targets that
    track), the active crop region (magenta) and a text strip with the scene
    strategy, boundary kind and per-track scores for the frame.
    """
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (frame_width, frame_height))
    scene_of_frame = []
    for si, scene in enumerate(scenes_analysis):
        scene_of_frame.append((scene["start_frame"], scene["end_frame"], si))
    debug_of_frame = {}
    for orig_idx, info in debug.items():
        for track in info["tracks"]:
            for n in track["boxes"]:
                debug_of_frame.setdefault(n, []).append((orig_idx, track))
    n = 0
    si = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while si + 1 < len(scene_of_frame) and n >= scene_of_frame[si][1]:
            si += 1
        scene = scenes_analysis[scene_of_frame[si][2]] if scene_of_frame else None
        target_track = None
        if scene and scene.get("speaker", {}).get("kind") == "track":
            target_track = scene["speaker"]["track_id"]
        for orig_idx, track in debug_of_frame.get(n, []):
            box = track["boxes"][n]
            color = (0, 220, 0) if track["id"] == target_track else (0, 220, 220)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"t{track['id']}"
            scores = debug[orig_idx]["scores"]
            if scores and track["id"] in scores:
                rel = n - debug[orig_idx]["start_frame"]
                seq = scores[track["id"]]
                if 0 <= rel < len(seq):
                    label += f" {float(seq[rel]):.2f}"
            cv2.putText(frame, label, (box[0], max(14, box[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if scene:
            x, w = resolve_region(scene, n, frame_width, frame_height)
            cv2.rectangle(frame, (int(x), 0), (int(x + w) - 1, frame_height - 1),
                          (255, 0, 255), 2)
            kind = scene.get("boundary_kind", "")
            spk = scene.get("speaker") or {}
            text = (f"f{n} {scene['strategy']} {kind} "
                    f"{spk.get('kind', '')}"
                    f"{' t' + str(spk['track_id']) if 'track_id' in spk else ''}")
            cv2.rectangle(frame, (0, frame_height - 28), (frame_width, frame_height),
                          (0, 0, 0), -1)
            cv2.putText(frame, text, (8, frame_height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    return n

