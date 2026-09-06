# AutoCrop-Vertical: Smart Video Cropper for Social Media

![Demo of AutoCrop-Vertical](https://github.com/kamilstanuch/Autocrop-vertical/blob/main/churchil_queen_vertical_short.gif?raw=true)

Automatically converts horizontal videos into vertical format for TikTok, Instagram Reels, and YouTube Shorts.

Instead of a static center crop, the script analyzes each scene using AI (YOLOv8), detects people, and decides whether to crop tightly on the subjects or letterbox to preserve the full shot.

---

### Quick Start

```bash
git clone https://github.com/kamilstanuch/AutoCrop-Vertical.git
cd AutoCrop-Vertical
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 main.py -i video.mp4 -o vertical.mp4
```

The `yolov8n.pt` model weights are downloaded automatically on first run.

**Prerequisites:** Python 3.8+ and [FFmpeg](https://ffmpeg.org/) (`ffmpeg` + `ffprobe`) in your PATH.

---

### Speaker Focus

```bash
autocrop -i in.mp4 -o out.mp4 --speaker-focus auto --speaker-min-dwell 1.2 \
  --debug-overlay overlay.mp4 --plan-json plan.json
```

`speaker.py` tracks faces per frame in multi-person scenes (OpenCV YuNet),
`asd.py` scores each face per frame with **Light-ASD** (audio-visual active
speaker detection: does this mouth match the audio?), and the scores become
speaker turns (dwell + hysteresis) that split the scene into sub-scenes the
pan/zoom planner eases between. When several people talk at once the frame
follows the face whose speech dominates the audio (`--speaker-overlap
loudest`, default) or widens to the group (`group`). A lone face among
several YOLO bodies (audience, bystanders) is tracked when it is the one
talking instead of letterboxing everyone. Scenes keep their scene-level
framing when there is no audio stream or the speaker extras are missing.

Requirements: `torch` (CPU is fine, ~1s per 6s of one face), `python_speech_features`,
`scipy`, `ffmpeg`. Model weights (YuNet 0.2MB, Light-ASD 4MB) are fetched
once to `~/.cache/autocrop/` and checksum-verified (override with
`AUTOCROP_FACE_MODEL` / `AUTOCROP_ASD_MODEL`).

**Accuracy on real footage** — `scripts/speaker_eval.py` turns clips into a
labelled eval set and scores the pipeline against it:

```bash
# 1. draft labels + a review video (tracks, scores, chosen speaker) per clip
python scripts/speaker_eval.py prefill clip.mp4 --labels clip.json --review clip_review.mp4
# 2. watch the review video, fix `speaker` x/y (frame fractions) or null, drop "draft"
# 3. gate: accuracy over labelled speech frames; drafts only report
python scripts/speaker_eval.py evaluate-set eval-set/ --min-accuracy 0.85
```

CI downloads the `eval-set-v1` release asset (public-domain NASA briefing
clips, 6–7 faces at the table) and publishes the table in the job summary.

### Render E2E (CI)

`scripts/e2e_render.py` is the release gate. It builds synthetic H.264
fixtures — `transitions/` (wide shot → speaker left → speaker right → wide
shot) and `speaker/` (two people, scripted speaker turns) — runs the real
`autocrop` CLI on them with only the ML detectors stubbed, and asserts that
the rendered output zooms, pans and follows speaker turns gradually instead
of snapping.
It writes `fixture.mp4` (before), `rendered.mp4` (after), a `contact-sheet.jpg`
of source-vs-output frames around each boundary, `plan.json`, `report.json`
and `summary.md`. The `CI` workflow runs it on every PR and uploads those
files as the `autocrop-e2e-<sha>` artifact; `viral-clip-extractor` runs the
same script at its pinned ref before every deploy.

```bash
pip install opencv-python-headless "scenedetect[opencv]" numpy tqdm python_speech_features scipy
pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 scripts/e2e_render.py --output-dir autocrop-e2e
```

The speaker case also smoke-tests the real Light-ASD model (weights, MFCC,
crops, 25fps resampling, inference) on the fixture's synthetic audio track.

### Local Pan Lab

Use the lightweight pan lab when iterating on transition timing. It skips
YOLO and scene detection but then runs the production code: the same
`plan_pan_transitions` decides where pans and zooms happen and the same
`render_output_frame` produces every pixel that the `autocrop` CLI writes.
It renders four durations side by side and creates a contact sheet.

```bash
# First run creates .pan-lab-venv with only OpenCV + NumPy.
./scripts/pan-lab

# Zoom from the full letterboxed frame onto a subject (or back out).
./scripts/pan-lab --transition zoom-in
./scripts/pan-lab --transition zoom-out

# Hand-specify a transition on a real clip without rendering all of it.
./scripts/pan-lab \
  --input /path/to/source.mp4 \
  --boundary-sec 42.8 \
  --from-x 360 \
  --to-x 1480 \
  --durations 0,0.25,0.4,0.65

# Replay the real plan for a clip. Run analysis once (YOLO, scene detection),
# then iterate on timing in seconds.
python3 main.py -i source.mp4 -o /dev/null --plan-only --plan-json plan.json
./scripts/pan-lab --input source.mp4 --plan plan.json --durations 0,0.3,0.4,0.6
```

With `--plan`, the boundary defaults to the first planned transition; pass
`--boundary-sec` to inspect a different one. Each `--durations` value is
applied to pans and zooms alike. `report.json` records the transition
summary (`pan / zoom / hold / layout-switch`) for every variant, so a plan
that yields zero transitions is visible immediately.

Artifacts are written to `pan-lab-output/`:

- `comparison.mp4` — all timing variants playing together
- `contact-sheet.jpg` — transition frames aligned by timestamp
- `report.json` — exact source region `[x, w]` for every rendered frame
- `pan_*s.mp4` — individual variants

For real clips, the lab renders only 1.5 seconds on either side of the
boundary by default. Adjust that with `--window-sec`.

---

### Usage Examples

```bash
# Basic — 9:16 vertical, balanced quality
python3 main.py -i video.mp4 -o vertical.mp4

# Instagram feed (4:5) with high quality
python3 main.py -i video.mp4 -o vertical.mp4 --ratio 4:5 --quality high

# Fast encode, square format
python3 main.py -i video.mp4 -o vertical.mp4 --ratio 1:1 --quality fast

# Preview the processing plan without encoding
python3 main.py -i video.mp4 -o vertical.mp4 --plan-only

# Full control over encoding parameters
python3 main.py -i video.mp4 -o vertical.mp4 --crf 20 --preset medium

# Use hardware encoder (macOS VideoToolbox / NVIDIA NVENC)
python3 main.py -i video.mp4 -o vertical.mp4 --encoder hw

# Maximum accuracy scene detection (slower)
python3 main.py -i video.mp4 -o vertical.mp4 --frame-skip 0
```

---

### All Flags

**Output:**

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | *(required)* | Path to input video |
| `-o`, `--output` | *(required)* | Path to output video (`.mp4` appended if no extension) |
| `--ratio` | `9:16` | Output aspect ratio. Examples: `9:16`, `4:5`, `1:1` |

**Encoding quality:**

| Flag | Default | Description |
|------|---------|-------------|
| `--quality` | `balanced` | Preset: `fast`, `balanced`, or `high` (see table below) |
| `--encoder` | `auto` | `auto` = libx264 (software), `hw` = hardware if available, or explicit name like `h264_videotoolbox` |
| `--crf` | *(from quality)* | Override CRF directly, 0-51 lower = better (libx264 only) |
| `--preset` | *(from quality)* | Override x264 preset directly: `ultrafast`..`veryslow` (libx264 only) |

**Quality presets (libx264):**

| `--quality` | CRF | Preset | Typical use |
|-------------|-----|--------|-------------|
| `fast` | 28 | veryfast | Quick previews, drafts |
| `balanced` | 23 | fast | Good quality, reasonable speed |
| `high` | 18 | slow | Best quality, largest file, slowest |

**Scene detection tuning:**

| Flag | Default | Description |
|------|---------|-------------|
| `--frame-skip` | `0` | Frames to skip during scene detection. `0` = every frame (most accurate). `1` = every other frame (~2x faster). Higher = faster but may miss cuts |
| `--downscale` | `0` (auto) | Downscale factor for scene detection. `0` = auto. `2`-`4` = faster but may miss subtle cuts |
| `--pan-duration` | `0.4` | Seconds used for an eased pan when the TRACK crop center jumps (including speaker switches). `0` disables panning |
| `--zoom-duration` | = `--pan-duration` | Seconds used for an eased zoom when the layout switches between the full letterboxed frame and a tracked crop (LETTERBOX↔TRACK). `0` restores instant layout switches |

**Other:**

| Flag | Default | Description |
|------|---------|-------------|
| `--plan-only` | off | Run scene detection + analysis only, print the plan, exit without encoding |
| `--plan-json PATH` | off | Write the scene/pan plan to JSON (works with or without `--plan-only`). Replay it with `scripts/pan-lab --plan` |

---

### Key Features

*   **Content-Aware Cropping:** YOLOv8 detects people and centers the vertical frame on them.
*   **Automatic Letterboxing:** When people are too spread out for a vertical crop, black bars are added to preserve the full shot.
*   **Scene-by-Scene Processing:** Decisions are made per scene for consistent, logical edits.
*   **Smooth Subject Pans:** TRACK-to-TRACK crop-center jumps (including speaker switches) ease over `--pan-duration`.
*   **Smooth Zooms:** switching between the full letterboxed frame and a tracked crop eases over `--zoom-duration` — the region tightens onto the subject (or widens back out) while the letterbox bars shrink (or grow), instead of swapping layouts in one frame.
*   **Native Resolution:** Output height matches the source to prevent quality loss from upscaling.
*   **Frame-Accurate Processing:** Every frame is processed individually with the correct per-scene strategy — no timestamp rounding or scene boundary drift.
*   **Hardware Encoder Support:** Optional `--encoder hw` auto-detects VideoToolbox (macOS) or NVENC (NVIDIA) with automatic fallback to libx264.
*   **VFR Handling:** Variable frame rate sources are automatically normalized before processing.
*   **Audio Sync:** Non-zero stream start times are detected and compensated to keep audio/video aligned.

---

### How It Works

```
Input Video
    |
    v
+-------------------------------+
| 1. Scene Detection            |  PySceneDetect splits the video into scenes
|    (--frame-skip, --downscale)|
+---------------+---------------+
                |
                v
+-------------------------------+
| 2. Content Analysis           |  YOLOv8 detects people in each scene's
|    (middle frame per scene)   |  middle frame; Haar cascade finds faces
+---------------+---------------+
                |
                v
+-------------------------------+
| 3. Strategy Decision          |  Per scene: TRACK (crop on subject)
|                               |  or LETTERBOX (scale + black bars)
+---------------+---------------+
                |
                v
+-------------------------------+
| 4. Frame Processing           |  Per-frame crop/scale/pad via OpenCV
|    (--quality, --encoder)     |  piped to FFmpeg for encoding
+---------------+---------------+
                |
                v
+-------------------------------+
| 5-6. Audio extract + merge    |  Audio synced with start-time offset
+---------------+---------------+
                |
                v
          Output Video
```

Steps 1-3 are the "planning" phase (Python + AI). Step 4 applies the plan frame-by-frame and encodes via FFmpeg.

---

### Performance

Benchmarks on Apple M1 MacBook Pro (AC power):

| Resolution | Duration | Total time | Speed |
|-----------|----------|-----------|-------|
| 1280x720 | 49s | ~6s | 8.3x real-time |
| 1920x1080 | 12 min | ~51s | 13.7x real-time |

Scene detection is the dominant bottleneck (~50% of total time).

---

### Technical Details

This script is built on a pipeline that uses specialized libraries for each step:

*   **Core Libraries:**
    *   `PySceneDetect`: For accurate, content-aware scene cut detection.
    *   `Ultralytics (YOLOv8)`: For fast and reliable person detection.
    *   `OpenCV`: Used for frame manipulation, face detection (as a fallback), and reading video properties.
    *   `FFmpeg` / `ffprobe`: The backbone of video encoding, audio extraction, and media stream analysis.
    *   `tqdm`: For clean and informative progress bars in the console.

*   **Processing Pipeline:**
    1.  **(Pre-processing)** If the source is VFR, it is normalized to constant frame rate.
    2.  `PySceneDetect` scans the video and returns a list of scene timestamps.
    3.  For each scene, `OpenCV` extracts a sample frame and `YOLOv8` detects people in it.
    4.  A set of rules determines the strategy (`TRACK` or `LETTERBOX`) for each scene based on the number and position of detected people.
    5.  OpenCV reads every frame sequentially. Each frame is cropped/resized (TRACK) or scaled/padded (LETTERBOX) according to its scene's strategy, then piped as raw pixels to FFmpeg for encoding. This frame-by-frame approach guarantees frame-accurate scene boundaries with no timestamp rounding errors.
    6.  Audio is extracted separately (with start-time offset correction), then merged with the processed video.

---

### Changelog

#### v1.8.0 — Speaker focus: audio-visual speaker scoring (phase 2)

*   **Light-ASD scorer** (`asd.py`, MIT, ~1M params, CPU): `speaker.score_speaking` now returns real per-frame P(speaking) per face track. Audio is pulled per scene with ffmpeg (16kHz mono), MFCC'd at 100Hz; each track's face crops are resampled to 25fps and scored in 6s chunks. Digital silence and off-screen frames are forced to 0. ~1s per 6s of one face on a laptop CPU; a 20s clip with a 7-face wide shot renders in ~14s end to end.
*   **`--speaker-overlap loudest|group`** (default `loudest`): during crosstalk the frame follows the face whose speech best matches the audio instead of widening to the group. Flapping leads never move the frame (dwell still applies).
*   **Lone speaking face**: a scene YOLO letterboxed because it saw several bodies (audience heads, bystanders) but with exactly one face track is switched to TRACK on that face when it is speaking ≥50% of its on-screen time. A silent lone face keeps the letterbox (e.g. b-roll with narration).
*   **Real-footage eval** (`scripts/speaker_eval.py`): `prefill` drafts a labelled speaker timeline + review video per clip, `evaluate`/`evaluate-set` score the pipeline against labels (accuracy over labelled speech frames, wrong-face/missed counts, switch counts). CI runs it on the `eval-set-v1` release asset; draft labels report only, verified labels gate at `--min-accuracy`.
*   **E2E**: fixtures now carry an audio track; the speaker case smoke-tests the real model and pins the `group` policy so the widen/narrow path stays covered. `test_asd.py` covers crops, 30→25fps resampling, padding and the silence gate.
*   Dependencies: `python_speech_features`, `scipy` (torch was already required by YOLO).

#### v1.7.0 — Speaker focus foundation (phase 1)

*   **`--speaker-focus auto`** (default `off`): in scenes with 2+ people, faces are tracked per frame (OpenCV YuNet, CPU, ~7ms/frame at 640px) and the scene is split at speaker turns so the crop pans to whoever is talking, widens to the group during crosstalk, and holds through silence. Turns need `--speaker-min-dwell` seconds (default 1.2) before the frame follows, so interjections don't move it. This release ships the plumbing only: `speaker.score_speaking` is a hook that returns `None` until the audio-visual scorer lands (phase 2), so production framing is unchanged with the flag on or off.
*   **`--debug-overlay PATH`**: writes the source video with face tracks, speaker scores, the active crop region and scene/boundary labels drawn on.
*   **Fix: cut-free clips are now cropped.** PySceneDetect returned no scenes for a clip without cuts and the CLI copied the input through unchanged, leaving steady single-shot clips horizontal. `detect_scenes` now returns one whole-video scene.
*   Plan JSON: scenes gain `boundary_source` (`speaker-turn`) and `speaker`; summary gains `speaker_turns`; the `Transitions:` line reports `N speaker-turns`.
*   Render E2E gains a `speaker/` case (two people, scripted turns incl. an ignored interjection and crosstalk) that checks segmentation, transition kinds and timing, uploads `overlay.mp4`, and smoke-loads the real face detector.

### v1.6.1 — Render E2E hardening

*   **CI-only change.** `scripts/e2e_render.py` judges pans/zooms by the largest per-frame step relative to total travel (a snap is 1.0, an eased transition ≈0.15) and reads crop position from a median patch, so decoded 4:2:0 chroma noise on Linux ffmpeg builds no longer produces false failures. Artifact names are safe for PR runs.

### v1.6.0 — Smooth zooms between letterbox and tracked crops

*   **LETTERBOX↔TRACK layout switches now ease** over `--zoom-duration` (default: same as `--pan-duration`). Going from the whole stream to a subject is a zoom-in whose source region shrinks from the full frame to the crop while its centre converges on the subject; going back is the reverse. Previously these boundaries swapped layouts in a single frame, which read as a jump cut even when pans were smooth.
*   **One framing model.** Every output frame is now a full-height source region `(x, w)` scaled to the output width and letterboxed if wider than the output aspect. TRACK, LETTERBOX, pans and zooms are all the same `render_region` call, so nothing can drift between them.
*   **Plan metadata:** scenes carry `transition = {kind, from_x, from_w, to_x, to_w, duration_frames}` (replaces `pan`), `boundary_kind` gains `zoom-in` / `zoom-out`, and the plan summary prints `N pan / N zoom / N hold / N layout-switch` with a warning when layout boundaries exist but no zoom was planned. `--plan-json` exports include `transition`.
*   **Pan lab:** `--transition {pan,zoom-in,zoom-out}` for the synthetic fixture; `report.json` records `regionByFrame`.
*   **Render E2E in CI:** `scripts/e2e_render.py` runs the real CLI on a fixture covering zoom-in, pan and zoom-out, asserts gradual motion in the encoded output, and uploads before/after videos plus a contact sheet as a workflow artifact.

#### v1.5.1 — TRACK-to-TRACK pans on production H.264

*   **Fixed pans never running on real MP4s.** v1.5.0 gated pans on an OpenCV random-access seek plus a pixel-difference “hard cut” check. Those seeks are unreliable on production H.264, so almost every scene boundary scored as a hard cut and speaker switches stayed as jump cuts. Pans are now planned from TRACK crop geometry only.
*   **TRACK-to-TRACK crop jumps ease over `--pan-duration`**, including speaker switches and over-segmented TRACK scenes. LETTERBOX layout switches stay instant. Small jitter is still skipped.
*   **Clamped pan interpolation** so the crop window cannot leave the source frame during a transition.
*   **One render path.** Per-frame crop placement now lives in `resolve_frame_crop` / `render_output_frame`, shared by the encode loop, the unit tests, and the pan lab, so the lab can no longer disagree with production.
*   **Transition summary in the plan output** (`N pan / hold / layout-switch over M TRACK->TRACK boundaries`) plus a warning when TRACK boundaries exist but no pan was planned.
*   **`--plan-json`** exports the scene/pan plan; `scripts/pan-lab --plan` replays it against the real clip through the production renderer without re-running YOLO.

#### v1.5.0 — Smooth Subject Pans

*   **Added eased lateral crop transitions.** Visually continuous TRACK-to-TRACK boundaries now pan over 0.4 seconds instead of jumping in one frame.
*   **Preserved editorial cuts.** Hard source cuts and TRACK/LETTERBOX layout changes remain immediate.
*   **Added `--pan-duration`.** Set the transition duration explicitly or use `0` to restore hard crop switches.

#### v1.4.1 (2026-02-15) — Cropping Accuracy Fix

*   **Fixed incorrect crop/letterbox decisions near scene boundaries.** The v1.3 `filter_complex` pipeline used seconds-based `trim` filters, which caused floating-point misalignment with the frame-based scene boundaries from PySceneDetect. This led to frames at scene transitions receiving the wrong strategy (e.g., a properly tracked person switching to letterbox mid-scene, or a group shot being cropped instead of letterboxed). Restored the original frame-by-frame processing pipeline which uses exact frame numbers for scene boundary matching, guaranteeing frame-accurate results.

#### v1.4.0 (2026-02-15) — Hardware Encoding & Scene Detection Tuning

**New Features:**

*   **Hardware encoder support (`--encoder`).** New flag with three modes: `auto` (libx264, default for best quality/compatibility), `hw` (auto-detect VideoToolbox on macOS or NVENC on NVIDIA), or an explicit encoder name. Quality presets (`--quality`) map automatically per encoder type.
*   **Configurable scene detection (`--frame-skip`, `--downscale`).** Power users can tune the speed/accuracy trade-off. Default `--frame-skip 0` processes every frame for maximum accuracy; increase for faster detection on longer videos.

#### v1.3.0 (2026-02-15) — Configurable Output & Quality Presets

**New Features:**

*   **Configurable aspect ratio (`--ratio`).** Output is no longer locked to 9:16. Use `--ratio 4:5` for Instagram feed, `--ratio 1:1` for square, or any custom W:H ratio.
*   **Quality presets (`--quality`).** Choose between `fast` (CRF 28, veryfast), `balanced` (CRF 23, fast — default), or `high` (CRF 18, slow). Power users can override directly with `--crf` and `--preset`.
*   **Dry-run mode (`--plan-only`).** Runs scene detection and analysis only, prints the processing plan, and exits without encoding. Useful for previewing decisions before committing to a long encode.
*   **Fixed output pixel format.** Encoder now outputs `yuv420p` instead of `yuv444p`, which is compatible with all players and platforms and produces smaller files.
*   **Improved logging and progress reporting.** Input file summary upfront (resolution, duration, fps, codec, file size, frame count), progress bars on all slow operations, and a final summary with output size, compression ratio, and processing speed.

#### v1.1.0 (2026-02-14)

**Bug Fixes:**

*   **Fixed audio/video desynchronization.** This was caused by two separate issues:
    *   The frame rate was being read from PySceneDetect while frames were read by OpenCV. A mismatch between the two (e.g. 29.97 vs 30.0) caused the encoded video duration to drift from the audio. FPS is now read from OpenCV (the same backend that reads the frames) with explicit `-vsync cfr` enforcement.
    *   Many source files (especially YouTube downloads) have a non-zero `start_time` on the video stream (e.g. audio at 0.0s, video at 1.8s). The script now detects this offset via `ffprobe` and trims the extracted audio to match, so the two streams stay aligned.
*   **Fixed crash on videos without an audio stream.** The script now detects whether audio exists using `ffprobe` and skips the audio extraction/merge steps gracefully.
*   **Fixed hardcoded `.aac` temp audio file.** The temp audio container is now `.mkv`, which accepts any audio codec. Previously, source files with non-AAC audio (MP3, Opus, AC3, etc.) could fail or produce corrupt output.
*   **Fixed crash when output path has no file extension.** The script now auto-appends `.mp4` if no extension is provided.
*   **Fixed orphaned temp files on failure.** Temporary files are now cleaned up on all exit paths, not just on success.

**Improvements:**

*   **Variable frame rate (VFR) handling.** Phone-recorded videos often use VFR, which caused frame timing drift. The script now detects VFR sources via `ffprobe` and normalizes them to constant frame rate before processing.
*   **Corrupt frame resilience.** If a frame fails to process (bad crop, corrupt data), it is duplicated from the previous good frame instead of being dropped. This preserves the total frame count and prevents audio drift.
*   **Lazy model loading.** YOLO and Haar cascade models are now loaded on first use instead of at import time. Heavy library imports (`torch`, `ultralytics`, `cv2`, etc.) are deferred until after argument parsing, so `--help` is instant.
*   **Pinned dependency versions.** `requirements.txt` now specifies compatible version ranges to prevent breakage from upstream changes.
*   **Replaced `exit()` with `sys.exit(1)`.** Ensures proper exit codes and reliable behavior in all environments.
