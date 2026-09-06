"""Light-ASD active speaker detection for speaker focus (phase 2).

Model: "A Light Weight Model for Active Speaker Detection" (Liao et al.,
CVPR 2023), https://github.com/Junhua-Liao/Light-ASD, MIT license. The
network definition below is a trimmed copy of that repository's
model/Encoder.py, model/Classifier.py, model/Model.py and loss.py so the
pretrained weights load unchanged. ~1.0M parameters, CPU friendly.

Contract (from the reference Columbia_test.py):
  audio   13-dim MFCC at 100 Hz from 16 kHz mono PCM
          (python_speech_features.mfcc, winlen 0.025, winstep 0.010)
  visual  grayscale face crops at 25 fps: the face box padded by crop_scale,
          resized to 224x224, centre-cropped to 112x112
  output  one logit pair per visual frame; softmax[..., 1] is P(speaking)

`score_track_crops()` handles resampling from the source frame rate to 25 fps
and back, so callers work in source frame indices throughout.
"""
import hashlib
import os
import urllib.request

import numpy as np

MODEL_URL = ("https://github.com/Junhua-Liao/Light-ASD/raw/"
             "ed38c232de5efe0261dbd68627c0ade7cdfe14eb/weight/pretrain_AVA_CVPR.model")
MODEL_SHA256 = "d44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317"
MODEL_ENV = "AUTOCROP_ASD_MODEL"

AUDIO_RATE = 16000
MFCC_HZ = 100
VISUAL_FPS = 25
CROP_SCALE = 0.40
CHUNK_SEC = 6          # score in chunks so memory stays flat on long scenes

_model = None


def model_path():
    """Locate the weights: $AUTOCROP_ASD_MODEL, else a cached download."""
    explicit = os.environ.get(MODEL_ENV)
    if explicit:
        return explicit
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "autocrop")
    path = os.path.join(cache_dir, "light_asd_pretrain_AVA_CVPR.model")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(MODEL_URL, tmp)
        digest = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
        if digest != MODEL_SHA256:
            os.remove(tmp)
            raise RuntimeError(f"Light-ASD weights checksum mismatch: {digest}")
        os.replace(tmp, path)
    return path


def available():
    """True when torch and python_speech_features can be imported."""
    try:
        import torch  # noqa: F401
        import python_speech_features  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Network (verbatim structure from Light-ASD so state_dict keys match)
# ---------------------------------------------------------------------------
def _build_modules():
    import torch
    import torch.nn as nn

    class Audio_Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.relu = nn.ReLU()
            self.m_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
            self.bn_m_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
            self.t_3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
            self.bn_t_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
            self.m_5 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), bias=False)
            self.bn_m_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
            self.t_5 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2), bias=False)
            self.bn_t_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
            self.last = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), bias=False)
            self.bn_last = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        def forward(self, x):
            x_3 = self.relu(self.bn_m_3(self.m_3(x)))
            x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))
            x_5 = self.relu(self.bn_m_5(self.m_5(x)))
            x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))
            return self.relu(self.bn_last(self.last(x_3 + x_5)))

    class Visual_Block(nn.Module):
        def __init__(self, in_channels, out_channels, is_down=False):
            super().__init__()
            self.relu = nn.ReLU()
            stride = (1, 2, 2) if is_down else (1, 1, 1)
            self.s_3 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.t_3 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
            self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.s_5 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5), stride=stride, padding=(0, 2, 2), bias=False)
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.t_5 = nn.Conv3d(out_channels, out_channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False)
            self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.last = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=False)
            self.bn_last = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        def forward(self, x):
            x_3 = self.relu(self.bn_s_3(self.s_3(x)))
            x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))
            x_5 = self.relu(self.bn_s_5(self.s_5(x)))
            x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))
            return self.relu(self.bn_last(self.last(x_3 + x_5)))

    class visual_encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = Visual_Block(1, 32, is_down=True)
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.block2 = Visual_Block(32, 64)
            self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.block3 = Visual_Block(64, 128)
            self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        def forward(self, x):
            x = self.pool1(self.block1(x))
            x = self.pool2(self.block2(x))
            x = self.block3(x)
            x = x.transpose(1, 2)
            B, T, C, W, H = x.shape
            x = x.reshape(B * T, C, W, H)
            x = self.maxpool(x)
            return x.view(B, T, C)

    class audio_encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = Audio_Block(1, 32)
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
            self.block2 = Audio_Block(32, 64)
            self.pool2 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
            self.block3 = Audio_Block(64, 128)

        def forward(self, x):
            x = self.pool1(self.block1(x))
            x = self.pool2(self.block2(x))
            x = self.block3(x)
            x = torch.mean(x, dim=2, keepdim=True)
            return x.squeeze(2).transpose(1, 2)

    class BGRU(nn.Module):
        def __init__(self, channel):
            super().__init__()
            self.gru_forward = nn.GRU(input_size=channel, hidden_size=channel, num_layers=1, bidirectional=False, bias=True, batch_first=True)
            self.gru_backward = nn.GRU(input_size=channel, hidden_size=channel, num_layers=1, bidirectional=False, bias=True, batch_first=True)
            self.gelu = nn.GELU()

        def forward(self, x):
            x, _ = self.gru_forward(x)
            x = self.gelu(x)
            x = torch.flip(x, dims=[1])
            x, _ = self.gru_backward(x)
            x = torch.flip(x, dims=[1])
            return self.gelu(x)

    class ASD_Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.visualEncoder = visual_encoder()
            self.audioEncoder = audio_encoder()
            self.GRU = BGRU(128)

        def forward_visual_frontend(self, x):
            B, T, W, H = x.shape
            x = x.view(B, 1, T, W, H)
            x = (x / 255 - 0.4161) / 0.1688
            return self.visualEncoder(x)

        def forward_audio_frontend(self, x):
            x = x.unsqueeze(1).transpose(2, 3)
            return self.audioEncoder(x)

        def forward_audio_visual_backend(self, x1, x2):
            x = self.GRU(x1 + x2)
            return torch.reshape(x, (-1, 128))

    class lossAV(nn.Module):
        def __init__(self):
            super().__init__()
            self.FC = nn.Linear(128, 2)

    class LightASD(nn.Module):
        """Inference wrapper whose state_dict keys match the released weights."""
        def __init__(self):
            super().__init__()
            self.model = ASD_Model()
            self.lossAV = lossAV()

        def speaking_probability(self, mfcc, faces):
            """mfcc: (Ta, 13) float; faces: (Tv, 112, 112) uint8 -> (Tv,) P(speaking)."""
            a = torch.as_tensor(np.asarray(mfcc, dtype=np.float32)).unsqueeze(0)
            v = torch.as_tensor(np.asarray(faces, dtype=np.float32)).unsqueeze(0)
            embed_a = self.model.forward_audio_frontend(a)
            embed_v = self.model.forward_visual_frontend(v)
            out = self.model.forward_audio_visual_backend(embed_a, embed_v)
            logits = self.lossAV.FC(out)
            return torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

    return LightASD


def get_model():
    global _model
    if _model is None:
        import torch
        LightASD = _build_modules()
        model = LightASD()
        state = torch.load(model_path(), map_location="cpu")
        own = model.state_dict()
        missing = []
        for name, param in state.items():
            key = name if name in own else name.replace("module.", "")
            if key in own and own[key].shape == param.shape:
                own[key].copy_(param)
            elif not key.startswith("lossV."):
                missing.append(name)
        if missing:
            raise RuntimeError(f"Light-ASD weights did not match model: {missing[:5]}")
        model.eval()
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        _model = model
    return _model


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def extract_audio(video_path, start_sec, end_sec):
    """16 kHz mono int16 PCM for [start_sec, end_sec) via ffmpeg (None if no audio)."""
    import subprocess
    duration = max(0.0, end_sec - start_sec)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start_sec:.3f}",
           "-t", f"{duration:.3f}", "-i", video_path, "-vn", "-ac", "1",
           "-ar", str(AUDIO_RATE), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return np.frombuffer(proc.stdout, dtype=np.int16)


def mfcc_features(audio):
    import python_speech_features
    return python_speech_features.mfcc(audio, AUDIO_RATE, numcep=13,
                                       winlen=0.025, winstep=0.010)


def crop_face(frame_gray, box, crop_scale=CROP_SCALE):
    """Light-ASD style crop: box padded by crop_scale, 224x224 -> centre 112x112."""
    import cv2
    x1, y1, x2, y2 = box
    bs = max(y2 - y1, x2 - x1) / 2.0
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bsi = int(bs * (1 + 2 * crop_scale))
    padded = np.pad(frame_gray, ((bsi, bsi), (bsi, bsi)), "constant", constant_values=110)
    my, mx = cy + bsi, cx + bsi
    face = padded[int(my - bs):int(my + bs * (1 + 2 * crop_scale)),
                  int(mx - bs * (1 + crop_scale)):int(mx + bs * (1 + crop_scale))]
    if face.size == 0:
        return np.full((112, 112), 110, dtype=np.uint8)
    face = cv2.resize(face, (224, 224))
    return face[56:168, 56:168]


def score_track_crops(model, mfcc, crops_by_frame, start_frame, end_frame, fps):
    """Score one face track.

    crops_by_frame: {source frame -> 112x112 uint8}. Frames of the scene the
    track does not cover get a neutral grey crop (the model then leans on
    audio alone, which reads as "not this face"). Returns per-source-frame
    P(speaking) over [start_frame, end_frame).
    """
    length = end_frame - start_frame
    duration = length / fps
    n_visual = int(round(duration * VISUAL_FPS))
    n_audio = int(round(duration * MFCC_HZ))
    if n_visual < 1:
        return np.zeros(length)
    neutral = np.full((112, 112), 110, dtype=np.uint8)
    faces = np.empty((n_visual, 112, 112), dtype=np.uint8)
    for k in range(n_visual):
        src = start_frame + min(length - 1, int(round(k * fps / VISUAL_FPS)))
        faces[k] = crops_by_frame.get(src, neutral)
    if mfcc.shape[0] < n_audio:
        mfcc = np.pad(mfcc, ((0, n_audio - mfcc.shape[0]), (0, 0)), "edge")
    mfcc = mfcc[:n_audio]

    import torch
    probs = np.empty(n_visual)
    chunk_v = CHUNK_SEC * VISUAL_FPS
    with torch.no_grad():
        for v0 in range(0, n_visual, chunk_v):
            v1 = min(n_visual, v0 + chunk_v)
            a0, a1 = v0 * 4, v1 * 4
            probs[v0:v1] = model.speaking_probability(mfcc[a0:a1], faces[v0:v1])[:v1 - v0]
    # Back to source frames: nearest 25fps sample.
    idx = np.minimum(n_visual - 1, np.round(np.arange(length) / fps * VISUAL_FPS).astype(int))
    return probs[idx]


def audio_energy_by_frame(audio, start_frame, end_frame, fps):
    """RMS (0..1 of full scale) per source frame; used to zero out digital silence."""
    length = end_frame - start_frame
    if audio is None or len(audio) == 0:
        return np.zeros(length)
    samples = audio.astype(np.float32) / 32768.0
    per_frame = AUDIO_RATE / fps
    out = np.zeros(length)
    for n in range(length):
        a = int(n * per_frame)
        b = int((n + 1) * per_frame)
        seg = samples[a:b]
        out[n] = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
    return out
