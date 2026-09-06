import unittest

import numpy as np

import asd
import speaker


class FakeModel:
    """Returns P(speaking)=1 whenever the crop is bright, else 0."""
    def speaking_probability(self, mfcc, faces):
        assert mfcc.shape[0] == faces.shape[0] * 4, (mfcc.shape, faces.shape)
        return (faces.reshape(faces.shape[0], -1).mean(axis=1) > 128).astype(float)


class CropFace(unittest.TestCase):
    def test_crop_is_112_square_centered_on_box(self):
        frame = np.zeros((720, 1280), np.uint8)
        frame[150:250, 300:400] = 255
        crop = asd.crop_face(frame, [300, 150, 400, 250])
        self.assertEqual(crop.shape, (112, 112))
        self.assertGreater(crop[56, 56], 200)       # face centre is bright
        # The reference crop keeps the face itself (224 -> centre 112) and is
        # biased downward to include the mouth: the top rows are still face,
        # a box far below the face is not.
        self.assertGreater(crop[5, 56], 200)
        far = asd.crop_face(frame, [300, 600, 400, 700])
        self.assertLess(far.mean(), 130)

    def test_crop_near_frame_edge_does_not_fail(self):
        frame = np.zeros((720, 1280), np.uint8)
        crop = asd.crop_face(frame, [0, 0, 60, 60])
        self.assertEqual(crop.shape, (112, 112))


class ScoreTrackCrops(unittest.TestCase):
    def test_resamples_30fps_to_25fps_and_back(self):
        fps, start, end = 30, 90, 240   # 5 s
        bright = np.full((112, 112), 255, np.uint8)
        dark = np.zeros((112, 112), np.uint8)
        crops = {n: (bright if n < 165 else dark) for n in range(start, end)}
        mfcc = np.zeros((int(5 * 100) + 7, 13))
        probs = asd.score_track_crops(FakeModel(), mfcc, crops, start, end, fps)
        self.assertEqual(len(probs), end - start)
        self.assertTrue(np.all(probs[:70] == 1.0))
        self.assertTrue(np.all(probs[80:] == 0.0))

    def test_frames_without_a_crop_score_zero_for_bright_model(self):
        probs = asd.score_track_crops(FakeModel(), np.zeros((300, 13)), {}, 0, 90, 30)
        self.assertEqual(len(probs), 90)
        self.assertTrue(np.all(probs == 0.0))

    def test_short_mfcc_is_padded(self):
        crops = {n: np.full((112, 112), 255, np.uint8) for n in range(60)}
        probs = asd.score_track_crops(FakeModel(), np.zeros((50, 13)), crops, 0, 60, 30)
        self.assertEqual(len(probs), 60)


class AudioEnergy(unittest.TestCase):
    def test_silence_then_tone(self):
        rate = asd.AUDIO_RATE
        audio = np.zeros(rate * 2, np.int16)
        t = np.arange(rate) / rate
        audio[rate:] = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
        energy = asd.audio_energy_by_frame(audio, 0, 60, 30)
        self.assertEqual(len(energy), 60)
        self.assertTrue(np.all(energy[:29] < speaker.SILENCE_RMS))
        self.assertTrue(np.all(energy[31:] > 0.1))

    def test_no_audio_is_all_zero(self):
        self.assertTrue(np.all(asd.audio_energy_by_frame(None, 0, 10, 30) == 0))


class Availability(unittest.TestCase):
    def test_available_is_bool(self):
        self.assertIn(asd.available(), (True, False))


if __name__ == "__main__":
    unittest.main()
