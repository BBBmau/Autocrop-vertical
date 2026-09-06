import unittest

import main


def track(start, end, x0, x1):
    return {"start_frame": start, "end_frame": end, "strategy": "TRACK",
            "target_box": [x0, 0, x1, 90]}


def letterbox(start, end):
    return {"start_frame": start, "end_frame": end, "strategy": "LETTERBOX",
            "target_box": None}


class PanMathTests(unittest.TestCase):
    def test_smoothstep_is_monotonic_and_clamped(self):
        values = [main.smoothstep(i / 20) for i in range(21)]
        self.assertEqual(main.smoothstep(-1), 0)
        self.assertEqual(main.smoothstep(2), 1)
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 1)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))

    def test_pan_starts_and_ends_at_requested_positions(self):
        positions = [
            main.interpolate_pan_x(10, 110, frame, 5)
            for frame in range(5)
        ]
        self.assertEqual(positions[0], 10)
        self.assertEqual(positions[-1], 110)
        self.assertTrue(all(a < b for a, b in zip(positions, positions[1:])))
        self.assertLess(positions[1] - positions[0], positions[2] - positions[1])
        self.assertGreater(positions[3] - positions[2], positions[4] - positions[3])

    def test_interpolate_pan_x_clamps_to_source_bounds(self):
        self.assertEqual(
            main.interpolate_pan_x(-20, 200, 0, 5, min_x=0, max_x=110),
            0,
        )
        self.assertEqual(
            main.interpolate_pan_x(-20, 200, 4, 5, min_x=0, max_x=110),
            110,
        )

    def test_crop_center_is_clamped_to_source_bounds(self):
        self.assertEqual(
            main.calculate_crop_box_for_center(0, 160, 90, 50),
            (0, 0, 50, 90),
        )
        self.assertEqual(
            main.calculate_crop_box_for_center(160, 160, 90, 50),
            (110, 0, 160, 90),
        )

    def test_region_interpolation_matches_pan_for_equal_widths(self):
        for frame in range(5):
            x, w = main.interpolate_region((10, 50), (110, 50), frame, 5, 160)
            self.assertEqual(w, 50)
            self.assertEqual(x, main.interpolate_pan_x(10, 110, frame, 5))

    def test_region_interpolation_zooms_between_full_frame_and_crop(self):
        regions = [
            main.interpolate_region((0, 160), (110, 50), frame, 6, 160)
            for frame in range(6)
        ]
        self.assertEqual(regions[0], (0, 160))
        self.assertEqual(regions[-1], (110, 50))
        widths = [w for _, w in regions]
        self.assertTrue(all(a > b for a, b in zip(widths, widths[1:])))
        for x, w in regions:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + w, 160)

    def test_track_to_track_crop_jump_pans_without_reading_pixels(self):
        scenes = [track(0, 10, 10, 40), track(10, 20, 120, 150)]
        # video_path is unused for pan eligibility; production H.264 must
        # still pan when adjacent frames would fail a pixel hard-cut check.
        main.plan_pan_transitions(
            "/nonexistent.mp4", scenes, 160, 90, 10, pan_duration=0.4)

        self.assertEqual(scenes[1]["boundary_kind"], "pan")
        transition = scenes[1]["transition"]
        self.assertIsNotNone(transition)
        self.assertEqual(transition["kind"], "pan")
        self.assertEqual(transition["from_w"], transition["to_w"])
        positions = [
            main.interpolate_pan_x(
                transition["from_x"], transition["to_x"], frame,
                transition["duration_frames"])
            for frame in range(transition["duration_frames"])
        ]
        self.assertEqual(len(positions), 4)
        self.assertEqual(positions[0], transition["from_x"])
        self.assertEqual(positions[-1], transition["to_x"])
        self.assertTrue(all(a < b for a, b in zip(positions, positions[1:])))

    def test_speaker_switch_zoom_and_jitter(self):
        scenes = [
            track(0, 10, 10, 40),
            track(10, 20, 120, 150),
            letterbox(20, 30),
        ]
        main.plan_pan_transitions(
            None, scenes, 160, 90, 10, pan_duration=0.4)

        self.assertEqual(scenes[1]["boundary_kind"], "pan")
        self.assertIsNotNone(scenes[1]["transition"])
        self.assertEqual(scenes[2]["boundary_kind"], "zoom-out")
        zoom = scenes[2]["transition"]
        self.assertEqual((zoom["from_x"], zoom["from_w"]), (110, 50))
        self.assertEqual((zoom["to_x"], zoom["to_w"]), (0, 160))
        self.assertEqual(zoom["duration_frames"], 4)

        jitter_scenes = [track(0, 10, 70, 90), track(10, 20, 72, 92)]
        main.plan_pan_transitions(
            None, jitter_scenes, 160, 90, 10, pan_duration=0.4)
        self.assertEqual(jitter_scenes[1]["boundary_kind"], "hold")
        self.assertIsNone(jitter_scenes[1]["transition"])

    def test_letterbox_to_track_zooms_in(self):
        scenes = [letterbox(0, 10), track(10, 20, 120, 150)]
        main.plan_pan_transitions(None, scenes, 160, 90, 10, pan_duration=0.4)
        self.assertEqual(scenes[1]["boundary_kind"], "zoom-in")
        zoom = scenes[1]["transition"]
        self.assertEqual((zoom["from_x"], zoom["from_w"]), (0, 160))
        self.assertEqual((zoom["to_x"], zoom["to_w"]), (110, 50))

    def test_zoom_duration_is_independent_of_pan_duration(self):
        scenes = [letterbox(0, 10), track(10, 30, 120, 150)]
        main.plan_pan_transitions(
            None, scenes, 160, 90, 10, pan_duration=0.4, zoom_duration=1.0)
        self.assertEqual(scenes[1]["transition"]["duration_frames"], 10)

        scenes = [letterbox(0, 10), track(10, 30, 120, 150)]
        main.plan_pan_transitions(
            None, scenes, 160, 90, 10, pan_duration=0.4, zoom_duration=0)
        self.assertEqual(scenes[1]["boundary_kind"], "layout-switch")
        self.assertIsNone(scenes[1]["transition"])

    def test_zoom_is_capped_to_scene_length(self):
        scenes = [letterbox(0, 10), track(10, 12, 120, 150)]
        main.plan_pan_transitions(None, scenes, 160, 90, 10, pan_duration=0.4)
        self.assertEqual(scenes[1]["transition"]["duration_frames"], 2)

    def test_zero_pan_duration_disables_transitions(self):
        scenes = [track(0, 10, 10, 40), track(10, 20, 120, 150)]
        main.plan_pan_transitions(
            None, scenes, 160, 90, 10, pan_duration=0)
        self.assertIsNone(scenes[1]["transition"])

    def test_summary_counts_transitions_against_boundaries(self):
        scenes = [
            track(0, 10, 10, 40),
            track(10, 20, 120, 150),
            track(20, 30, 121, 151),
            letterbox(30, 40),
            letterbox(40, 50),
            track(50, 60, 10, 40),
        ]
        main.plan_pan_transitions(None, scenes, 160, 90, 10, pan_duration=0.4)
        summary = main.summarize_pan_plan(scenes)
        self.assertEqual(summary["track_to_track"], 2)
        self.assertEqual(summary["layout_boundaries"], 2)
        self.assertEqual(summary["pan"], 1)
        self.assertEqual(summary["hold"], 1)
        self.assertEqual(summary["zoom"], 2)
        self.assertEqual(summary["layout_switch"], 0)

        main.plan_pan_transitions(
            None, scenes, 160, 90, 10, pan_duration=0.4, zoom_duration=0)
        summary = main.summarize_pan_plan(scenes)
        self.assertEqual(summary["zoom"], 0)
        self.assertEqual(summary["layout_switch"], 2)


class ProductionRenderPathTests(unittest.TestCase):
    """Drive the exact per-frame functions the encode loop uses."""

    WIDTH, HEIGHT, FPS = 160, 90, 10

    def two_scene_plan(self, pan_duration=0.4):
        scenes = [track(0, 10, 10, 40), track(10, 30, 120, 150)]
        main.plan_pan_transitions(
            None, scenes, self.WIDTH, self.HEIGHT, self.FPS,
            pan_duration=pan_duration)
        return scenes

    def zoom_plan(self, zoom_duration=0.6):
        scenes = [letterbox(0, 10), track(10, 30, 120, 150), letterbox(30, 50)]
        main.plan_pan_transitions(
            None, scenes, self.WIDTH, self.HEIGHT, self.FPS,
            pan_duration=0.4, zoom_duration=zoom_duration)
        return scenes

    def test_frame_crops_move_gradually_through_a_pan(self):
        scenes = self.two_scene_plan()
        positions = main.plan_frame_crops(scenes, 30, self.WIDTH, self.HEIGHT)

        before = positions[:10]
        pan_frames = scenes[1]["transition"]["duration_frames"]
        during = positions[10:10 + pan_frames]
        after = positions[10 + pan_frames:]

        self.assertEqual(set(before), {0})
        self.assertEqual(set(after), {110})
        self.assertEqual(during[0], 0)
        self.assertEqual(during[-1], 110)
        # A real pan has several distinct intermediate positions, not a snap.
        self.assertGreaterEqual(len(set(during)), 4)
        self.assertTrue(all(a < b for a, b in zip(during, during[1:])))

    def test_frame_regions_zoom_in_and_out_gradually(self):
        scenes = self.zoom_plan()
        regions = main.plan_frame_regions(scenes, 50, self.WIDTH, self.HEIGHT)

        self.assertEqual(set(regions[:10]), {(0, 160)})
        zoom_in = regions[10:16]
        self.assertEqual(zoom_in[0], (0, 160))
        self.assertEqual(zoom_in[-1], (110, 50))
        widths = [w for _, w in zoom_in]
        self.assertTrue(all(a > b for a, b in zip(widths, widths[1:])))
        self.assertEqual(set(regions[16:30]), {(110, 50)})

        zoom_out = regions[30:36]
        self.assertEqual(zoom_out[0], (110, 50))
        self.assertEqual(zoom_out[-1], (0, 160))
        widths = [w for _, w in zoom_out]
        self.assertTrue(all(a < b for a, b in zip(widths, widths[1:])))
        self.assertEqual(set(regions[36:]), {(0, 160)})

    def test_zero_pan_duration_snaps_in_one_frame(self):
        scenes = self.two_scene_plan(pan_duration=0)
        positions = main.plan_frame_crops(scenes, 30, self.WIDTH, self.HEIGHT)
        self.assertEqual(positions[9], 0)
        self.assertEqual(positions[10], 110)

    def test_scene_cursor_advances_and_never_rewinds(self):
        scenes = self.two_scene_plan()
        self.assertEqual(main.scene_index_for_frame(scenes, 0, 0), 0)
        self.assertEqual(main.scene_index_for_frame(scenes, 9, 0), 0)
        self.assertEqual(main.scene_index_for_frame(scenes, 10, 0), 1)
        self.assertEqual(main.scene_index_for_frame(scenes, 5, 1), 1)

    def _column_encoded_frame(self):
        import numpy as np
        # Encode each source column's x coordinate as its pixel intensity so
        # output pixels report exactly which source columns were shown.
        frame = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(self.WIDTH, dtype=np.uint8)[None, :]
        frame[:, :, 1] = 200  # non-black so letterbox bars are detectable
        return frame

    def test_rendered_pixels_shift_gradually_across_boundary(self):
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"OpenCV unavailable: {exc}")

        scenes = self.two_scene_plan()
        out_w, out_h = main.compute_output_size(self.HEIGHT)
        frame = self._column_encoded_frame()

        rendered_crop_x = []
        index = 0
        for frame_number in range(30):
            index = main.scene_index_for_frame(scenes, frame_number, index)
            out = main.render_output_frame(
                frame, scenes[index], frame_number,
                self.WIDTH, self.HEIGHT, out_w, out_h)
            self.assertEqual(out.shape, (out_h, out_w, 3))
            rendered_crop_x.append(int(out[0, 0, 0]))

        planned = main.plan_frame_crops(scenes, 30, self.WIDTH, self.HEIGHT)
        self.assertEqual(rendered_crop_x, planned)
        pan_frames = scenes[1]["transition"]["duration_frames"]
        during = rendered_crop_x[10:10 + pan_frames]
        self.assertGreaterEqual(len(set(during)), 4)
        self.assertTrue(all(a < b for a, b in zip(during, during[1:])))

    def test_rendered_zoom_shrinks_letterbox_bars_gradually(self):
        try:
            import cv2  # noqa: F401
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest(f"OpenCV unavailable: {exc}")

        scenes = self.zoom_plan()
        out_w, out_h = main.compute_output_size(self.HEIGHT)
        frame = self._column_encoded_frame()

        bar_heights = []
        left_columns = []
        index = 0
        for frame_number in range(50):
            index = main.scene_index_for_frame(scenes, frame_number, index)
            out = main.render_output_frame(
                frame, scenes[index], frame_number,
                self.WIDTH, self.HEIGHT, out_w, out_h)
            self.assertEqual(out.shape, (out_h, out_w, 3))
            content_rows = np.where(out[:, out_w // 2, 1] > 0)[0]
            bar_heights.append(int(content_rows[0]))
            left_columns.append(int(out[out_h // 2, 0, 0]))

        # LETTERBOX steady state: bars present, whole frame visible (the
        # downscale averages the first few source columns into pixel 0).
        self.assertGreater(bar_heights[0], 0)
        self.assertLessEqual(left_columns[0], 3)
        # TRACK steady state: no bars, crop starts at x=110.
        self.assertEqual(bar_heights[20], 0)
        self.assertEqual(left_columns[20], 110)

        zoom_in = bar_heights[10:16]
        self.assertTrue(all(a >= b for a, b in zip(zoom_in, zoom_in[1:])))
        self.assertGreaterEqual(len(set(zoom_in)), 4)
        zoom_out = bar_heights[30:36]
        self.assertTrue(all(a <= b for a, b in zip(zoom_out, zoom_out[1:])))
        self.assertGreaterEqual(len(set(zoom_out)), 4)
        # The visible left edge converges on the subject instead of jumping.
        lefts = left_columns[10:16]
        self.assertTrue(all(a <= b + 2 for a, b in zip(lefts, lefts[1:])))
        self.assertGreater(lefts[-1] - lefts[0], 90)

    def test_serialized_plan_round_trips_through_json(self):
        import json
        scenes = self.zoom_plan()
        for scene in scenes:
            scene["analysis"] = [{"person_box": [1, 2, 3, 4]}]
        payload = json.loads(json.dumps(main.serialize_plan(
            scenes, self.WIDTH, self.HEIGHT, self.FPS, "9:16")))
        self.assertEqual(payload["summary"]["zoom"], 2)
        self.assertEqual(payload["scenes"][1]["boundary_kind"], "zoom-in")
        self.assertEqual(payload["scenes"][1]["transition"]["kind"], "zoom-in")
        self.assertEqual(payload["scenes"][1]["people"], 1)
        replayed = main.plan_frame_regions(
            payload["scenes"], 50, payload["width"], payload["height"])
        self.assertEqual(
            [tuple(r) for r in replayed],
            main.plan_frame_regions(scenes, 50, self.WIDTH, self.HEIGHT))


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"OpenCV test dependencies unavailable: {exc}")

    def test_frame_difference_separates_motion_from_hard_cut(self):
        import numpy as np

        before = np.full((90, 160, 3), 80, dtype=np.uint8)
        soft = before.copy()
        soft[30:50, 40:60] = 100
        hard = np.full((90, 160, 3), 240, dtype=np.uint8)

        self.assertLess(main.frame_difference_score(before, soft), 0.18)
        self.assertGreater(main.frame_difference_score(before, hard), 0.18)


if __name__ == "__main__":
    unittest.main()
