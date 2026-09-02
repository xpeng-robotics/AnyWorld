from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "anyworld_world_infer", REPO / "world_model/scripts/infer.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ManifestValidationTest(unittest.TestCase):
    def test_valid_record_and_safe_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "frame.png"
            control = root / "control.mp4"
            extrinsics = root / "camera.npy"
            for path in (image, control, extrinsics):
                path.touch()
            record = {
                "episode": "demo/one",
                "caption": "An IRON robot lifts an object.",
                "image_path": str(image),
                "control_video_path": str(control),
                "state_camera_extrinsic_path": str(extrinsics),
                "camera_fx": 100.0,
                "camera_fy": 100.0,
                "camera_cx": 50.0,
                "camera_cy": 40.0,
                "camera_orig_width": 100,
                "camera_orig_height": 80,
                "spatial_preprocess": "center_crop_resize",
            }
            MODULE.validate_records([record])
            self.assertEqual(MODULE.safe_name(record["episode"]), "demo_one")

    def test_rejects_geometry_mismatch(self) -> None:
        record = {field: 1 for field in MODULE.REQUIRED_FIELDS}
        record.update({"episode": "bad", "spatial_preprocess": "direct_resize"})
        with self.assertRaisesRegex(ValueError, "center_crop_resize"):
            MODULE.validate_records([record])


    def test_check_only_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "base_model"
            weights = root / "world_model_weights"
            (weights / "transformer").mkdir(parents=True)
            model.mkdir()
            paths = [root / "frame.png", root / "control.mp4", root / "camera.npy"]
            for path in paths:
                path.touch()
            record = {
                "episode": "check-only",
                "caption": "An IRON robot lifts an object.",
                "image_path": str(paths[0]),
                "control_video_path": str(paths[1]),
                "state_camera_extrinsic_path": str(paths[2]),
                "camera_fx": 100.0,
                "camera_fy": 100.0,
                "camera_cx": 50.0,
                "camera_cy": 40.0,
                "camera_orig_width": 100,
                "camera_orig_height": 80,
                "spatial_preprocess": "center_crop_resize",
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"data": [record]}))
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "world_model/scripts/infer.py"),
                    "--base-model-path", str(model),
                    "--anyworld-model-path", str(weights),
                    "--validation-file", str(manifest),
                    "--output-dir", str(root / "output"),
                    "--check-only",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Validated 1 samples", result.stdout)

if __name__ == "__main__":
    unittest.main()
