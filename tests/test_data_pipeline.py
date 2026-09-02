from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO = Path(__file__).resolve().parents[1]


class DataPipelineTest(unittest.TestCase):
    def test_metadata_and_default_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            human_dir = dataset / "human_hand_output"
            robot_dir = dataset / "robot_hand_frames"
            human_dir.mkdir()
            robot_dir.mkdir()
            Image.new("RGB", (32, 24), "red").save(
                human_dir / "sample_human_hands_v1.png"
            )
            Image.new("RGB", (32, 24), "blue").save(robot_dir / "sample.png")

            metadata = dataset / "metadata.json"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "image_editing/data/create_metadata.py"),
                    "--dataset-root", str(dataset),
                    "--output", str(metadata),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            records = json.loads(metadata.read_text())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["image"], "robot_hand_frames/sample.png")
            self.assertEqual(
                records[0]["edit_image"],
                ["human_hand_output/sample_human_hands_v1.png"],
            )

            augmented = dataset / "augmented.json"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "image_editing/data/augment_pairs.py"),
                    "--dataset-root", str(dataset),
                    "--input-json", str(metadata),
                    "--output-json", str(augmented),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            outputs = json.loads(augmented.read_text())
            self.assertEqual(len(outputs), 18)
            for record in outputs:
                self.assertTrue((dataset / record["image"]).is_file())
                self.assertTrue((dataset / record["edit_image"][0]).is_file())


    def test_public_image_entrypoints_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "robot.png"
            checkpoint = root / "editor.safetensors"
            Image.new("RGB", (32, 24), "green").save(image)
            checkpoint.touch()

            generated = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "image_editing/data/generate_reverse_pairs.py"),
                    "--input-images", str(image),
                    "--output-dir", str(root / "reverse"),
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("planned outputs: 3", generated.stdout)

            inferred = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "image_editing/infer.py"),
                    "--input", str(image),
                    "--output-dir", str(root / "edited"),
                    "--checkpoint", str(checkpoint),
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Validated 1 inputs", inferred.stdout)

if __name__ == "__main__":
    unittest.main()
