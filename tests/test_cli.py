import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CliTests(unittest.TestCase):
    def test_toy_input_and_existing_output_protection(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "toy_input.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selection.json"
            command = [sys.executable, "-m", "mixture_then_select.cli", "--input", str(example), "--output", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            original = output.read_bytes()
            self.assertEqual(len(json.loads(original)["selected_indices"]), 6)
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(output.read_bytes(), original)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            source.write_text('{"budget": 1, "budget": 2}', encoding="utf-8")
            output = Path(directory) / "output.json"
            command = [sys.executable, "-m", "mixture_then_select.cli", "--input", str(source), "--output", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicate JSON key", completed.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
