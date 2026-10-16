"""No credentials or provider calls: test acceptance of external agent receipts."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "pilot", Path(__file__).parents[1] / "agent-pilot.py"
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class ReceiptTests(unittest.TestCase):
    def test_grok_actual_wire_format(self):
        self.assertEqual(
            pilot.completed_answer(
                "grok", json.dumps({"text": "PILOT_OK", "stopReason": "end_turn"})
            ),
            "PILOT_OK",
        )

    def test_incomplete_and_provider_errors_are_rejected(self):
        for backend, payload in [
            ("grok", {"text": "partial", "stopReason": "max_turns"}),
            ("grok", {"text": "answer", "stopReason": "end_turn", "error": "failed"}),
            ("muse", {"type": "error", "error": "failed"}),
            ("muse", {"type": "text", "part": {"text": "commentary only"}}),
        ]:
            with (
                self.subTest(backend=backend, payload=payload),
                self.assertRaises(ValueError),
            ):
                pilot.completed_answer(backend, json.dumps(payload))

    def test_muse_uses_final_text(self):
        events = [
            {"type": "text", "part": {"text": "starting"}},
            {"type": "text", "part": {"text": "verified"}},
            {"type": "step_finish", "part": {"reason": "stop"}},
        ]
        self.assertEqual(
            pilot.completed_answer("muse", "\n".join(map(json.dumps, events))),
            "verified",
        )

    def test_fingerprint_detects_edits_when_status_would_still_be_modified(self):
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[2] / ".runtime"
        ) as directory:
            root = Path(directory)
            target = root / "already-modified.txt"
            target.write_bytes(b"before")
            with patch.object(
                pilot.subprocess, "check_output", return_value=b"already-modified.txt\0"
            ):
                before = pilot.fingerprint(root)
                target.write_bytes(b"after")
                self.assertNotEqual(before, pilot.fingerprint(root))


if __name__ == "__main__":
    unittest.main()
