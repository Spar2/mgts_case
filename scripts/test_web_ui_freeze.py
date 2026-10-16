"""Exercise the freeze check in disposable fixtures, without touching web UI."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("freeze_check", REPO / "scripts/check-web-ui-freeze.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class FreezeCheckTests(unittest.TestCase):
    def setUp(self):
        runtime = REPO / ".runtime/ui-freeze-tests"
        runtime.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=runtime)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "apps/web/src/main.tsx"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"approved\ninterface\n")
        self.manifest = self.root / "baseline.json"
        self.manifest.write_text(json.dumps({
            "schema_version": "dts-frozen-ui-v1", "baseline_id": "fixture",
            "protected_directories": ["apps/web/src"], "protected_files": [],
            "sha256_lf": {"apps/web/src/main.tsx": hashlib.sha256(self.source.read_bytes()).hexdigest()},
        }), encoding="utf-8")
        self.original_root, self.original_manifest = CHECK.ROOT, CHECK.MANIFEST
        CHECK.ROOT, CHECK.MANIFEST = self.root, self.manifest

    def tearDown(self):
        CHECK.ROOT, CHECK.MANIFEST = self.original_root, self.original_manifest
        self.temp.cleanup()

    def result(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = CHECK.check()
        return code, output.getvalue()

    def test_approved_files_pass(self):
        self.assertEqual(self.result()[0], 0)

    def test_windows_line_endings_are_not_a_redesign(self):
        self.source.write_bytes(b"approved\r\ninterface\r\n")
        self.assertEqual(self.result()[0], 0)

    def test_modified_interface_fails(self):
        self.source.write_bytes(b"unapproved redesign\n")
        code, output = self.result()
        self.assertEqual(code, 1)
        self.assertIn("CHANGED apps/web/src/main.tsx", output)

    def test_added_page_fails(self):
        self.source.with_name("new-page.tsx").write_bytes(b"new page\n")
        code, output = self.result()
        self.assertEqual(code, 1)
        self.assertIn("ADDED apps/web/src/new-page.tsx", output)

    def test_removed_page_fails(self):
        self.source.unlink()
        code, output = self.result()
        self.assertEqual(code, 1)
        self.assertIn("MISSING apps/web/src/main.tsx", output)


if __name__ == "__main__":
    unittest.main()
