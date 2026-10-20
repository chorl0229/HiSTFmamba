"""Behavioral regressions for citation synchronization and source packaging."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import shutil
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_docs import check_document
from prepare_archives import source_files, weight_files
from sync_publication import START, END, HEADER_START, HEADER_END, synchronize


class MaintenanceTests(unittest.TestCase):
    def test_citation_check_detects_drift_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            paper = {
                "name": "Demo",
                "title": "An accepted paper",
                "authors": [{"given": "Ada", "family": "Lovelace"}],
                "arxiv": "2601.00001",
                "year": 2026,
                "venue": "Example 2026",
                "booktitle": "Example Conference",
                "bibtex_key": "example2026",
                "status": "accepted",
                "doi": None,
            }
            (root / "docs/publication.json").write_text(json.dumps(paper))
            (root / "README.md").write_text(f"{HEADER_START}\n{HEADER_END}\n\n{START}\n{END}\n")
            synchronize(root)
            self.assertEqual(synchronize(root, check=True), [])
            citation = root / "CITATION.bib"
            citation.write_text("stale\n")
            self.assertEqual(synchronize(root, check=True), ["CITATION.bib"])
            self.assertEqual(citation.read_text(), "stale\n")
            synchronize(root)
            self.assertIn("Accepted at Example 2026", citation.read_text())
            self.assertNotIn("doi:", (root / "CITATION.cff").read_text())
            paper["authors"].append({"given": "Grace", "family": "Hopper"})
            (root / "docs/publication.json").write_text(json.dumps(paper))
            self.assertEqual(
                set(synchronize(root, check=True)), {"README.md", "CITATION.cff", "CITATION.bib"}
            )

    def test_local_document_links_and_fragments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            readme = root / "README.md"
            guide = root / "guide.md"
            guide.write_text("# Guide\n\n## Run `demo`\n")
            readme.write_text("[Usage](guide.md#run-demo)\n[Web](https://example.org)\n")
            self.assertEqual(check_document(readme, root), [])
            readme.write_text("[Usage](guide.md#missing)\n[Missing](absent.md)\n")
            self.assertEqual(len(check_document(readme, root)), 2)

    def test_source_package_excludes_runtime_files_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in [
                "README.md",
                "src/data/loader.py",
                ".env",
                ".env.local",
                ".DS_Store",
                "data/private.npy",
                "outputs/result.npy",
                "checkpoints/run/model.ckpt",
                "third_party/build/kernel.so",
                "release_bundles/source-only/a.txt",
                ".git/config",
                ".venv/private.txt",
                "cache/model.pt",
                "cache/model.tar.gz",
            ]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            (root / "link").symlink_to(root / "README.md")
            self.assertEqual(
                {p.relative_to(root).as_posix() for p in source_files(root)},
                {"README.md", "src/data/loader.py"},
            )

    def test_weight_manifest_rejects_paths_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../outside.pt", "/tmp/outside.pt"):
                with self.assertRaises(ValueError):
                    weight_files(root, {"artifacts": [{"file": name, "sha256": "0" * 64}]})

    @unittest.skipUnless(shutil.which("git"), "Git is optional in extracted source packages")
    def test_source_package_honors_repository_ignore_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("local-notes/\n")
            (root / "local-notes").mkdir()
            (root / "local-notes/private.txt").write_text("not for release")
            (root / "new-source.py").write_text("# Newly added source\n")
            files = {p.relative_to(root).as_posix() for p in source_files(root)}
            self.assertIn("new-source.py", files)
            self.assertNotIn("local-notes/private.txt", files)

    def test_source_only_build_does_not_require_declared_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tools").mkdir()
            (root / "docs").mkdir()
            (root / "weights").mkdir()
            shutil.copy2(
                Path(__file__).resolve().parents[1] / "tools/prepare_archives.py",
                root / "tools/prepare_archives.py",
            )
            (root / "docs/source_manifest.json").write_text('{"files": []}\n')
            (root / "weights/manifest.json").write_text(
                json.dumps(
                    {
                        "method": "Fixture",
                        "artifacts": [{"file": "checkpoints/missing.pt", "sha256": "0" * 64}],
                    }
                )
            )
            (root / "README.md").write_text("# Source-only fixture\n")
            result = subprocess.run(
                [sys.executable, str(root / "tools/prepare_archives.py"), "--source-only"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = root / "release_bundles/source-only"
            with zipfile.ZipFile(output / "Fixture-source.zip") as archive:
                self.assertIn("Fixture/README.md", archive.namelist())
                self.assertFalse(any("checkpoints/" in name for name in archive.namelist()))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["method_weights"], "not_requested")


if __name__ == "__main__":
    unittest.main()
