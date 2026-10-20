"""Build this repository's source and available method-weight archives locally."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_bundles"
EXCLUDE = {
    ".git",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".hydra",
    "build",
    "dist",
    ".venv",
}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_files(repository):
    """Prune runtime directories before traversal; also honor Git ignores when available."""
    runtime = {"data", "deps", "checkpoints", "logs", "outputs", "visual_datas", "release_bundles"}

    def excluded(name):
        return (
            name in EXCLUDE
            or name.endswith(".egg-info")
            or name == ".env"
            or name.startswith(".env.")
            or name == ".DS_Store"
        )

    candidates = []
    for directory, folders, files in os.walk(repository, followlinks=False):
        parent = Path(directory)
        folders[:] = sorted(
            name
            for name in folders
            if not excluded(name)
            and not (parent == repository and name in runtime)
            and not (parent / name).is_symlink()
        )
        for name in sorted(files):
            path = parent / name
            if excluded(name) or path.is_symlink():
                continue
            if path.suffix in {
                ".pyc",
                ".so",
                ".o",
                ".ckpt",
                ".pth",
                ".pt",
                ".tar",
                ".zip",
                ".gz",
                ".npy",
                ".npz",
                ".pkl",
                ".log",
            }:
                continue
            candidates.append(path)
    ignored = set()
    if (repository / ".git").exists() and candidates:
        try:
            result = subprocess.run(
                ["git", "check-ignore", "--no-index", "-z", "--stdin"],
                input="".join(p.relative_to(repository).as_posix() + "\0" for p in candidates),
                capture_output=True,
                text=True,
                cwd=repository,
            )
            if result.returncode in (0, 1):
                ignored = set(result.stdout.split("\0"))
        except FileNotFoundError:
            pass  # Extracted source packages do not require Git.
    for path in sorted(candidates):
        if path.relative_to(repository).as_posix() not in ignored:
            yield path


def weight_files(repository, manifest):
    """Resolve declared files before hashing or adding them to a weight archive."""
    files = []
    seen = set()
    for artifact in manifest["artifacts"]:
        for entry in [artifact, *artifact.get("support_files", [])]:
            relative = Path(entry["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Weight path must stay inside the repository: {relative}")
            path = repository / relative
            try:
                path.resolve().relative_to(repository.resolve())
            except ValueError as exc:
                raise ValueError(f"Weight path leaves the repository: {relative}") from exc
            if path.is_symlink():
                raise ValueError(f"Weight files must not be symlinks: {relative}")
            if not path.is_file() or digest(path) != entry["sha256"]:
                raise RuntimeError(f"Missing file or checksum mismatch: {relative}")
            if relative not in seen:
                files.append((path, relative.as_posix()))
                seen.add(relative)
    return files


def refresh_source_manifest():
    """Preserve provenance while recording the current distributable source."""
    path = ROOT / "docs/source_manifest.json"
    manifest = json.loads(path.read_text())
    previous = {entry["path"]: entry for entry in manifest["files"]}
    files = []
    for source in source_files(ROOT):
        if source == path:
            continue
        relative = source.relative_to(ROOT).as_posix()
        entry = previous.get(
            relative,
            {"path": relative, "source": "Added during release preparation"},
        )
        entry.update(
            release_sha256=digest(source),
            bytes=source.stat().st_size,
        )
        files.append(entry)
    manifest["files"] = files
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-only", action="store_true", help="Do not require or package weights."
    )
    args = parser.parse_args()
    output = OUTPUT / "source-only" if args.source_only else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "weights/manifest.json").read_text())
    method = manifest["method"]
    # Fail before writing packages when a declared weight is absent or changed.
    weights = [] if args.source_only else weight_files(ROOT, manifest)

    refresh_source_manifest()
    artifacts = []
    archive = output / f"{method}-source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in source_files(ROOT):
            stream.write(path, Path(method) / path.relative_to(ROOT))
    artifacts.append(archive)
    print(f"Prepared {archive.name}", flush=True)

    if weights:
        archive = output / f"{method}-weights.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as stream:
            for path, name in weights:
                stream.add(path, arcname=name)
            for name in [
                "MODEL_CARD.md",
                "LICENSE",
                "THIRD_PARTY_NOTICES.md",
                "third_party/CLIP_LICENSE",
                "weights/manifest.json",
            ]:
                stream.add(ROOT / name, arcname=name)
        artifacts.append(archive)
        print(f"Prepared {archive.name}", flush=True)

    entries = [
        {"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
        for path in artifacts
    ]
    (output / "SHA256SUMS").write_text(
        "".join(f"{entry['sha256']}  {entry['file']}\n" for entry in entries)
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "method": method,
                "artifacts": entries,
                "method_weights": "not_requested"
                if args.source_only
                else ("included" if weights else manifest.get("status", "not_available")),
                "remote_upload_performed": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("Prepared all archives and SHA256SUMS.", flush=True)


if __name__ == "__main__":
    main()
