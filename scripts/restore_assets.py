"""Download and verify the private migration release; works after cloning or forking.

Requires GitHub CLI authentication for downloads. --archive-dir supports offline restore.
No analysis dependencies are needed unless --prepare is requested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_archive(path: Path, asset: dict) -> None:
    if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
        raise ValueError(f"Checksum/size mismatch: {path}. Do not extract this file.")


def download(asset: dict, manifest: dict, cache: Path, repository: str) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / asset["name"]
    if target.exists():
        verify_archive(target, asset)
        return target
    if shutil.which("gh") is None:
        raise RuntimeError("Install GitHub CLI and run 'gh auth login', or supply --archive-dir.")
    if shutil.disk_usage(cache).free < asset["bytes"] + asset["expanded_bytes"] + 100 * 1024**2:
        raise RuntimeError(f"Insufficient space on the drive containing {cache}.")
    for attempt in range(1, 4):
        with tempfile.TemporaryDirectory(prefix="download-", dir=cache) as tmp:
            try:
                subprocess.run(["gh", "release", "download", manifest["release_tag"], "--repo", repository,
                                "--pattern", asset["name"], "--dir", tmp], check=True)
            except subprocess.CalledProcessError:
                if attempt == 3:
                    raise
                print(f"Download attempt {attempt} failed; retrying {asset['name']}.", flush=True)
                time.sleep(2)
                continue
            candidate = Path(tmp) / asset["name"]
            verify_archive(candidate, asset)
            candidate.rename(target)
            break
    return target


def validated_members(archive: zipfile.ZipFile, asset: dict) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    files = [m for m in members if not m.is_dir()]
    if len(files) != asset["file_count"] or sum(m.file_size for m in files) != asset["expanded_bytes"]:
        raise ValueError("Archive inventory does not match the pinned manifest.")
    seen = set()
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        if ("\\" in name or ":" in name or path.is_absolute() or
                any(p in (".", "..") for p in name.split("/")) or
                not path.parts or path.parts[0] != asset["root"] or
                stat.S_ISLNK(member.external_attr >> 16) or name.casefold() in seen):
            raise ValueError(f"Unsafe archive member: {name}")
        seen.add(name.casefold())
    return files


def restore(path: Path, asset: dict, destination: Path) -> Path:
    verify_archive(path, asset)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / asset["root"]
    with zipfile.ZipFile(path) as archive:
        files = validated_members(archive, asset)
        if target.exists():
            # Never silently overwrite a previously edited research/dataset folder.
            for member in files:
                existing = destination / member.filename
                if not existing.resolve().is_relative_to(destination) or existing.is_symlink():
                    raise ValueError(f"Unsafe existing path: {existing}")
                if not existing.is_file() or existing.stat().st_size != member.file_size:
                    raise ValueError(f"Existing restore is incomplete/changed: {existing}. Use a new --destination.")
                with archive.open(member) as stream:
                    h = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                if sha256(existing) != h.hexdigest():
                    raise ValueError(f"Existing restore was edited: {existing}. Use a new --destination.")
            print(f"Verified existing restore: {target}")
            return target
        if shutil.disk_usage(destination).free < asset["expanded_bytes"] + 100 * 1024**2:
            raise RuntimeError(f"Insufficient extraction space in {destination}.")
        # Extract only after validating every name; publish the directory atomically.
        with tempfile.TemporaryDirectory(prefix="extract-", dir=destination) as tmp:
            archive.extractall(tmp)
            (Path(tmp) / asset["root"]).rename(target)
    print(f"Restored {len(files)} files to {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-research", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Restore the app's data/local/ tables only")
    parser.add_argument("--archive-dir", type=Path, help="Use downloaded ZIPs here without network access")
    parser.add_argument("--destination", type=Path, default=ROOT / "runtime" / "datasets")
    parser.add_argument("--repo", help="Override the release repository if assets were copied to another repo")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data" / "dataset_release.json").read_text(encoding="utf-8"))
    dataset = None
    for asset in manifest["assets"]:
        if asset["kind"] == "research" and not args.include_research:
            continue
        path = args.archive_dir / asset["name"] if args.archive_dir else download(
            asset, manifest, ROOT / "runtime" / "downloads", args.repo or manifest["repository"])
        folder = restore(path, asset, args.destination)
        if asset["kind"] == "dataset":
            dataset = folder
            subprocess.run([sys.executable, str(folder / "scripts" / "verify_hashes.py")], check=True)
    if args.prepare and dataset is not None:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_data.py"),
                        "--package-dir", str(dataset), "--local-only"], cwd=ROOT, check=True)
    print("Restore complete. IGBT remains quarantined and is not used by the screening models.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"Restore failed: {exc}") from exc
