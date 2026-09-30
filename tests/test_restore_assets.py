"""Restoration must reject corruption/traversal and preserve edited local data."""
import hashlib
import zipfile

import pytest

from scripts.restore_assets import download, restore


def make_archive(tmp_path, name="dataset/data.csv", contents=b"value\n1\n"):
    path = tmp_path / "asset.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, contents)
    asset = {"name": path.name, "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "root": "dataset",
             "file_count": 1, "expanded_bytes": len(contents)}
    return path, asset


def test_restore_is_verified_and_idempotent(tmp_path):
    archive, asset = make_archive(tmp_path)
    dest = tmp_path / "restore"
    folder = restore(archive, asset, dest)
    assert (folder / "data.csv").read_bytes() == b"value\n1\n"
    assert restore(archive, asset, dest) == folder
    (folder / "data.csv").write_bytes(b"value\n2\n")
    with pytest.raises(ValueError, match="edited"):
        restore(archive, asset, dest)
    assert (folder / "data.csv").read_bytes() == b"value\n2\n"


def test_corrupt_archive_not_extracted(tmp_path):
    archive, asset = make_archive(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"corruption")
    dest = tmp_path / "restore"
    with pytest.raises(ValueError, match="Checksum"):
        restore(archive, asset, dest)
    assert not dest.exists()


@pytest.mark.parametrize("name", ["dataset/../../escape.txt", "/absolute.txt", "C:/escape.txt",
                                  "dataset\\..\\escape.txt", "other/data.csv"])
def test_unsafe_paths_never_extracted(tmp_path, name):
    archive, asset = make_archive(tmp_path, name)
    dest = tmp_path / "restore"
    with pytest.raises(ValueError, match="Unsafe"):
        restore(archive, asset, dest)
    assert not list(dest.iterdir())


def test_download_reuses_verified_file_without_gh(tmp_path, monkeypatch):
    archive, asset = make_archive(tmp_path)
    monkeypatch.setattr("scripts.restore_assets.subprocess.run", lambda *a, **kw: pytest.fail("Unexpected network"))
    assert download(asset, {}, tmp_path, "unused/repo") == archive
