"""Add license and notice files to built wheels.

The wheel RECORD file must be updated after modifying the archive; this script
rewrites it with hashes for all files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path


def record_hash(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}", len(data)


def rewrite_record(root: Path, dist_info: Path) -> None:
    record = dist_info / "RECORD"
    rows: list[list[str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if path == record:
            rows.append([rel, "", ""])
            continue
        digest, size = record_hash(path)
        rows.append([rel, digest, str(size)])

    with record.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)


def add_notices_to_wheel(wheel: Path, notice_files: list[Path]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "wheel"
        root.mkdir()
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)

        dist_info_dirs = sorted(root.glob("*.dist-info"))
        if len(dist_info_dirs) != 1:
            raise RuntimeError(f"{wheel}: expected one .dist-info directory")
        licenses_dir = dist_info_dirs[0] / "licenses"
        licenses_dir.mkdir(exist_ok=True)

        for source in notice_files:
            shutil.copy2(source, licenses_dir / source.name)

        rewrite_record(root, dist_info_dirs[0])

        rebuilt = wheel.with_suffix(".tmp.whl")
        with zipfile.ZipFile(rebuilt, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                archive.write(path, path.relative_to(root).as_posix())
        rebuilt.replace(wheel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-glob", required=True)
    parser.add_argument("notice_files", nargs="+")
    args = parser.parse_args()

    wheels = [Path(path) for path in glob.glob(args.wheel_glob)]
    if not wheels:
        raise SystemExit(f"no wheels matched {args.wheel_glob!r}")

    notice_files = [Path(path) for path in args.notice_files]
    for notice in notice_files:
        if not notice.is_file():
            raise SystemExit(f"notice file not found: {notice}")

    for wheel in wheels:
        add_notices_to_wheel(wheel, notice_files)


if __name__ == "__main__":
    main()
