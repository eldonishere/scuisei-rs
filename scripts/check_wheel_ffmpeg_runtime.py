#!/usr/bin/env python3
"""Fail if a wheel does not bundle expected FFmpeg runtime libraries."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import zipfile
from pathlib import PurePosixPath

FFMPEG_COMPONENTS = ("avcodec", "avformat", "avutil", "swscale")


def _matches_extension(filename: str, platform: str) -> bool:
    if platform == "windows":
        return filename.endswith(".dll")
    if platform == "macos":
        return ".dylib" in filename
    return ".so" in filename


def _component_present(files: list[str], component: str, platform: str) -> bool:
    for path in files:
        base = PurePosixPath(path).name.lower()
        if component in base and _matches_extension(base, platform):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        choices=("linux", "macos", "windows"),
        required=True,
        help="Platform of the wheel being validated.",
    )
    parser.add_argument(
        "--wheel-glob",
        default="dist/*.whl",
        help="Glob that resolves to one built wheel.",
    )
    parser.add_argument(
        "--allow-static",
        action="store_true",
        help="Allow wheels without bundled FFmpeg shared libraries (for static linking).",
    )
    args = parser.parse_args()

    wheels = sorted(glob.glob(args.wheel_glob))
    if not wheels:
        print(f"No wheels matched glob: {args.wheel_glob}", file=sys.stderr)
        return 1
    if len(wheels) > 1:
        print(f"Expected one wheel, found {len(wheels)}: {wheels}", file=sys.stderr)
        return 1

    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        files = archive.namelist()

    missing = [
        component
        for component in FFMPEG_COMPONENTS
        if not _component_present(files, component, args.platform)
    ]
    if missing:
        if args.allow_static and args.platform == "linux":
            size_mb = os.path.getsize(wheel) / (1024 * 1024)
            print(
                "No bundled FFmpeg shared libraries detected; "
                f"assuming static linking in {os.path.basename(wheel)} ({size_mb:.1f} MB)."
            )
            return 0
        print(
            f"Wheel {os.path.basename(wheel)} is missing bundled FFmpeg runtime libraries: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    size_mb = os.path.getsize(wheel) / (1024 * 1024)
    print(
        f"Verified bundled FFmpeg runtimes in {os.path.basename(wheel)} ({size_mb:.1f} MB)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
