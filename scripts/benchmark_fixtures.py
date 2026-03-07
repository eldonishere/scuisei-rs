#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = {
    "bleach": REPO_ROOT / ".github" / "fixtures" / "bleach.h264",
    "monogatari": REPO_ROOT / ".github" / "fixtures" / "monogatari.h265",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark checked-in fixture videos with scuisei-rs --format frames."
    )
    parser.add_argument("--binary", type=Path, required=True, help="Path to scuisei-rs binary")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Measured runs per fixture (default: 5)",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Warmup runs per fixture (default: 1)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write raw benchmark results as JSON",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="Optional baseline JSON to compare medians against",
    )
    return parser.parse_args()


def run_once(binary: Path, fixture: Path) -> float:
    start = time.perf_counter()
    completed = subprocess.run(
        [str(binary), "-i", str(fixture), "--format", "frames"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark command failed for {fixture}:\n{completed.stderr.strip()}"
        )
    return elapsed


def summarize(times: list[float]) -> dict[str, float]:
    if not times:
        raise ValueError("expected at least one measured run")
    mean_s = statistics.fmean(times)
    stdev_s = statistics.stdev(times) if len(times) > 1 else 0.0
    return {
        "median_s": statistics.median(times),
        "mean_s": mean_s,
        "stdev_s": stdev_s,
        "min_s": min(times),
        "max_s": max(times),
    }


def load_baseline(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    fixtures = payload.get("fixtures", [])
    return {entry["name"]: entry for entry in fixtures}


def build_payload(args: argparse.Namespace, baseline: dict[str, Any]) -> dict[str, Any]:
    binary = args.binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"missing benchmark binary: {binary}")
    if args.runs <= 0:
        raise ValueError("--runs must be greater than 0")
    if args.warmups < 0:
        raise ValueError("--warmups must be greater than or equal to 0")

    fixture_entries: list[dict[str, Any]] = []
    for name, fixture in DEFAULT_FIXTURES.items():
        if not fixture.is_file():
            raise FileNotFoundError(f"missing fixture: {fixture}")

        for _ in range(args.warmups):
            run_once(binary, fixture)

        times = [run_once(binary, fixture) for _ in range(args.runs)]
        summary = summarize(times)
        entry: dict[str, Any] = {
            "name": name,
            "path": str(fixture.relative_to(REPO_ROOT)),
            "times_s": times,
            **summary,
        }
        baseline_entry = baseline.get(name)
        if baseline_entry is not None:
            baseline_median = float(baseline_entry["median_s"])
            entry["baseline_median_s"] = baseline_median
            entry["speedup_vs_baseline"] = (
                baseline_median / summary["median_s"] if summary["median_s"] > 0 else 0.0
            )
        fixture_entries.append(entry)

    return {
        "binary": str(binary),
        "runs": args.runs,
        "warmups": args.warmups,
        "fixtures": fixture_entries,
    }


def format_float(value: float) -> str:
    return f"{value:.3f}"


def print_markdown(payload: dict[str, Any]) -> None:
    has_baseline = any("speedup_vs_baseline" in entry for entry in payload["fixtures"])
    columns = [
        "fixture",
        "median_s",
        "mean_s",
        "stdev_s",
        "min_s",
        "max_s",
    ]
    if has_baseline:
        columns.append("speedup_vs_baseline")

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    print(header)
    print(divider)
    for entry in payload["fixtures"]:
        row = [
            entry["name"],
            format_float(entry["median_s"]),
            format_float(entry["mean_s"]),
            format_float(entry["stdev_s"]),
            format_float(entry["min_s"]),
            format_float(entry["max_s"]),
        ]
        if has_baseline:
            speedup = entry.get("speedup_vs_baseline")
            row.append(f"{speedup:.2f}x" if speedup is not None else "-")
        print("| " + " | ".join(row) + " |")


def main() -> int:
    args = parse_args()
    try:
        baseline = load_baseline(args.baseline_json)
        payload = build_payload(args, baseline)
    except Exception as error:  # noqa: BLE001
        print(str(error), file=sys.stderr)
        return 1

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    print_markdown(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
