<p>
    <a href="https://pypi.org/project/scuisei-rs/" alt="PyPI">
        <img src="https://img.shields.io/pypi/v/scuisei-rs" /></a>
    <a href="https://crates.io/crates/scuisei-rs" alt="Cargo">
        <img src="https://img.shields.io/crates/v/scuisei-rs" /></a>
</p>

# scuisei-rs

Blazing fast successor for [SCXvid](https://github.com/soyokaze/SCXvid-standalone), with arguably better scene change detections than existing solutions. Also outputs compatible `.pass` files.
Intended for fansubbing (e.g. timing within Aegisub) but may have other uses.

## Build

Requires Rust and FFmpeg development libraries (libavcodec/libavformat/libswscale + `pkg-config`).

```bash
cargo build --release
```

## Usage (CLI)

```bash
$ ./scuisei-rs --help
$ ./scuisei-rs -i input.mp4 > output.agi
$ ./scuisei-rs -i input.mp4 --format xvid -o output.pass
$ ./scuisei-rs -i input.mp4 --format xvid --hwdec vaapi > output.pass
$ ./scuisei-rs -i input.mp4 --format frames
$ ./scuisei-rs -i input.mp4 --native-res # slow - and default thresholds are tuned for the downsampled clip
```

`--hwdec` keeps decode on the requested device when possible, but frames are still transferred back to CPU memory for analysis, so end-to-end speedups depend on the input and hardware stack.

## Benchmarking

The repo includes a fixture benchmark script for the checked-in `bleach` and `monogatari` clips.

```bash
cargo build --release
python3 scripts/benchmark_fixtures.py --binary target/release/scuisei-rs --output-json target/bench/baseline.json

# after making changes
cargo build --release
python3 scripts/benchmark_fixtures.py \
  --binary target/release/scuisei-rs \
  --output-json target/bench/final.json \
  --baseline-json target/bench/baseline.json
```

The script benchmarks `target/release/scuisei-rs -i <fixture> --format frames`, prints a Markdown table, and writes JSON with per-fixture `median_s`, `mean_s`, `stdev_s`, `min_s`, `max_s`, and optional `speedup_vs_baseline` fields.

## API (Rust)

```rust
use scuisei_rs::{AnalyzeOptions, PostprocessConfig, analyze_video};

let mut options = AnalyzeOptions::defaults_for_input("input.mp4");
options.postprocess_config = PostprocessConfig::default();
let result = analyze_video(&options)?;
println!("{:?}", result.keyframes);
```

`analyze_video` and output helpers return a typed `SCuiseiError` with stable categories: `config`, `io`, `decode`, `unsupported`, `internal`.

## API (Python)

```bash
uvx maturin develop --features python
uv run python -c "import scuisei_rs; print(scuisei_rs.detect_frames('input.mp4')); print(scuisei_rs.detect_pass('input.mp4'))"
```

Python also exposes `scuisei_rs.PostprocessConfig()` for optional keyframe postprocess tuning.

## Release

After committing version bump: (`Cargo.toml`/`pyproject.toml`)

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Disclaimer

This was vibecoded.
