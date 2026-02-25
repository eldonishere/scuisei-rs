# scuisei-rs

![PyPI](https://img.shields.io/pypi/wheel/scuisei-rs) ![crates.io](https://img.shields.io/crates/msrv/:crate)

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
$ ./scuisei-rs -i input.mp4 -o output.pass
$ ./scuisei-rs -i input.mp4 --hwdec vaapi > output.pass
$ ./scuisei-rs -i input.mp4 --frames
$ ./scuisei-rs -i input.mp4 --native-res # slow - and default thresholds are tuned for the downsampled clip
```

## API (Rust)

```rust
use scuisei_rs::{AnalyzeOptions, analyze_video};

let options = AnalyzeOptions::defaults_for_input("input.mp4");
let result = analyze_video(&options)?;
println!("{:?}", result.keyframes);
```

`analyze_video` and output helpers return a typed `SCuiseiError` with stable categories: `config`, `io`, `decode`, `unsupported`, `internal`.

## API (Python)

```bash
uvx maturin develop --features python
uv run python -c "import scuisei_rs; print(scuisei_rs.detect_frames('input.mp4')); print(scuisei_rs.detect_pass('input.mp4'))"
```

## Release

After committing version bump: (`Cargo.toml`/`pyproject.toml`)

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Disclaimer

This was vibecoded.
