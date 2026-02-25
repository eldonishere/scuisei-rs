#!/usr/bin/env python3
from pathlib import Path


def ensure_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 16
    height = 16
    header = f"YUV4MPEG2 W{width} H{height} F30:1 Ip A0:0 C420\n".encode()
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    payload = bytearray(header)
    for y_value in (0, 128, 255):
        payload.extend(b"FRAME\n")
        payload.extend(bytes([y_value]) * y_size)
        payload.extend(bytes([128]) * uv_size)
        payload.extend(bytes([128]) * uv_size)
    path.write_bytes(payload)


def main() -> None:
    import scuisei_rs

    fixture = Path("target/fixtures/python_test_video.y4m")
    ensure_fixture(fixture)

    frames = scuisei_rs.detect_frames(str(fixture))
    assert frames == [0, 1, 2], f"unexpected detect_frames output: {frames}"

    pass_output = scuisei_rs.detect_pass(str(fixture))
    lines = pass_output.splitlines()
    assert len(lines) >= 6, f"unexpected detect_pass output: {pass_output!r}"
    assert lines[0].startswith("# XviD 2pass stat file (core version scuisei-rs "), lines
    assert lines[0].endswith(")"), lines
    assert lines[1] == "# Please do not modify this file"
    assert lines[2] == ""
    assert lines[3:] == ["i", "p", "p"], lines


if __name__ == "__main__":
    main()
