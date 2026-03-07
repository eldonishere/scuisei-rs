mod common;

use assert_cmd::Command;
use predicates::str::contains;
use std::fs;
use std::path::Path;

#[test]
fn test_scxvid_output_compatibility() {
    let fixture_path = Path::new("target/fixtures/test_video.y4m");
    common::ensure_fixture_y4m(fixture_path);

    let output_path = "target/output.pass";

    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i")
        .arg(fixture_path)
        .arg("--format")
        .arg("xvid")
        .arg("-o")
        .arg(output_path)
        .assert()
        .success();

    let output_content = fs::read_to_string(output_path).unwrap();
    let lines: Vec<&str> = output_content.lines().collect();

    assert_eq!(
        lines.len(),
        6,
        "expected 2 header lines + spacer + 3 frame decisions"
    );
    assert_eq!(
        lines[0],
        &format!(
            "# XviD 2pass stat file (core version scuisei-rs {})",
            env!("CARGO_PKG_VERSION")
        )
    );
    assert_eq!(lines[1], "# Please do not modify this file");
    assert_eq!(lines[2], "");
    assert_eq!(lines[3], "i");
    assert_eq!(lines[4], "p");
    assert_eq!(lines[5], "p");
}

#[test]
fn test_frames_output() {
    let fixture_path = Path::new("target/fixtures/test_video.y4m");
    common::ensure_fixture_y4m(fixture_path);

    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i")
        .arg(fixture_path)
        .arg("--format")
        .arg("frames");

    cmd.assert().success().stdout("0,1,2\n");
}

#[test]
fn test_default_output_is_agi() {
    let fixture_path = Path::new("target/fixtures/test_video.y4m");
    common::ensure_fixture_y4m(fixture_path);

    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i").arg(fixture_path);

    cmd.assert()
        .success()
        .stdout("# keyframe format v1\nfps 0\n\n0 I -1\n1 I -1\n2 I -1\n");
}

#[test]
fn test_missing_input_fails_with_context() {
    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i")
        .arg("target/fixtures/does-not-exist.y4m")
        .arg("--format")
        .arg("frames");

    cmd.assert()
        .failure()
        .stderr(contains("failed to open input"));
}

#[test]
fn test_invalid_hwdec_fails_with_context() {
    let fixture_path = Path::new("target/fixtures/test_video.y4m");
    common::ensure_fixture_y4m(fixture_path);

    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i")
        .arg(fixture_path)
        .arg("--hwdec")
        .arg("not-a-device");

    cmd.assert().failure().stderr(contains("unknown --hwdec"));
}

#[test]
fn test_invalid_window_size_is_rejected_by_cli() {
    let fixture_path = Path::new("target/fixtures/test_video.y4m");
    common::ensure_fixture_y4m(fixture_path);

    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i")
        .arg(fixture_path)
        .arg("--window-size")
        .arg("0");

    cmd.assert()
        .failure()
        .stderr(contains("must be greater than 0"));
}

#[test]
fn test_zero_detector_weights_fail_with_config_error() {
    let fixture_path = Path::new("target/fixtures/test_video.y4m");
    common::ensure_fixture_y4m(fixture_path);

    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i")
        .arg(fixture_path)
        .arg("--sad-weight")
        .arg("0")
        .arg("--hist-weight")
        .arg("0");

    cmd.assert().failure().stderr(contains(
        "adaptive_config.sad_weight + adaptive_config.hist_weight must be greater than 0",
    ));
}
