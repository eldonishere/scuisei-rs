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
        .arg("-o")
        .arg(output_path)
        .assert()
        .success();

    let output_content = fs::read_to_string(output_path).unwrap();
    let lines: Vec<&str> = output_content.lines().collect();

    assert_eq!(lines.len(), 4, "expected header + 3 frame decisions");
    assert_eq!(lines[0], "# xvid 2pass log file");
    assert_eq!(lines[1], "i");
    assert_eq!(lines[2], "p");
    assert_eq!(lines[3], "p");
}

#[test]
fn test_frames_output() {
    let fixture_path = Path::new("target/fixtures/test_video.y4m");
    common::ensure_fixture_y4m(fixture_path);

    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i").arg(fixture_path).arg("--frames");

    cmd.assert().success().stdout("0,1,2\n");
}

#[test]
fn test_missing_input_fails_with_context() {
    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i")
        .arg("target/fixtures/does-not-exist.y4m")
        .arg("--frames");

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
