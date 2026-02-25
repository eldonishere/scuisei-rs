mod common;

use assert_cmd::Command;
use scuisei_rs::{AnalyzeOptions, SCuiseiError, analyze_video, write_frames_csv, write_pass_log};
use std::fs;
use std::path::Path;

#[test]
fn test_library_api_matches_expected_fixture_outputs() {
    let fixture_path = Path::new("target/fixtures/test_video.y4m");
    common::ensure_fixture_y4m(fixture_path);

    let options = AnalyzeOptions::defaults_for_input(fixture_path);
    let result = analyze_video(&options).expect("analysis should succeed");

    assert_eq!(result.keyframes, vec![0, 1, 2]);
    assert_eq!(result.pass_decisions, vec![true, false, false]);
    assert_eq!(result.frame_count(), 3);

    let mut frames_output = Vec::new();
    write_frames_csv(&mut frames_output, &result.keyframes).expect("frames output should format");
    let frames_string = String::from_utf8(frames_output).expect("frames output should be UTF-8");
    assert_eq!(frames_string, "0,1,2\n");

    let mut pass_output = Vec::new();
    write_pass_log(&mut pass_output, &result.pass_decisions).expect("pass output should format");
    let pass_string = String::from_utf8(pass_output).expect("pass output should be UTF-8");
    assert_eq!(pass_string, "# xvid 2pass log file\ni\np\np\n");
}

#[test]
fn test_cli_and_library_outputs_match() {
    let fixture_path = Path::new("target/fixtures/test_video.y4m");
    common::ensure_fixture_y4m(fixture_path);

    let options = AnalyzeOptions::defaults_for_input(fixture_path);
    let result = analyze_video(&options).expect("analysis should succeed");

    let mut expected_frames = Vec::new();
    write_frames_csv(&mut expected_frames, &result.keyframes).expect("frames output should format");
    let expected_frames =
        String::from_utf8(expected_frames).expect("frames output should be UTF-8");

    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i").arg(fixture_path).arg("--frames");
    cmd.assert().success().stdout(expected_frames);

    let mut expected_pass = Vec::new();
    write_pass_log(&mut expected_pass, &result.pass_decisions).expect("pass output should format");
    let expected_pass = String::from_utf8(expected_pass).expect("pass output should be UTF-8");

    let output_path = "target/output.cli.parity.pass";
    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i").arg(fixture_path).arg("-o").arg(output_path);
    cmd.assert().success();

    let actual_pass = fs::read_to_string(output_path).expect("pass file should exist");
    assert_eq!(actual_pass, expected_pass);
}

#[test]
fn test_library_api_classifies_missing_input_as_io_error() {
    let options = AnalyzeOptions::defaults_for_input("target/fixtures/does-not-exist.y4m");
    let error = analyze_video(&options).expect_err("missing input should fail");
    assert!(matches!(error, SCuiseiError::Io(_)));
}
