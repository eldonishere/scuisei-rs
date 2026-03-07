mod common;

use assert_cmd::Command;
use scuisei_rs::{
    AnalyzeOptions, SCuiseiError, analyze_keyframes, analyze_pass_decisions, analyze_video,
    write_agi, write_frames_csv, write_pass_log,
};
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

    let mut agi_output = Vec::new();
    write_agi(&mut agi_output, &result.keyframes).expect("agi output should format");
    let agi_string = String::from_utf8(agi_output).expect("agi output should be UTF-8");
    assert_eq!(
        agi_string,
        "# keyframe format v1\nfps 0\n\n0 I -1\n1 I -1\n2 I -1\n"
    );

    let mut pass_output = Vec::new();
    write_pass_log(&mut pass_output, &result.pass_decisions).expect("pass output should format");
    let pass_string = String::from_utf8(pass_output).expect("pass output should be UTF-8");
    assert_eq!(
        pass_string,
        format!(
            "# XviD 2pass stat file (core version scuisei-rs {})\n# Please do not modify this file\n\ni\np\np\n",
            env!("CARGO_PKG_VERSION")
        )
    );
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
    cmd.arg("-i")
        .arg(fixture_path)
        .arg("--format")
        .arg("frames");
    cmd.assert().success().stdout(expected_frames);

    let mut expected_pass = Vec::new();
    write_pass_log(&mut expected_pass, &result.pass_decisions).expect("pass output should format");
    let expected_pass = String::from_utf8(expected_pass).expect("pass output should be UTF-8");

    let output_path = "target/output.cli.parity.pass";
    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i")
        .arg(fixture_path)
        .arg("--format")
        .arg("xvid")
        .arg("-o")
        .arg(output_path);
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

#[test]
fn test_library_api_rejects_invalid_config_as_config_error() {
    let fixture_path = Path::new("target/fixtures/test_video.y4m");
    common::ensure_fixture_y4m(fixture_path);

    let mut options = AnalyzeOptions::defaults_for_input(fixture_path);
    options.adaptive_config.window_size = 0;
    let error = analyze_video(&options).expect_err("invalid config should fail");
    assert!(matches!(error, SCuiseiError::Config(_)));

    options.adaptive_config.window_size = 30;
    options.adaptive_config.sad_weight = 0.0;
    options.adaptive_config.hist_weight = 0.0;
    let error = analyze_video(&options).expect_err("zero detector weights should fail");
    assert!(matches!(error, SCuiseiError::Config(_)));
}

#[test]
fn test_specialized_analysis_paths_match_full_analysis() {
    let fixture_path = Path::new("target/fixtures/test_video.y4m");
    common::ensure_fixture_y4m(fixture_path);

    let options = AnalyzeOptions::defaults_for_input(fixture_path);
    let full = analyze_video(&options).expect("full analysis should succeed");
    let keyframes = analyze_keyframes(&options).expect("keyframe analysis should succeed");
    let pass_decisions =
        analyze_pass_decisions(&options).expect("pass-decision analysis should succeed");

    assert_eq!(keyframes, full.keyframes);
    assert_eq!(pass_decisions, full.pass_decisions);
}
