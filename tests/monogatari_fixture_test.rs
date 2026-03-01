use assert_cmd::Command;
use std::path::PathBuf;

const MONOGATARI_EXPECTED_SCENES: &[usize] = &[
    0, 27, 54, 96, 111, 129, 157, 188, 218, 234, 264, 282, 296, 332, 363, 400, 432, 447, 465, 481,
    495, 520, 540, 564, 612, 660, 672, 720,
];

fn monogatari_fixture_path() -> PathBuf {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".github/fixtures/monogatari.h265");
    assert!(
        fixture.is_file(),
        "missing monogatari fixture at {}",
        fixture.display()
    );
    fixture
}

fn run_frames_output(fixture: &PathBuf) -> Vec<usize> {
    let output = Command::cargo_bin("scuisei-rs")
        .unwrap()
        .arg("-i")
        .arg(fixture)
        .arg("--format")
        .arg("frames")
        .output()
        .expect("failed to run scuisei-rs --format frames");

    assert!(
        output.status.success(),
        "scuisei-rs --format frames failed with status {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    parse_frame_list(&String::from_utf8(output.stdout).expect("stdout should be valid UTF-8"))
}

fn parse_frame_list(stdout: &str) -> Vec<usize> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    trimmed
        .split(',')
        .map(|token| {
            token
                .trim()
                .parse::<usize>()
                .unwrap_or_else(|err| panic!("invalid frame index '{token}': {err}"))
        })
        .collect()
}

#[test]
fn test_monogatari_fixture_scene_changes_exact() {
    let fixture = monogatari_fixture_path();
    let predicted = run_frames_output(&fixture);

    assert!(
        predicted.windows(2).all(|window| window[0] < window[1]),
        "predicted frames must be strictly increasing: {predicted:?}"
    );
    assert_eq!(
        predicted, MONOGATARI_EXPECTED_SCENES,
        "monogatari scene output mismatch"
    );
}
