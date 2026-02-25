use assert_cmd::Command;
use std::path::PathBuf;

const BLEACH_EXPECTED_SCENES: &[usize] = &[
    0, 28, 151, 196, 243, 315, 343, 382, 436, 461, 476, 496, 526, 548, 579, 592, 607, 655, 709,
    748, 785, 799, 847, 883, 957,
];
const MATCH_TOLERANCE_FRAMES: usize = 2;
const MAX_FALSE_POSITIVES: usize = 1;
const MAX_FP_RATIO: f64 = 0.05;
const TWO_POW_32_F64: f64 = 4_294_967_296.0;

#[derive(Debug)]
struct SceneValidationMetrics {
    tp: usize,
    fp: usize,
    fn_count: usize,
    fp_ratio: f64,
    unmatched_expected: Vec<usize>,
    unmatched_predicted: Vec<usize>,
}

fn bleach_fixture_path() -> PathBuf {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".github/fixtures/bleach.h264");
    assert!(
        fixture.is_file(),
        "missing bleach fixture at {}",
        fixture.display()
    );
    fixture
}

fn run_frames_output(fixture: &PathBuf) -> Vec<usize> {
    let output = Command::cargo_bin("scuisei-rs")
        .unwrap()
        .arg("-i")
        .arg(fixture)
        .arg("--frames")
        .output()
        .expect("failed to run scuisei-rs --frames");

    assert!(
        output.status.success(),
        "scuisei-rs --frames failed with status {:?}\nstderr:\n{}",
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

fn validate_frames(
    expected: &[usize],
    predicted: &[usize],
    tolerance: usize,
) -> SceneValidationMetrics {
    let mut expected_idx = 0;
    let mut predicted_idx = 0;
    let mut tp = 0;
    let mut unmatched_expected = Vec::new();
    let mut unmatched_predicted = Vec::new();

    while expected_idx < expected.len() && predicted_idx < predicted.len() {
        let expected_frame = expected[expected_idx];
        let predicted_frame = predicted[predicted_idx];

        if predicted_frame.saturating_add(tolerance) < expected_frame {
            unmatched_predicted.push(predicted_frame);
            predicted_idx += 1;
            continue;
        }

        if expected_frame.saturating_add(tolerance) < predicted_frame {
            unmatched_expected.push(expected_frame);
            expected_idx += 1;
            continue;
        }

        tp += 1;
        expected_idx += 1;
        predicted_idx += 1;
    }

    unmatched_expected.extend_from_slice(&expected[expected_idx..]);
    unmatched_predicted.extend_from_slice(&predicted[predicted_idx..]);

    let fp = unmatched_predicted.len();
    let fn_count = unmatched_expected.len();
    let fp_ratio = usize_ratio_to_f64(fp, predicted.len());

    SceneValidationMetrics {
        tp,
        fp,
        fn_count,
        fp_ratio,
        unmatched_expected,
        unmatched_predicted,
    }
}

fn usize_ratio_to_f64(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        return 0.0;
    }

    let numerator_u64 = u64::try_from(numerator).unwrap_or(u64::MAX);
    let denominator_u64 = u64::try_from(denominator).unwrap_or(u64::MAX);
    if denominator_u64 == 0 {
        return 0.0;
    }

    u64_to_f64_exact(numerator_u64) / u64_to_f64_exact(denominator_u64)
}

fn u64_to_f64_exact(value: u64) -> f64 {
    let high = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low_mask = u64::from(u32::MAX);
    let low = u32::try_from(value & low_mask).unwrap_or(u32::MAX);
    (f64::from(high) * TWO_POW_32_F64) + f64::from(low)
}

#[test]
fn test_bleach_fixture_scene_changes_metrics() {
    let fixture = bleach_fixture_path();
    let predicted = run_frames_output(&fixture);

    assert!(
        predicted.windows(2).all(|window| window[0] < window[1]),
        "predicted frames must be strictly increasing: {predicted:?}"
    );

    let metrics = validate_frames(BLEACH_EXPECTED_SCENES, &predicted, MATCH_TOLERANCE_FRAMES);
    eprintln!(
        "bleach scene metrics: expected={} predicted={} tolerance={} tp={} fp={} fn={} fp_ratio={:.4} unmatched_expected={:?} unmatched_predicted={:?}",
        BLEACH_EXPECTED_SCENES.len(),
        predicted.len(),
        MATCH_TOLERANCE_FRAMES,
        metrics.tp,
        metrics.fp,
        metrics.fn_count,
        metrics.fp_ratio,
        metrics.unmatched_expected,
        metrics.unmatched_predicted
    );

    assert_eq!(
        metrics.fn_count, 0,
        "false negatives are not allowed on bleach fixture"
    );
    assert!(
        metrics.fp <= MAX_FALSE_POSITIVES,
        "too many false positives: {} > {} (unmatched predicted: {:?})",
        metrics.fp,
        MAX_FALSE_POSITIVES,
        metrics.unmatched_predicted
    );
    assert!(
        metrics.fp_ratio <= MAX_FP_RATIO,
        "false-positive ratio too high: {:.4} > {:.4}",
        metrics.fp_ratio,
        MAX_FP_RATIO
    );
}
