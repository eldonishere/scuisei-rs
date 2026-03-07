use assert_cmd::Command;
use std::env;

#[test]
#[ignore = "requires SCUISEI_HWDEC_INPUT and SCUISEI_HWDEC_DEVICE with a real hardware decode setup"]
fn test_hwdec_success_smoke() {
    let input = env::var("SCUISEI_HWDEC_INPUT")
        .expect("set SCUISEI_HWDEC_INPUT to a decodable media file for this smoke test");
    let device = env::var("SCUISEI_HWDEC_DEVICE")
        .expect("set SCUISEI_HWDEC_DEVICE to a supported hwaccel name such as vaapi/qsv/cuda");

    let mut cmd = Command::cargo_bin("scuisei-rs").unwrap();
    cmd.arg("-i")
        .arg(input)
        .arg("--hwdec")
        .arg(device)
        .arg("--format")
        .arg("frames");

    cmd.assert().success();
}
