use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::sync::OnceLock;

static FIXTURE_INIT: OnceLock<()> = OnceLock::new();

pub fn ensure_fixture_y4m(path: &Path) {
    FIXTURE_INIT.get_or_init(|| {
        let parent = path.parent().expect("fixture should have parent dir");
        fs::create_dir_all(parent).expect("failed to create fixture directory");

        let width: usize = 16;
        let height: usize = 16;
        let header = format!("YUV4MPEG2 W{width} H{height} F30:1 Ip A0:0 C420\n");

        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(header.as_bytes());

        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);

        for &y_value in &[0_u8, 128_u8, 255_u8] {
            bytes.extend_from_slice(b"FRAME\n");
            bytes.extend(std::iter::repeat_n(y_value, y_size));
            bytes.extend(std::iter::repeat_n(128_u8, uv_size));
            bytes.extend(std::iter::repeat_n(128_u8, uv_size));
        }

        let mut file = fs::File::create(path).expect("failed to create fixture file");
        file.write_all(&bytes)
            .expect("failed to write fixture bytes");
    });
}
