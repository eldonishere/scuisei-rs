use clap::{Parser, ValueEnum};
use std::path::PathBuf;

pub const DEFAULT_WINDOW_SIZE: usize = 30;
pub const DEFAULT_SIGMA: f64 = 3.0;
pub const DEFAULT_BASE_THRESHOLD: f64 = 0.20;
pub const DEFAULT_MIN_HIST_DISTANCE: f64 = 0.03;
pub const DEFAULT_MIN_SCORE: f64 = 0.0;
pub const DEFAULT_SAD_WEIGHT: f64 = 0.70;
pub const DEFAULT_HIST_WEIGHT: f64 = 0.30;
pub const DEFAULT_ME_SEARCH_RADIUS: i32 = 4;
pub const DEFAULT_ME_INTRA_THRESH: i32 = 2000;
pub const DEFAULT_ME_INTRA_THRESH2: f64 = 90.0;

fn parse_positive_usize(raw: &str) -> Result<usize, String> {
    let value = raw.parse::<usize>().map_err(|error| error.to_string())?;
    if value == 0 {
        return Err("must be greater than 0".to_string());
    }
    Ok(value)
}

fn parse_nonnegative_i32(raw: &str) -> Result<i32, String> {
    let value = raw.parse::<i32>().map_err(|error| error.to_string())?;
    if value < 0 {
        return Err("must be greater than or equal to 0".to_string());
    }
    Ok(value)
}

fn parse_nonnegative_f64(raw: &str) -> Result<f64, String> {
    let value = raw.parse::<f64>().map_err(|error| error.to_string())?;
    if !value.is_finite() || value < 0.0 {
        return Err("must be a finite number greater than or equal to 0".to_string());
    }
    Ok(value)
}

fn parse_unit_interval_f64(raw: &str) -> Result<f64, String> {
    let value = parse_nonnegative_f64(raw)?;
    if value > 1.0 {
        return Err("must be a finite number between 0 and 1".to_string());
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum OutputFormat {
    Agi,
    Xvid,
    Frames,
}

#[derive(Parser, Debug)]
#[command(
    name = "scuisei-rs",
    about = "SIMD-accelerated scene-change detection (scxvid-compatible output)",
    version
)]
pub struct Cli {
    /// Input media file
    #[arg(short = 'i', long)]
    pub input: PathBuf,

    /// Output .pass file path (defaults to stdout)
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Output format: agi, xvid, or frames
    #[arg(long, value_enum, default_value_t = OutputFormat::Agi)]
    pub format: OutputFormat,

    /// Run detection at native resolution instead of the default downscaled working resolution
    #[arg(long)]
    pub native_res: bool,

    /// Sliding window size (frames) for adaptive thresholding
    #[arg(long, value_parser = parse_positive_usize, default_value_t = DEFAULT_WINDOW_SIZE)]
    pub window_size: usize,

    /// Threshold = mean + sigma * stddev (within the sliding window)
    #[arg(long, value_parser = parse_nonnegative_f64, default_value_t = DEFAULT_SIGMA)]
    pub sigma: f64,

    /// Lower bound for the adaptive threshold
    #[arg(long, value_parser = parse_unit_interval_f64, default_value_t = DEFAULT_BASE_THRESHOLD)]
    pub base_threshold: f64,

    /// Minimum histogram distance gating (helps avoid motion-only false positives)
    #[arg(long, value_parser = parse_unit_interval_f64, default_value_t = DEFAULT_MIN_HIST_DISTANCE)]
    pub min_hist_distance: f64,

    /// Minimum combined score gating (useful during warm-up)
    #[arg(long, value_parser = parse_unit_interval_f64, default_value_t = DEFAULT_MIN_SCORE)]
    pub min_score: f64,

    /// Weight of SAD (normalized to 0..1)
    #[arg(long, value_parser = parse_nonnegative_f64, default_value_t = DEFAULT_SAD_WEIGHT)]
    pub sad_weight: f64,

    /// Weight of histogram distance (0..1)
    #[arg(long, value_parser = parse_nonnegative_f64, default_value_t = DEFAULT_HIST_WEIGHT)]
    pub hist_weight: f64,

    /// Motion-estimation search radius (pixels in the working luma)
    /// (downscaled by default, or native when --native-res is set)
    #[arg(long, value_parser = parse_nonnegative_i32, default_value_t = DEFAULT_ME_SEARCH_RADIUS)]
    pub me_search_radius: i32,

    /// XviD-like intra threshold (block SAD is "too big" vs. block complexity)
    #[arg(long, value_parser = parse_nonnegative_i32, default_value_t = DEFAULT_ME_INTRA_THRESH)]
    pub me_intra_thresh: i32,

    /// XviD-like intra score threshold (higher => fewer I-frames)
    #[arg(long, value_parser = parse_nonnegative_f64, default_value_t = DEFAULT_ME_INTRA_THRESH2)]
    pub me_intra_thresh2: f64,

    /// Require hardware decoding (no software fallback), e.g. vaapi/qsv/cuda.
    /// Frames are still transferred back to CPU memory for analysis, so end-to-end speedups vary.
    #[arg(long, value_name = "HWDEV")]
    pub hwdec: Option<String>,
}
