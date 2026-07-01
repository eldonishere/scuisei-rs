#![deny(clippy::pedantic)]

pub mod analysis;
pub mod cli;
mod decoder;
pub mod detector;
pub mod error;
mod formatter;
mod postprocess;
pub mod simd_metrics;
mod validation;

#[cfg(feature = "python")]
mod python;

use crate::cli::OutputFormat;
pub use analysis::{
    AnalysisResult, AnalyzeOptions, analyze_keyframes, analyze_pass_decisions, analyze_video,
};
pub use detector::{DetectorConfig, XvidDetectorConfig};
pub use error::{SCuiseiError, SCuiseiResult};
pub use postprocess::PostprocessConfig;
use std::io::Write;
use std::path::PathBuf;

fn build_xvid_config(cli: &cli::Cli) -> detector::XvidDetectorConfig {
    detector::XvidDetectorConfig {
        search_radius: cli.me_search_radius,
        intra_thresh: cli.me_intra_thresh,
        intra_thresh2: cli.me_intra_thresh2,
    }
}

fn build_adaptive_config(cli: &cli::Cli) -> detector::DetectorConfig {
    detector::DetectorConfig {
        window_size: cli.window_size,
        sigma: cli.sigma,
        base_threshold: cli.base_threshold,
        min_hist_distance: cli.min_hist_distance,
        min_score: cli.min_score,
        sad_weight: cli.sad_weight,
        hist_weight: cli.hist_weight,
    }
}

impl AnalyzeOptions {
    #[must_use]
    pub fn from_cli(cli: &cli::Cli, dump_scores: bool) -> Self {
        Self {
            input: cli.input.clone(),
            native_res: cli.native_res,
            hwdec: cli.hwdec.clone(),
            dump_scores,
            progress: cli.progress,
            xvid_config: build_xvid_config(cli),
            adaptive_config: build_adaptive_config(cli),
            postprocess_config: postprocess::PostprocessConfig::default(),
        }
    }

    #[must_use]
    pub fn defaults_for_input(input: impl Into<PathBuf>) -> Self {
        Self {
            input: input.into(),
            native_res: false,
            hwdec: None,
            dump_scores: false,
            progress: false,
            xvid_config: detector::XvidDetectorConfig::default(),
            adaptive_config: detector::DetectorConfig::default(),
            postprocess_config: postprocess::PostprocessConfig::default(),
        }
    }

    /// Validate library-supplied configuration before analysis begins.
    ///
    /// # Errors
    /// Returns `SCuiseiError::Config` if any option is out of range.
    pub fn validate(&self) -> SCuiseiResult<()> {
        self.xvid_config.validate()?;
        self.adaptive_config.validate()?;
        self.postprocess_config.validate()?;
        Ok(())
    }
}

/// Write comma-separated keyframe indices and a trailing newline.
///
/// # Errors
/// Returns an error if writing to the output stream fails.
pub fn write_frames_csv<W: Write>(writer: &mut W, keyframes: &[usize]) -> SCuiseiResult<()> {
    let mut writer = std::io::BufWriter::new(writer);
    for (index, frame) in keyframes.iter().enumerate() {
        if index > 0 {
            write!(writer, ",")
                .map_err(|error| SCuiseiError::io("failed to write frame separator", &error))?;
        }
        write!(writer, "{frame}")
            .map_err(|error| SCuiseiError::io("failed to write frame index", &error))?;
    }
    writeln!(writer)
        .map_err(|error| SCuiseiError::io("failed to write trailing newline", &error))?;
    writer
        .flush()
        .map_err(|error| SCuiseiError::io("failed to flush frames output", &error))?;
    Ok(())
}

/// Write AGI keyframe format v1 output with a trailing newline.
///
/// # Errors
/// Returns an error if writing to the output stream fails.
pub fn write_agi<W: Write>(writer: &mut W, keyframes: &[usize]) -> SCuiseiResult<()> {
    let mut writer = std::io::BufWriter::new(writer);
    writeln!(writer, "# keyframe format v1")
        .map_err(|error| SCuiseiError::io("failed to write AGI header", &error))?;
    writeln!(writer, "fps 0")
        .map_err(|error| SCuiseiError::io("failed to write AGI fps line", &error))?;
    writeln!(writer).map_err(|error| SCuiseiError::io("failed to write AGI spacer", &error))?;

    for frame in keyframes {
        writeln!(writer, "{frame} I -1")
            .map_err(|error| SCuiseiError::io("failed to write AGI keyframe line", &error))?;
    }
    writer
        .flush()
        .map_err(|error| SCuiseiError::io("failed to flush AGI output", &error))?;
    Ok(())
}

/// Write SCXvid-compatible frame decisions (`i`/`p`) to a writer.
///
/// # Errors
/// Returns an error if writing to the output stream fails.
pub fn write_pass_log<W: Write>(writer: &mut W, pass_decisions: &[bool]) -> SCuiseiResult<()> {
    let mut writer = std::io::BufWriter::new(writer);
    {
        let mut pass_formatter = formatter::ScxvidPassFormatter::new(&mut writer)
            .map_err(|error| SCuiseiError::io_with("failed to write scxvid header", &error))?;
        for (index, is_cut) in pass_decisions.iter().copied().enumerate() {
            if index == 0 {
                pass_formatter.write_first_frame().map_err(|error| {
                    SCuiseiError::io_with("failed to write first frame", &error)
                })?;
                continue;
            }
            pass_formatter
                .write_frame(is_cut)
                .map_err(|error| SCuiseiError::io_with("failed to write frame decision", &error))?;
        }
    }
    writer
        .flush()
        .map_err(|error| SCuiseiError::io_with("failed to flush scxvid output", &error))?;
    Ok(())
}

/// Run the CLI command using the shared library pipeline.
///
/// # Errors
/// Returns an error if analysis fails or writing formatted output fails.
pub fn run_cli<W: Write>(cli: &cli::Cli, writer: &mut W) -> SCuiseiResult<()> {
    let dump_scores = std::env::var_os("SCUISEI_DUMP_SCORES").is_some();
    let options = AnalyzeOptions::from_cli(cli, dump_scores);

    match cli.format {
        OutputFormat::Agi => {
            let keyframes = analyze_keyframes(&options)?;
            write_agi(writer, &keyframes)?;
        }
        OutputFormat::Xvid => {
            let pass_decisions = analyze_pass_decisions(&options)?;
            write_pass_log(writer, &pass_decisions)?;
        }
        OutputFormat::Frames => {
            let keyframes = analyze_keyframes(&options)?;
            write_frames_csv(writer, &keyframes)?;
        }
    }

    Ok(())
}
