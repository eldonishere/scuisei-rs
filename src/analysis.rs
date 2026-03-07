use crate::{SCuiseiError, SCuiseiResult};
use crate::{decoder, detector, postprocess, simd_metrics};
use std::path::PathBuf;

const ADAPTIVE_PROMOTION_MIN_RATIO: f64 = 0.85;

#[derive(Clone, Debug)]
/// Parameters that control a full scene-detection run.
pub struct AnalyzeOptions {
    /// Input media path to decode.
    pub input: PathBuf,
    /// Analyze at source resolution instead of downsampled working resolution.
    pub native_res: bool,
    /// Optional hardware decoding device name (`vaapi`, `qsv`, `cuda`, etc.).
    pub hwdec: Option<String>,
    /// Emit per-frame debug scores to stderr.
    pub dump_scores: bool,
    /// Motion-estimation detector settings.
    pub xvid_config: detector::XvidDetectorConfig,
    /// Adaptive detector settings.
    pub adaptive_config: detector::DetectorConfig,
    /// Keyframe postprocess tuning.
    pub postprocess_config: postprocess::PostprocessConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Final outputs produced by a scene-detection run.
pub struct AnalysisResult {
    /// Refined keyframe indices suitable for AGI or `frames` output.
    pub keyframes: Vec<usize>,
    /// Per-frame `SCXvid` decisions where `true` means `i` and `false` means `p`.
    pub pass_decisions: Vec<bool>,
}

impl AnalysisResult {
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.pass_decisions.len()
    }
}

#[derive(Clone, Copy, Debug)]
struct CurrentFrameSnapshot {
    width: usize,
    height: usize,
    hist: [u32; 16],
    grid_hist: [u32; simd_metrics::GRID_HIST_LEN],
}

struct CurrentFrame<'a> {
    pixels: &'a [u8],
    snapshot: CurrentFrameSnapshot,
}

#[derive(Clone, Copy, Debug)]
struct DetectionRecord {
    is_cut: bool,
    score: f64,
    threshold: f64,
    intra_blocks: usize,
    total_blocks: usize,
    mc_sad: u64,
    hist_distance: f64,
    grid_hist_distance: f64,
    grid_hist_median: f64,
}

#[derive(Debug)]
struct DetectionState {
    prev_small: Vec<u8>,
    prev_hist: [u32; 16],
    prev_grid_hist: [u32; simd_metrics::GRID_HIST_LEN],
    prev_dims: Option<(usize, usize)>,
    frame_index: usize,
}

impl DetectionState {
    fn new() -> Self {
        Self {
            prev_small: Vec::new(),
            prev_hist: [0; 16],
            prev_grid_hist: [0; simd_metrics::GRID_HIST_LEN],
            prev_dims: None,
            frame_index: 0,
        }
    }

    fn is_first_frame(&self) -> bool {
        self.frame_index == 0
    }

    fn absorb(
        &mut self,
        curr_luma: &mut Vec<u8>,
        curr_small: &mut Vec<u8>,
        native_res: bool,
        snapshot: &CurrentFrameSnapshot,
    ) {
        self.prev_dims = Some((snapshot.width, snapshot.height));
        self.prev_hist = snapshot.hist;
        self.prev_grid_hist = snapshot.grid_hist;
        if native_res {
            std::mem::swap(&mut self.prev_small, curr_luma);
        } else {
            std::mem::swap(&mut self.prev_small, curr_small);
        }
        self.frame_index = self.frame_index.saturating_add(1);
    }
}

fn compute_downscale_dims(width: usize, height: usize) -> (usize, usize) {
    const MAX_WIDTH: usize = 160;
    const MAX_HEIGHT: usize = 96;

    if width == 0 || height == 0 {
        return (0, 0);
    }

    let mut out_w = width.min(MAX_WIDTH);
    let mut out_h = height.saturating_mul(out_w) / width;
    if out_h == 0 {
        out_h = 1;
    }

    if out_h > MAX_HEIGHT {
        out_h = height.min(MAX_HEIGHT);
        out_w = width.saturating_mul(out_h) / height;
        if out_w == 0 {
            out_w = 1;
        }
    }

    (out_w, out_h)
}

fn prepare_working_luma(
    curr_luma: &[u8],
    width: usize,
    height: usize,
    working_luma: &mut Vec<u8>,
) -> (usize, usize) {
    let (work_w, work_h) = compute_downscale_dims(width, height);
    simd_metrics::downscale_gray8_nearest(curr_luma, width, height, work_w, work_h, working_luma);
    (work_w, work_h)
}

fn prepare_current_frame<'a>(
    curr_luma: &'a [u8],
    info: decoder::FrameInfo,
    native_res: bool,
    curr_small: &'a mut Vec<u8>,
) -> CurrentFrame<'a> {
    let (width, height, pixels): (usize, usize, &[u8]) = if native_res {
        (info.width, info.height, curr_luma)
    } else {
        let (work_w, work_h) = prepare_working_luma(curr_luma, info.width, info.height, curr_small);
        (work_w, work_h, curr_small.as_slice())
    };

    let snapshot = CurrentFrameSnapshot {
        width,
        height,
        hist: simd_metrics::histogram_16(pixels),
        grid_hist: simd_metrics::grid_histogram_16(pixels, width, height),
    };
    CurrentFrame { pixels, snapshot }
}

fn analyze_frame(
    state: &DetectionState,
    xvid_detector: &mut detector::XvidDetector,
    adaptive_detector: &mut detector::Detector,
    current: &CurrentFrame<'_>,
) -> DetectionRecord {
    let dims_changed = state.prev_dims != Some((current.snapshot.width, current.snapshot.height))
        || state.prev_small.len() != current.pixels.len();
    if dims_changed || state.prev_small.is_empty() {
        xvid_detector.reset();
        adaptive_detector.reset();
        return DetectionRecord {
            is_cut: true,
            score: f64::INFINITY,
            threshold: f64::INFINITY,
            intra_blocks: 0,
            total_blocks: 0,
            mc_sad: 0,
            hist_distance: 1.0,
            grid_hist_distance: 1.0,
            grid_hist_median: 1.0,
        };
    }

    let xvid_decision = xvid_detector.decide(
        &state.prev_small,
        current.pixels,
        current.snapshot.width,
        current.snapshot.height,
    );
    let hist_distance = simd_metrics::histogram_distance_16_from_hists(
        &state.prev_hist,
        &current.snapshot.hist,
        current.pixels.len(),
    );
    let grid_hist_distance = simd_metrics::grid_histogram_distance_16_from_hists(
        &state.prev_grid_hist,
        &current.snapshot.grid_hist,
        current.pixels.len(),
    );
    let grid_hist_median = simd_metrics::grid_histogram_median_cell_distance_16_from_hists(
        &state.prev_grid_hist,
        &current.snapshot.grid_hist,
    );

    let sad = simd_metrics::sad_u8(&state.prev_small, current.pixels);
    let sad_score =
        simd_metrics::normalize_sad(sad, current.snapshot.width, current.snapshot.height);
    let adaptive_score = adaptive_detector.blended_score(sad_score, hist_distance);
    let (adaptive_cut, _) = adaptive_detector.decide_with_threshold(adaptive_score, hist_distance);
    adaptive_detector.observe(adaptive_score);
    let near_xvid_cut = xvid_decision.threshold.is_finite()
        && xvid_decision.threshold > 0.0
        && xvid_decision.score >= (xvid_decision.threshold * ADAPTIVE_PROMOTION_MIN_RATIO);

    DetectionRecord {
        is_cut: xvid_decision.is_cut || (near_xvid_cut && adaptive_cut),
        score: xvid_decision.score,
        threshold: xvid_decision.threshold,
        intra_blocks: xvid_decision.intra_blocks,
        total_blocks: xvid_decision.total_blocks,
        mc_sad: xvid_decision.mc_sad,
        hist_distance,
        grid_hist_distance,
        grid_hist_median,
    }
}

fn dump_score_line(frame_index: usize, record: DetectionRecord) {
    eprintln!(
        "{frame_index},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{}",
        record.score,
        record.threshold,
        record.hist_distance,
        record.grid_hist_distance,
        record.grid_hist_median,
        record.intra_blocks,
        record.total_blocks,
        record.mc_sad,
        i32::from(record.is_cut)
    );
}

/// Analyze a video and return both pass-frame decisions and refined keyframe indices.
///
/// # Errors
/// Returns an error if `FFmpeg` initialization fails, the input cannot be decoded,
/// or frame processing encounters an I/O/codec error.
pub fn analyze_video(options: &AnalyzeOptions) -> SCuiseiResult<AnalysisResult> {
    analyze_video_impl(options)
}

fn analyze_video_impl(options: &AnalyzeOptions) -> SCuiseiResult<AnalysisResult> {
    ffmpeg_next::init()
        .map_err(|error| SCuiseiError::decode_with("failed to initialize ffmpeg", &error))?;

    let mut xvid_detector = detector::XvidDetector::new(options.xvid_config);
    let mut adaptive_detector = detector::Detector::new(options.adaptive_config);
    let mut state = DetectionState::new();
    let mut curr_small: Vec<u8> = Vec::new();
    let mut frame_stats: Vec<postprocess::FrameCutStats> = Vec::new();
    let mut pass_decisions: Vec<bool> = Vec::new();

    decoder::Decoder::open(&options.input, options.hwdec.as_deref())?.decode_luma_frames(
        |curr_luma, info| {
            let current =
                prepare_current_frame(curr_luma, info, options.native_res, &mut curr_small);
            let snapshot = current.snapshot;

            if state.is_first_frame() {
                pass_decisions.push(true);
                state.absorb(curr_luma, &mut curr_small, options.native_res, &snapshot);
                return Ok(());
            }

            let record =
                analyze_frame(&state, &mut xvid_detector, &mut adaptive_detector, &current);
            if options.dump_scores {
                dump_score_line(state.frame_index, record);
            }
            pass_decisions.push(record.is_cut);
            frame_stats.push(postprocess::FrameCutStats {
                frame_index: state.frame_index,
                is_cut: record.is_cut,
                score: record.score,
                hist_distance: record.hist_distance,
                grid_hist_distance: record.grid_hist_distance,
                grid_hist_median: record.grid_hist_median,
            });

            state.absorb(curr_luma, &mut curr_small, options.native_res, &snapshot);
            Ok(())
        },
    )?;

    let keyframes =
        postprocess::refine_frame_keyframes_with_config(&frame_stats, &options.postprocess_config);
    Ok(AnalysisResult {
        keyframes,
        pass_decisions,
    })
}
