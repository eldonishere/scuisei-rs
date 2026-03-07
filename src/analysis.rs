use crate::{SCuiseiError, SCuiseiResult};
use crate::{decoder, detector, postprocess, simd_metrics};
use std::path::PathBuf;
use std::sync::OnceLock;

const ADAPTIVE_PROMOTION_MIN_RATIO: f64 = 0.85;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AnalysisTarget {
    Keyframes,
    PassDecisions,
    Both,
}

impl AnalysisTarget {
    fn needs_keyframes(self) -> bool {
        matches!(self, Self::Keyframes | Self::Both)
    }

    fn needs_pass_decisions(self) -> bool {
        matches!(self, Self::PassDecisions | Self::Both)
    }
}

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
    width: usize,
    height: usize,
    grid_hist_plan: &'a simd_metrics::GridHistogramPlan,
}

#[derive(Debug)]
struct WorkingFramePlan {
    source_dims: (usize, usize),
    downscale_plan: simd_metrics::DownscalePlan,
    grid_hist_plan: simd_metrics::GridHistogramPlan,
}

#[derive(Debug)]
struct FramePreparationCache {
    native_grid_plan: Option<simd_metrics::GridHistogramPlan>,
    working_plan: Option<WorkingFramePlan>,
}

impl FramePreparationCache {
    fn new() -> Self {
        Self {
            native_grid_plan: None,
            working_plan: None,
        }
    }

    fn native_grid_plan(
        &mut self,
        width: usize,
        height: usize,
    ) -> &simd_metrics::GridHistogramPlan {
        if self
            .native_grid_plan
            .as_ref()
            .is_none_or(|plan| plan.dimensions() != (width, height))
        {
            self.native_grid_plan = Some(simd_metrics::GridHistogramPlan::new(width, height));
        }

        self.native_grid_plan
            .as_ref()
            .expect("native grid plan must exist")
    }

    fn working_plan(&mut self, width: usize, height: usize) -> &WorkingFramePlan {
        if self
            .working_plan
            .as_ref()
            .is_none_or(|plan| plan.source_dims != (width, height))
        {
            let (work_w, work_h) = compute_downscale_dims(width, height);
            let downscale_plan = simd_metrics::DownscalePlan::new(width, height, work_w, work_h);
            let grid_hist_plan = simd_metrics::GridHistogramPlan::new(work_w, work_h);
            self.working_plan = Some(WorkingFramePlan {
                source_dims: (width, height),
                downscale_plan,
                grid_hist_plan,
            });
        }

        self.working_plan
            .as_ref()
            .expect("working frame plan must exist")
    }
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
        curr_luma: &[u8],
        curr_small: &mut Vec<u8>,
        native_res: bool,
        snapshot: &CurrentFrameSnapshot,
    ) {
        self.prev_dims = Some((snapshot.width, snapshot.height));
        self.prev_hist = snapshot.hist;
        self.prev_grid_hist = snapshot.grid_hist;
        if native_res {
            self.prev_small.resize(curr_luma.len(), 0);
            self.prev_small.copy_from_slice(curr_luma);
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
    cache: &mut FramePreparationCache,
) -> (usize, usize) {
    let working_plan = cache.working_plan(width, height);
    working_plan.downscale_plan.run(curr_luma, working_luma);
    working_plan.downscale_plan.dst_dimensions()
}

fn prepare_current_frame<'a>(
    curr_luma: &'a [u8],
    info: decoder::FrameInfo,
    native_res: bool,
    curr_small: &'a mut Vec<u8>,
    cache: &'a mut FramePreparationCache,
) -> CurrentFrame<'a> {
    let (width, height, pixels, grid_hist_plan) = if native_res {
        let grid_hist_plan = cache.native_grid_plan(info.width, info.height);
        (info.width, info.height, curr_luma, grid_hist_plan)
    } else {
        let (work_w, work_h) =
            prepare_working_luma(curr_luma, info.width, info.height, curr_small, cache);
        let grid_hist_plan = &cache.working_plan(info.width, info.height).grid_hist_plan;
        (work_w, work_h, curr_small.as_slice(), grid_hist_plan)
    };

    CurrentFrame {
        pixels,
        width,
        height,
        grid_hist_plan,
    }
}

fn analyze_frame(
    state: &DetectionState,
    xvid_detector: &mut detector::XvidDetector,
    adaptive_detector: &mut detector::Detector,
    current: &CurrentFrame<'_>,
    snapshot: &CurrentFrameSnapshot,
    sad: u64,
) -> DetectionRecord {
    let dims_changed = state.prev_dims != Some((current.width, current.height))
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
        current.width,
        current.height,
    );
    let hist_distance = simd_metrics::histogram_distance_16_from_hists(
        &state.prev_hist,
        &snapshot.hist,
        current.pixels.len(),
    );
    let grid_hist_distance = simd_metrics::grid_histogram_distance_16_from_hists(
        &state.prev_grid_hist,
        &snapshot.grid_hist,
        current.pixels.len(),
    );
    let grid_hist_median = simd_metrics::grid_histogram_median_cell_distance_16_from_hists(
        &state.prev_grid_hist,
        &snapshot.grid_hist,
    );

    let sad_score = simd_metrics::normalize_sad(sad, current.width, current.height);
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

fn ensure_ffmpeg_initialized() -> SCuiseiResult<()> {
    static FFMPEG_INIT: OnceLock<Result<(), String>> = OnceLock::new();

    match FFMPEG_INIT.get_or_init(|| {
        ffmpeg_next::init().map_err(|error| format!("failed to initialize ffmpeg: {error}"))
    }) {
        Ok(()) => Ok(()),
        Err(message) => Err(SCuiseiError::decode(message.clone())),
    }
}

/// Analyze a video and return both pass-frame decisions and refined keyframe indices.
///
/// # Errors
/// Returns an error if `FFmpeg` initialization fails, the input cannot be decoded,
/// or frame processing encounters an I/O/codec error.
pub fn analyze_video(options: &AnalyzeOptions) -> SCuiseiResult<AnalysisResult> {
    options.validate()?;
    analyze_video_impl(options, AnalysisTarget::Both)
}

/// Analyze a video and return only refined keyframe indices.
///
/// # Errors
/// Returns an error if analysis fails.
pub fn analyze_keyframes(options: &AnalyzeOptions) -> SCuiseiResult<Vec<usize>> {
    options.validate()?;
    analyze_video_impl(options, AnalysisTarget::Keyframes).map(|result| result.keyframes)
}

/// Analyze a video and return only SCXvid-style pass decisions.
///
/// # Errors
/// Returns an error if analysis fails.
pub fn analyze_pass_decisions(options: &AnalyzeOptions) -> SCuiseiResult<Vec<bool>> {
    options.xvid_config.validate()?;
    options.adaptive_config.validate()?;
    analyze_video_impl(options, AnalysisTarget::PassDecisions).map(|result| result.pass_decisions)
}

fn analyze_video_impl(
    options: &AnalyzeOptions,
    target: AnalysisTarget,
) -> SCuiseiResult<AnalysisResult> {
    ensure_ffmpeg_initialized()?;

    let mut decoder = decoder::Decoder::open(&options.input, options.hwdec.as_deref())?;
    let frame_count_hint = decoder.frame_count_hint();

    let mut xvid_detector = detector::XvidDetector::new(options.xvid_config);
    let mut adaptive_detector = detector::Detector::new(options.adaptive_config);
    let mut state = DetectionState::new();
    let mut curr_small: Vec<u8> = Vec::new();
    let mut frame_preparation_cache = FramePreparationCache::new();
    let mut frame_stats: Vec<postprocess::FrameCutStats> = if target.needs_keyframes() {
        Vec::with_capacity(frame_count_hint.map_or(0, |count| count.saturating_sub(1)))
    } else {
        Vec::with_capacity(0)
    };
    let mut pass_decisions: Vec<bool> = if target.needs_pass_decisions() {
        Vec::with_capacity(frame_count_hint.unwrap_or(0))
    } else {
        Vec::with_capacity(0)
    };

    decoder.decode_luma_frames(|curr_luma, info| {
        let current = prepare_current_frame(
            curr_luma,
            info,
            options.native_res,
            &mut curr_small,
            &mut frame_preparation_cache,
        );
        let frame_metrics = current.grid_hist_plan.accumulate_frame_metrics(
            (!state.is_first_frame()).then_some(state.prev_small.as_slice()),
            current.pixels,
        );
        let snapshot = CurrentFrameSnapshot {
            width: current.width,
            height: current.height,
            hist: frame_metrics.hist,
            grid_hist: frame_metrics.grid_hist,
        };

        if state.is_first_frame() {
            if target.needs_pass_decisions() {
                pass_decisions.push(true);
            }
            state.absorb(curr_luma, &mut curr_small, options.native_res, &snapshot);
            return Ok(());
        }

        let record = analyze_frame(
            &state,
            &mut xvid_detector,
            &mut adaptive_detector,
            &current,
            &snapshot,
            frame_metrics.sad,
        );
        if options.dump_scores {
            dump_score_line(state.frame_index, record);
        }
        if target.needs_pass_decisions() {
            pass_decisions.push(record.is_cut);
        }
        if target.needs_keyframes() {
            frame_stats.push(postprocess::FrameCutStats {
                frame_index: state.frame_index,
                is_cut: record.is_cut,
                score: record.score,
                hist_distance: record.hist_distance,
                grid_hist_distance: record.grid_hist_distance,
                grid_hist_median: record.grid_hist_median,
            });
        }

        state.absorb(curr_luma, &mut curr_small, options.native_res, &snapshot);
        Ok(())
    })?;

    let keyframes = if target.needs_keyframes() {
        postprocess::refine_frame_keyframes_with_config(&frame_stats, &options.postprocess_config)
    } else {
        Vec::new()
    };
    Ok(AnalysisResult {
        keyframes,
        pass_decisions,
    })
}

#[cfg(test)]
mod tests {
    use super::ensure_ffmpeg_initialized;

    #[test]
    fn ffmpeg_initialization_is_idempotent() {
        assert!(ensure_ffmpeg_initialized().is_ok());
        assert!(ensure_ffmpeg_initialized().is_ok());
    }
}
