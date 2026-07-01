use crate::{SCuiseiError, SCuiseiResult};
use crate::{decoder, detector, postprocess, simd_metrics};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::mpsc;

const ADAPTIVE_PROMOTION_MIN_RATIO: f64 = 0.85;
/// Decode-thread → detection-thread pipeline depth (frames in flight).
const PIPELINE_DEPTH: usize = 4;

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
    /// Show a progress bar on stderr while analyzing.
    pub progress: bool,
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

/// Detection-thread caches for downscale and grid-histogram plans.
struct PlanCache {
    downscale: Option<((usize, usize), simd_metrics::DownscalePlan)>,
    grid: Option<simd_metrics::GridHistogramPlan>,
}

impl PlanCache {
    fn new() -> Self {
        Self {
            downscale: None,
            grid: None,
        }
    }

    fn downscale_plan(&mut self, width: usize, height: usize) -> &simd_metrics::DownscalePlan {
        if self
            .downscale
            .as_ref()
            .is_none_or(|(dims, _)| *dims != (width, height))
        {
            let (work_w, work_h) = compute_downscale_dims(width, height);
            self.downscale = Some((
                (width, height),
                simd_metrics::DownscalePlan::new(width, height, work_w, work_h),
            ));
        }

        &self
            .downscale
            .as_ref()
            .expect("downscale plan must exist")
            .1
    }

    fn grid_plan(&mut self, width: usize, height: usize) -> &simd_metrics::GridHistogramPlan {
        if self
            .grid
            .as_ref()
            .is_none_or(|plan| plan.dimensions() != (width, height))
        {
            self.grid = Some(simd_metrics::GridHistogramPlan::new(width, height));
        }

        self.grid.as_ref().expect("grid histogram plan must exist")
    }
}

/// Copy or downscale the frame's luma into an owned analysis-resolution
/// buffer, returning the buffer's dimensions.
fn prepare_analysis_pixels(
    source: decoder::LumaSource<'_>,
    info: decoder::FrameInfo,
    native_res: bool,
    plans: &mut PlanCache,
    pixels: &mut Vec<u8>,
) -> (usize, usize) {
    match source {
        decoder::LumaSource::Packed8(view) => {
            if native_res {
                simd_metrics::extract_packed8(
                    view.data,
                    view.stride,
                    info.width,
                    info.height,
                    pixels,
                );
                return (info.width, info.height);
            }

            let plan = plans.downscale_plan(info.width, info.height);
            plan.run_packed8(view.data, view.stride, pixels);
            plan.dst_dimensions()
        }
        decoder::LumaSource::Planar16(view) => {
            if native_res {
                simd_metrics::extract_planar16_packed8(
                    view.data,
                    view.stride,
                    info.width,
                    info.height,
                    view.params,
                    pixels,
                );
                return (info.width, info.height);
            }

            let plan = plans.downscale_plan(info.width, info.height);
            plan.run_planar16_packed8(view.data, view.stride, view.params, pixels);
            plan.dst_dimensions()
        }
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

    /// Swap the current frame in as the new previous frame.
    fn absorb(&mut self, curr_small: &mut Vec<u8>, snapshot: &CurrentFrameSnapshot) {
        self.prev_dims = Some((snapshot.width, snapshot.height));
        self.prev_hist = snapshot.hist;
        self.prev_grid_hist = snapshot.grid_hist;
        self.frame_index = self.frame_index.saturating_add(1);
        std::mem::swap(&mut self.prev_small, curr_small);
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

fn analyze_frame(
    state: &DetectionState,
    xvid_detector: &mut detector::XvidDetector,
    adaptive_detector: &mut detector::Detector,
    pixels: &[u8],
    snapshot: &CurrentFrameSnapshot,
    sad: u64,
) -> DetectionRecord {
    let dims_changed = state.prev_dims != Some((snapshot.width, snapshot.height))
        || state.prev_small.len() != pixels.len();
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

    let xvid_decision =
        xvid_detector.decide(&state.prev_small, pixels, snapshot.width, snapshot.height);
    let hist_distance = simd_metrics::histogram_distance_16_from_hists(
        &state.prev_hist,
        &snapshot.hist,
        pixels.len(),
    );
    let (grid_hist_distance, grid_hist_median) =
        simd_metrics::grid_histogram_distances_16_from_hists(
            &state.prev_grid_hist,
            &snapshot.grid_hist,
            pixels.len(),
        );

    let sad_score = simd_metrics::normalize_sad(sad, snapshot.width, snapshot.height);
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

/// Best-effort total frame count for progress reporting (header probe only).
fn probe_frame_count(path: &std::path::Path) -> Option<u64> {
    let ictx = ffmpeg_next::format::input(path).ok()?;
    let stream = ictx.streams().best(ffmpeg_next::media::Type::Video)?;
    let frames = stream.frames();
    if frames > 0 {
        return u64::try_from(frames).ok();
    }
    let rate = stream.avg_frame_rate();
    // Stream duration first; fall back to the container duration (AV_TIME_BASE
    // units), which is all some containers (e.g. Matroska) report.
    estimate_frames(stream.duration(), stream.time_base(), rate)
        .or_else(|| estimate_frames(ictx.duration(), ffmpeg_next::rescale::TIME_BASE, rate))
}

/// `duration * time_base * rate` in integer math, truncated; `None` when any
/// input is unset (ffmpeg reports unset durations as negative sentinels).
fn estimate_frames(
    duration: i64,
    time_base: ffmpeg_next::Rational,
    rate: ffmpeg_next::Rational,
) -> Option<u64> {
    let num =
        i128::from(duration) * i128::from(time_base.numerator()) * i128::from(rate.numerator());
    let den = i128::from(time_base.denominator()) * i128::from(rate.denominator());
    if den <= 0 {
        return None;
    }
    u64::try_from(num / den).ok().filter(|&frames| frames > 0)
}

fn make_progress_bar(options: &AnalyzeOptions) -> indicatif::ProgressBar {
    // dump_scores streams CSV to stderr; a redrawing bar would clobber it.
    if !options.progress || options.dump_scores {
        return indicatif::ProgressBar::hidden();
    }
    if let Some(total) = probe_frame_count(&options.input) {
        let bar = indicatif::ProgressBar::new(total);
        bar.set_style(
            indicatif::ProgressStyle::with_template(
                "{bar:40} {pos}/{len} frames ({per_sec}, ETA {eta})",
            )
            .expect("progress template must be valid"),
        );
        bar
    } else {
        let bar = indicatif::ProgressBar::new_spinner();
        bar.set_style(
            indicatif::ProgressStyle::with_template("{spinner} {pos} frames ({per_sec})")
                .expect("progress template must be valid"),
        );
        bar
    }
}

/// Per-frame detection driver shared by the pipelined and inline paths.
struct DetectionRun<'a> {
    options: &'a AnalyzeOptions,
    target: AnalysisTarget,
    xvid_detector: detector::XvidDetector,
    adaptive_detector: detector::Detector,
    state: DetectionState,
    extractor: decoder::LumaExtractor,
    plans: PlanCache,
    curr_small: Vec<u8>,
    frame_stats: Vec<postprocess::FrameCutStats>,
    pass_decisions: Vec<bool>,
    progress: indicatif::ProgressBar,
}

impl<'a> DetectionRun<'a> {
    fn new(options: &'a AnalyzeOptions, target: AnalysisTarget) -> Self {
        Self {
            options,
            target,
            xvid_detector: detector::XvidDetector::new(options.xvid_config),
            adaptive_detector: detector::Detector::new(options.adaptive_config),
            state: DetectionState::new(),
            extractor: decoder::LumaExtractor::new(),
            plans: PlanCache::new(),
            curr_small: Vec::new(),
            frame_stats: Vec::with_capacity(1024),
            pass_decisions: Vec::with_capacity(1024),
            progress: make_progress_bar(options),
        }
    }

    fn process_frame(&mut self, frame: &ffmpeg_next::frame::Video) -> SCuiseiResult<()> {
        self.progress.inc(1);
        let info = decoder::FrameInfo {
            width: frame.width() as usize,
            height: frame.height() as usize,
        };
        let source = self
            .extractor
            .luma_source(frame, !self.options.native_res)?;
        let (width, height) = prepare_analysis_pixels(
            source,
            info,
            self.options.native_res,
            &mut self.plans,
            &mut self.curr_small,
        );

        let frame_metrics = self
            .plans
            .grid_plan(width, height)
            .accumulate_frame_metrics(
                (!self.state.is_first_frame()).then_some(self.state.prev_small.as_slice()),
                &self.curr_small,
            );
        let snapshot = CurrentFrameSnapshot {
            width,
            height,
            hist: frame_metrics.hist,
            grid_hist: frame_metrics.grid_hist,
        };

        if self.state.is_first_frame() {
            if self.target.needs_pass_decisions() {
                self.pass_decisions.push(true);
            }
            self.state.absorb(&mut self.curr_small, &snapshot);
            return Ok(());
        }

        let record = analyze_frame(
            &self.state,
            &mut self.xvid_detector,
            &mut self.adaptive_detector,
            &self.curr_small,
            &snapshot,
            frame_metrics.sad,
        );
        if self.options.dump_scores {
            dump_score_line(self.state.frame_index, record);
        }
        if self.target.needs_pass_decisions() {
            self.pass_decisions.push(record.is_cut);
        }
        if self.target.needs_keyframes() {
            self.frame_stats.push(postprocess::FrameCutStats {
                frame_index: self.state.frame_index,
                is_cut: record.is_cut,
                score: record.score,
                hist_distance: record.hist_distance,
                grid_hist_distance: record.grid_hist_distance,
                grid_hist_median: record.grid_hist_median,
            });
        }

        self.state.absorb(&mut self.curr_small, &snapshot);
        Ok(())
    }

    fn finish(mut self) -> AnalysisResult {
        let keyframes = if self.target.needs_keyframes() {
            postprocess::refine_frame_keyframes_with_config(
                &self.frame_stats,
                &self.options.postprocess_config,
            )
        } else {
            Vec::new()
        };
        AnalysisResult {
            keyframes,
            pass_decisions: std::mem::take(&mut self.pass_decisions),
        }
    }
}

impl Drop for DetectionRun<'_> {
    /// Clears the progress bar on both success and mid-stream errors.
    fn drop(&mut self) {
        self.progress.finish_and_clear();
    }
}

fn analyze_video_impl(
    options: &AnalyzeOptions,
    target: AnalysisTarget,
) -> SCuiseiResult<AnalysisResult> {
    ensure_ffmpeg_initialized()?;

    let mut run = DetectionRun::new(options, target);

    if options.native_res {
        // Native-res detection is internally parallel (rayon) and would fight
        // the decoder's thread pool; run it inline on the decode thread.
        let mut decoder = decoder::Decoder::open(&options.input, options.hwdec.as_deref())?;
        decoder.decode_frames(|frame| {
            run.process_frame(&frame)?;
            Ok(frame)
        })?;
        return Ok(run.finish());
    }

    let (frame_tx, frame_rx) = mpsc::sync_channel::<ffmpeg_next::frame::Video>(PIPELINE_DEPTH);
    let (recycle_tx, recycle_rx) = mpsc::channel::<ffmpeg_next::frame::Video>();

    std::thread::scope(|scope| -> SCuiseiResult<()> {
        let producer = scope.spawn(move || -> SCuiseiResult<()> {
            let mut decoder = decoder::Decoder::open(&options.input, options.hwdec.as_deref())?;
            decoder.decode_frames(|frame| {
                frame_tx
                    .send(frame)
                    .map_err(|_| SCuiseiError::decode("analysis stage stopped unexpectedly"))?;
                Ok(recycle_rx
                    .try_recv()
                    .unwrap_or_else(|_| ffmpeg_next::frame::Video::empty()))
            })
        });

        for frame in frame_rx {
            run.process_frame(&frame)?;
            let _ = recycle_tx.send(frame);
        }

        producer.join().expect("decode thread panicked")
    })?;

    Ok(run.finish())
}

#[cfg(test)]
mod tests {
    use super::{ensure_ffmpeg_initialized, estimate_frames};

    #[test]
    fn ffmpeg_initialization_is_idempotent() {
        assert!(ensure_ffmpeg_initialized().is_ok());
        assert!(ensure_ffmpeg_initialized().is_ok());
    }

    #[test]
    fn estimate_frames_handles_ntsc_and_unset_inputs() {
        let tb = ffmpeg_next::Rational(1, 1000);
        let ntsc = ffmpeg_next::Rational(30000, 1001);
        // 10s at 29.97 fps = 299.7, truncated.
        assert_eq!(estimate_frames(10_000, tb, ntsc), Some(299));
        // Unset duration (negative sentinel), zero rate, unset time base.
        assert_eq!(estimate_frames(i64::MIN, tb, ntsc), None);
        assert_eq!(
            estimate_frames(10_000, tb, ffmpeg_next::Rational(0, 1)),
            None
        );
        assert_eq!(
            estimate_frames(10_000, ffmpeg_next::Rational(0, 0), ntsc),
            None
        );
    }
}
