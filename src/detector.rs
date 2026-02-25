use std::collections::VecDeque;

const MIN_SAMPLES_FOR_ADAPTIVE_THRESHOLD: usize = 5;
const TWO_POW_32_F64: f64 = 4_294_967_296.0;

#[derive(Clone, Copy, Debug)]
pub struct XvidDetectorConfig {
    pub search_radius: i32,
    pub intra_thresh: i32,
    pub intra_thresh2: f64,
}

impl Default for XvidDetectorConfig {
    fn default() -> Self {
        Self {
            search_radius: crate::cli::DEFAULT_ME_SEARCH_RADIUS,
            intra_thresh: crate::cli::DEFAULT_ME_INTRA_THRESH,
            intra_thresh2: crate::cli::DEFAULT_ME_INTRA_THRESH2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct XvidDecision {
    pub is_cut: bool,
    pub score: f64,
    pub threshold: f64,
    pub intra_blocks: usize,
    pub total_blocks: usize,
    pub mc_sad: u64,
}

pub struct XvidDetector {
    config: XvidDetectorConfig,
    intra_count: usize,
}

impl XvidDetector {
    #[must_use]
    pub fn new(config: XvidDetectorConfig) -> Self {
        Self {
            config,
            intra_count: 1,
        }
    }

    pub fn reset(&mut self) {
        self.intra_count = 1;
    }

    #[must_use]
    pub fn decide(
        &mut self,
        prev: &[u8],
        curr: &[u8],
        width: usize,
        height: usize,
    ) -> XvidDecision {
        let mut intra_thresh = self.config.intra_thresh;
        let mut intra_thresh2 = self.config.intra_thresh2;

        if self.intra_count < 30 {
            if self.intra_count < 10 {
                let d =
                    i32::try_from(10_usize.saturating_sub(self.intra_count)).unwrap_or(i32::MAX);
                intra_thresh =
                    intra_thresh.saturating_add(15_i32.saturating_mul(d.saturating_mul(d)));
            }
            let remaining = u32::try_from(30_usize.saturating_sub(self.intra_count)).unwrap_or(0);
            intra_thresh2 += 4.0 * f64::from(remaining);
        }

        let analysis = crate::simd_metrics::meanalysis_xvid_like(
            prev,
            curr,
            width,
            height,
            self.config.search_radius,
            intra_thresh,
        );

        let mut is_cut = false;
        if analysis.total_blocks > 0
            && analysis.intra_blocks.saturating_mul(2) > analysis.total_blocks
        {
            is_cut = true;
        }
        if analysis.score > intra_thresh2 {
            is_cut = true;
        }

        self.intra_count = if is_cut {
            1
        } else {
            self.intra_count.saturating_add(1)
        };

        XvidDecision {
            is_cut,
            score: analysis.score,
            threshold: intra_thresh2,
            intra_blocks: analysis.intra_blocks,
            total_blocks: analysis.total_blocks,
            mc_sad: analysis.mc_sad,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DetectorConfig {
    pub window_size: usize,
    pub sigma: f64,
    pub base_threshold: f64,
    pub min_hist_distance: f64,
    pub min_score: f64,
    pub sad_weight: f64,
    pub hist_weight: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            window_size: crate::cli::DEFAULT_WINDOW_SIZE,
            sigma: crate::cli::DEFAULT_SIGMA,
            base_threshold: crate::cli::DEFAULT_BASE_THRESHOLD,
            min_hist_distance: crate::cli::DEFAULT_MIN_HIST_DISTANCE,
            min_score: crate::cli::DEFAULT_MIN_SCORE,
            sad_weight: crate::cli::DEFAULT_SAD_WEIGHT,
            hist_weight: crate::cli::DEFAULT_HIST_WEIGHT,
        }
    }
}

pub struct Detector {
    config: DetectorConfig,
    window: VecDeque<f64>,
}

impl Detector {
    #[must_use]
    pub fn new(config: DetectorConfig) -> Self {
        let window = VecDeque::with_capacity(config.window_size);
        Self { config, window }
    }

    pub fn reset(&mut self) {
        self.window.clear();
    }

    #[must_use]
    pub fn threshold(&self) -> f64 {
        if self.window.len() < MIN_SAMPLES_FOR_ADAPTIVE_THRESHOLD {
            return self.config.base_threshold;
        }

        let mut values: Vec<f64> = self.window.iter().copied().collect();
        values.sort_by(f64::total_cmp);

        let len = values.len();
        let trim = (len / 10).clamp(1, len / 4);
        let start = trim;
        let end = len.saturating_sub(trim);
        let slice = &values[start..end];
        if slice.len() < 2 {
            return self.config.base_threshold;
        }

        let n = usize_to_f64(slice.len());
        let mean = slice.iter().sum::<f64>() / n;
        let mut var = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        if var.is_sign_negative() {
            var = 0.0;
        }
        let stddev = var.sqrt();

        let threshold = mean + (self.config.sigma * stddev);
        threshold.max(self.config.base_threshold)
    }

    #[must_use]
    pub fn blended_score(&self, sad_score: f64, hist_distance: f64) -> f64 {
        let sad_component = sad_score.clamp(0.0, 1.0);
        let hist_component = hist_distance.clamp(0.0, 1.0);
        let sad_weight = self.config.sad_weight.max(0.0);
        let hist_weight = self.config.hist_weight.max(0.0);
        let total_weight = sad_weight + hist_weight;
        if total_weight <= f64::EPSILON {
            return sad_component.max(hist_component);
        }
        ((sad_weight * sad_component) + (hist_weight * hist_component)) / total_weight
    }

    #[must_use]
    pub fn decide_with_threshold(&self, score: f64, hist_distance: f64) -> (bool, f64) {
        let threshold = self.threshold();
        if score < self.config.min_score {
            return (false, threshold);
        }
        if hist_distance < self.config.min_hist_distance {
            return (false, threshold);
        }
        (score >= threshold, threshold)
    }

    pub fn observe(&mut self, score: f64) {
        if self.config.window_size == 0 {
            return;
        }

        if self.window.len() == self.config.window_size {
            self.window.pop_front();
        }

        self.window.push_back(score);
    }
}

fn usize_to_f64(value: usize) -> f64 {
    let value_u64 = u64::try_from(value).unwrap_or(u64::MAX);
    u64_to_f64(value_u64)
}

fn u64_to_f64(value: u64) -> f64 {
    let high = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low_mask = u64::from(u32::MAX);
    let low = u32::try_from(value & low_mask).unwrap_or(u32::MAX);
    (f64::from(high) * TWO_POW_32_F64) + f64::from(low)
}
