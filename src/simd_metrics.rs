use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::OnceLock;
use wide::u32x8;
#[cfg(not(target_arch = "x86_64"))]
use wide::{u8x16, u16x8};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, _mm_cvtsi128_si64, _mm_loadu_si128, _mm_sad_epu8, _mm_set1_epi8, _mm_setzero_si128,
    _mm_srli_si128,
};

pub const GRID_HIST_X: usize = 4;
pub const GRID_HIST_Y: usize = 4;
pub const GRID_HIST_LEN: usize = 16 * GRID_HIST_X * GRID_HIST_Y;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownscalePlan {
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    x_indices: Vec<usize>,
    y_indices: Vec<usize>,
    exact_decimation_steps: Option<(usize, usize)>,
}

impl DownscalePlan {
    #[must_use]
    pub fn new(src_width: usize, src_height: usize, dst_width: usize, dst_height: usize) -> Self {
        let degenerate = src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0;
        let x_indices = if degenerate {
            Vec::new()
        } else {
            (0..dst_width)
                .map(|x_out| x_out.saturating_mul(src_width) / dst_width)
                .collect()
        };
        let y_indices = if degenerate {
            Vec::new()
        } else {
            (0..dst_height)
                .map(|y_out| y_out.saturating_mul(src_height) / dst_height)
                .collect()
        };
        let exact_decimation_steps = (!degenerate
            && src_width.is_multiple_of(dst_width)
            && src_height.is_multiple_of(dst_height))
        .then_some((src_width / dst_width, src_height / dst_height));

        Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            x_indices,
            y_indices,
            exact_decimation_steps,
        }
    }

    #[must_use]
    pub fn dst_dimensions(&self) -> (usize, usize) {
        (self.dst_width, self.dst_height)
    }

    fn prepare_dst(&self, dst: &mut Vec<u8>) -> bool {
        if self.dst_width == 0 || self.dst_height == 0 {
            dst.clear();
            return false;
        }
        dst.resize(self.dst_width.saturating_mul(self.dst_height), 0);
        true
    }

    pub fn run(&self, src: &[u8], dst: &mut Vec<u8>) {
        self.run_packed8(src, self.src_width, dst);
    }

    /// Sample an 8-bit luma plane with an arbitrary row stride (in bytes).
    pub fn run_packed8(&self, src: &[u8], src_stride: usize, dst: &mut Vec<u8>) {
        if !self.prepare_dst(dst) {
            return;
        }

        if let Some((x_step, y_step)) = self.exact_decimation_steps {
            for y_out in 0..self.dst_height {
                let row_in = y_out.saturating_mul(y_step).saturating_mul(src_stride);
                let row_out = y_out.saturating_mul(self.dst_width);
                for x_out in 0..self.dst_width {
                    dst[row_out + x_out] = src[row_in + x_out.saturating_mul(x_step)];
                }
            }
            return;
        }

        for (y_out, y_in) in self.y_indices.iter().copied().enumerate() {
            let row_in = y_in.saturating_mul(src_stride);
            let row_out = y_out.saturating_mul(self.dst_width);
            for (x_out, x_in) in self.x_indices.iter().copied().enumerate() {
                dst[row_out + x_out] = src[row_in + x_in];
            }
        }
    }

    /// Sample a 16-bit luma plane with an arbitrary row stride into packed 8-bit luma.
    ///
    /// # Panics
    /// Panics if `src` is too short for the sampled plane coordinates.
    pub fn run_planar16_packed8(
        &self,
        src: &[u8],
        src_stride: usize,
        params: Planar16Params,
        dst: &mut Vec<u8>,
    ) {
        if !self.prepare_dst(dst) {
            return;
        }
        let needed = self
            .planar16_required_len(src_stride)
            .expect("validated plane");
        assert!(src.len() >= needed, "validated plane");

        if let Some((x_step, y_step)) = self.exact_decimation_steps {
            for y_out in 0..self.dst_height {
                let row_in = y_out.saturating_mul(y_step).saturating_mul(src_stride);
                let row_out = y_out.saturating_mul(self.dst_width);
                for x_out in 0..self.dst_width {
                    let src_idx = row_in + x_out.saturating_mul(x_step).saturating_mul(2);
                    let sample = read_u16_sample(src, src_idx, params.little_endian);
                    dst[row_out + x_out] = scale_luma16_to_u8(sample, params);
                }
            }
            return;
        }

        for (y_out, y_in) in self.y_indices.iter().copied().enumerate() {
            let row_in = y_in.saturating_mul(src_stride);
            let row_out = y_out.saturating_mul(self.dst_width);
            for (x_out, x_in) in self.x_indices.iter().copied().enumerate() {
                let src_idx = row_in + x_in.saturating_mul(2);
                let sample = read_u16_sample(src, src_idx, params.little_endian);
                dst[row_out + x_out] = scale_luma16_to_u8(sample, params);
            }
        }
    }

    fn planar16_required_len(&self, src_stride: usize) -> Option<usize> {
        let max_y = self.y_indices.last().copied()?;
        let max_x = self.x_indices.last().copied()?;
        max_y
            .checked_mul(src_stride)?
            .checked_add(max_x.checked_mul(2)?)?
            .checked_add(2)
    }
}

/// Copy an 8-bit luma plane with an arbitrary row stride into a contiguous buffer.
pub fn extract_packed8(
    src: &[u8],
    src_stride: usize,
    width: usize,
    height: usize,
    dst: &mut Vec<u8>,
) {
    dst.resize(width.saturating_mul(height), 0);
    if src_stride == width {
        dst.copy_from_slice(&src[..width * height]);
        return;
    }
    for row in 0..height {
        let src_start = row * src_stride;
        let dst_start = row * width;
        dst[dst_start..dst_start + width].copy_from_slice(&src[src_start..src_start + width]);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Planar16Params {
    pub bit_depth: u8,
    pub little_endian: bool,
    pub full_range: bool,
}

/// Copy a 16-bit luma plane into a contiguous packed 8-bit buffer.
///
/// # Panics
/// Panics if `src` is too short for `width`, `height`, and `src_stride`.
pub fn extract_planar16_packed8(
    src: &[u8],
    src_stride: usize,
    width: usize,
    height: usize,
    params: Planar16Params,
    dst: &mut Vec<u8>,
) {
    dst.resize(width.saturating_mul(height), 0);
    if height == 0 || width == 0 {
        return;
    }
    let needed = height
        .saturating_sub(1)
        .checked_mul(src_stride)
        .and_then(|bytes| bytes.checked_add(width.saturating_mul(2)))
        .expect("validated plane");
    assert!(src.len() >= needed, "validated plane");

    for row in 0..height {
        let src_start = row.saturating_mul(src_stride);
        let dst_start = row.saturating_mul(width);
        for col in 0..width {
            let src_idx = src_start + col.saturating_mul(2);
            let sample = read_u16_sample(src, src_idx, params.little_endian);
            dst[dst_start + col] = scale_luma16_to_u8(sample, params);
        }
    }
}

fn read_u16_sample(src: &[u8], index: usize, little_endian: bool) -> u16 {
    debug_assert!(index + 1 < src.len());
    // SAFETY: public planar16 entry points validate the sampled plane length once.
    let sample = unsafe { std::ptr::read_unaligned(src.as_ptr().add(index).cast::<u16>()) };
    if little_endian {
        u16::from_le(sample)
    } else {
        u16::from_be(sample)
    }
}

fn scale_luma16_to_u8(sample: u16, params: Planar16Params) -> u8 {
    if params.full_range {
        return match params.bit_depth {
            9 => scale_full_range::<511>(sample),
            10 => scale_full_range::<1_023>(sample),
            12 => scale_full_range::<4_095>(sample),
            14 => scale_full_range::<16_383>(sample),
            _ => scale_full_range::<65_535>(sample),
        };
    }

    match params.bit_depth {
        9 => scale_limited_range::<32, 470>(sample),
        10 => scale_limited_range::<64, 940>(sample),
        12 => scale_limited_range::<256, 3_760>(sample),
        14 => scale_limited_range::<1_024, 15_040>(sample),
        _ => scale_limited_range::<4_096, 60_160>(sample),
    }
}

fn scale_full_range<const MAX: u32>(sample: u16) -> u8 {
    let sample = u32::from(sample).min(MAX);
    u8::try_from(((sample * 255) + (MAX / 2)) / MAX).unwrap_or(u8::MAX)
}

fn scale_limited_range<const BLACK: u32, const WHITE: u32>(sample: u16) -> u8 {
    let sample = u32::from(sample);
    if sample <= BLACK {
        return 0;
    }
    if sample >= WHITE {
        return u8::MAX;
    }

    u8::try_from(((sample - BLACK) * 255) / (WHITE - BLACK)).unwrap_or(u8::MAX)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridHistogramPlan {
    width: usize,
    row_cell_bases: Vec<usize>,
    x_cell_offsets: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameMetrics {
    pub sad: u64,
    pub hist: [u32; 16],
    pub grid_hist: [u32; GRID_HIST_LEN],
}

impl GridHistogramPlan {
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        let x_cell_offsets = if width == 0 || height == 0 {
            Vec::new()
        } else {
            (0..width)
                .map(|x| ((x * GRID_HIST_X) / width).saturating_mul(16))
                .collect()
        };
        let row_cell_bases = if width == 0 || height == 0 {
            Vec::new()
        } else {
            (0..height)
                .map(|y| ((y * GRID_HIST_Y) / height).saturating_mul(GRID_HIST_X * 16))
                .collect()
        };

        Self {
            width,
            row_cell_bases,
            x_cell_offsets,
        }
    }

    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.row_cell_bases.len())
    }

    #[must_use]
    pub fn run(&self, frame: &[u8]) -> [u32; GRID_HIST_LEN] {
        let mut hist = [0_u32; GRID_HIST_LEN];
        let height = self.row_cell_bases.len();
        if self.width == 0 || height == 0 || frame.len() < self.width.saturating_mul(height) {
            return hist;
        }

        for (y, row_cell_base) in self.row_cell_bases.iter().copied().enumerate() {
            let row = y * self.width;
            for (x, cell_offset) in self.x_cell_offsets.iter().copied().enumerate() {
                let bin = usize::from(frame[row + x] >> 4);
                hist[row_cell_base + cell_offset + bin] += 1;
            }
        }

        hist
    }

    #[must_use]
    pub fn accumulate_frame_metrics(&self, prev: Option<&[u8]>, curr: &[u8]) -> FrameMetrics {
        let mut hist = [0_u32; 16];
        let mut grid_hist = [0_u32; GRID_HIST_LEN];
        let height = self.row_cell_bases.len();
        let needed = self.width.saturating_mul(height);
        if self.width == 0 || height == 0 || curr.len() < needed {
            return FrameMetrics {
                sad: 0,
                hist,
                grid_hist,
            };
        }

        let prev = prev.filter(|frame| frame.len() >= needed);
        let mut sad = 0_u64;

        for (y, row_cell_base) in self.row_cell_bases.iter().copied().enumerate() {
            let row = y * self.width;
            if let Some(prev_frame) = prev {
                sad = sad.saturating_add(sad_u8(
                    &prev_frame[row..row + self.width],
                    &curr[row..row + self.width],
                ));
            }
            for (x, cell_offset) in self.x_cell_offsets.iter().copied().enumerate() {
                let idx = row + x;
                let pixel = curr[idx];
                let bin = usize::from(pixel >> 4);
                hist[bin] += 1;
                grid_hist[row_cell_base + cell_offset + bin] += 1;
            }
        }

        FrameMetrics {
            sad,
            hist,
            grid_hist,
        }
    }
}

#[must_use]
pub fn frame_pair_metrics(
    prev: Option<&[u8]>,
    curr: &[u8],
    grid_plan: &GridHistogramPlan,
) -> FrameMetrics {
    grid_plan.accumulate_frame_metrics(prev, curr)
}

#[must_use]
pub fn sad_u8(prev: &[u8], curr: &[u8]) -> u64 {
    if prev.len() != curr.len() {
        return 0;
    }

    let mut total: u64 = 0;
    let mut i: usize = 0;
    let len = prev.len();

    while i + 16 <= len {
        total += sad16_bytes(&prev[i..], &curr[i..]);
        i += 16;
    }

    for j in i..len {
        total += u64::from(prev[j].abs_diff(curr[j]));
    }

    total
}

#[must_use]
pub fn normalize_sad(sad: u64, width: usize, height: usize) -> f64 {
    let pixels = width.saturating_mul(height);
    if pixels == 0 {
        return 0.0;
    }
    let pixels_u64 = u64::try_from(pixels).unwrap_or(u64::MAX);
    let denom = u64_to_f64_exact(pixels_u64) * 255.0;
    u64_to_f64_exact(sad) / denom
}

#[must_use]
pub fn histogram_distance_16_from_hists(
    prev: &[u32; 16],
    curr: &[u32; 16],
    total_pixels: usize,
) -> f64 {
    if total_pixels == 0 {
        return 0.0;
    }

    let a0 = u32x8::new([
        prev[0], prev[1], prev[2], prev[3], prev[4], prev[5], prev[6], prev[7],
    ]);
    let b0 = u32x8::new([
        curr[0], curr[1], curr[2], curr[3], curr[4], curr[5], curr[6], curr[7],
    ]);
    let a1 = u32x8::new([
        prev[8], prev[9], prev[10], prev[11], prev[12], prev[13], prev[14], prev[15],
    ]);
    let b1 = u32x8::new([
        curr[8], curr[9], curr[10], curr[11], curr[12], curr[13], curr[14], curr[15],
    ]);

    let i0 = sum_u32x8(a0.min(b0));
    let i1 = sum_u32x8(a1.min(b1));
    let intersection_u64 = i0 + i1;

    let total_u64 = u64::try_from(total_pixels).unwrap_or(u64::MAX);
    if total_u64 == 0 {
        return 0.0;
    }

    let similarity = u64_to_f64_exact(intersection_u64) / u64_to_f64_exact(total_u64);
    (1.0 - similarity).clamp(0.0, 1.0)
}

pub fn downscale_gray8_nearest(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    dst: &mut Vec<u8>,
) {
    DownscalePlan::new(src_width, src_height, dst_width, dst_height).run(src, dst);
}

#[must_use]
pub fn grid_histogram_16(frame: &[u8], width: usize, height: usize) -> [u32; GRID_HIST_LEN] {
    GridHistogramPlan::new(width, height).run(frame)
}

#[must_use]
pub fn grid_histogram_median_cell_distance_16_from_hists(
    prev: &[u32; GRID_HIST_LEN],
    curr: &[u32; GRID_HIST_LEN],
) -> f64 {
    grid_histogram_distances_16_from_hists(prev, curr, 0).1
}

#[must_use]
pub fn grid_histogram_distance_16_from_hists(
    prev: &[u32; GRID_HIST_LEN],
    curr: &[u32; GRID_HIST_LEN],
    total_pixels: usize,
) -> f64 {
    grid_histogram_distances_16_from_hists(prev, curr, total_pixels).0
}

#[must_use]
pub fn grid_histogram_distances_16_from_hists(
    prev: &[u32; GRID_HIST_LEN],
    curr: &[u32; GRID_HIST_LEN],
    total_pixels: usize,
) -> (f64, f64) {
    let cells = GRID_HIST_X.saturating_mul(GRID_HIST_Y);
    let mut distances: [f64; GRID_HIST_X * GRID_HIST_Y] = [0.0; GRID_HIST_X * GRID_HIST_Y];
    let mut count: usize = 0;
    let mut intersection: u64 = 0;

    for cell in 0..cells {
        let base = cell.saturating_mul(16);
        let mut cell_intersection: u64 = 0;
        let mut cell_total: u64 = 0;

        for bin in 0..16 {
            let a = prev[base + bin];
            let b = curr[base + bin];
            let bin_intersection = u64::from(a.min(b));
            cell_intersection += bin_intersection;
            intersection += bin_intersection;
            cell_total += u64::from(a);
        }

        if cell_total == 0 {
            continue;
        }

        let similarity = u64_to_f64_exact(cell_intersection) / u64_to_f64_exact(cell_total);
        distances[count] = (1.0 - similarity).clamp(0.0, 1.0);
        count = count.saturating_add(1);
    }

    let median = median_distance(&mut distances, count);
    let total_u64 = u64::try_from(total_pixels).unwrap_or(u64::MAX);
    if total_u64 == 0 {
        return (0.0, median);
    }

    let similarity = u64_to_f64_exact(intersection) / u64_to_f64_exact(total_u64);
    ((1.0 - similarity).clamp(0.0, 1.0), median)
}

fn median_distance(distances: &mut [f64; GRID_HIST_X * GRID_HIST_Y], count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }

    let slice = &mut distances[..count];
    let mid = count / 2;
    slice.select_nth_unstable_by(mid, f64::total_cmp);
    let upper = slice[mid];
    if count % 2 == 1 || mid == 0 {
        upper
    } else {
        let lower = slice[..mid]
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .unwrap_or(upper);
        (lower + upper) * 0.5
    }
}

#[must_use]
pub fn histogram_16(frame: &[u8]) -> [u32; 16] {
    let mut hist = [0_u32; 16];
    for &px in frame {
        hist[usize::from(px >> 4)] += 1;
    }
    hist
}

#[derive(Clone, Copy, Debug)]
pub struct MeAnalysis {
    pub score: f64,
    pub intra_blocks: usize,
    pub total_blocks: usize,
    pub mc_sad: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MotionAnalysisPlan {
    width: usize,
    height: usize,
    search_radius: i32,
    total_blocks: usize,
    blocks: u64,
    x_indices: Vec<usize>,
    y_indices: Vec<usize>,
    should_parallelize: bool,
    max_horizontal_limit: usize,
    max_vertical_limit: usize,
}

thread_local! {
    static MOTION_PLAN_CACHE: RefCell<Option<MotionAnalysisPlan>> = const { RefCell::new(None) };
}

const BLOCK_EDGE: usize = 16;
const MIN_DEV: u32 = 300;
const DEV_STATIC: u32 = 1000;
const SAD_STATIC: u32 = 1000;
const DEV_BIAS: u32 = 512;
const DEV_HALF: u32 = 3000;
const PARALLEL_SUPERBLOCK_THRESHOLD: usize = 24;
const TWO_POW_32_F64: f64 = 4_294_967_296.0;

#[derive(Clone, Copy, Debug)]
struct MotionMatch {
    sad: u32,
    dx: i32,
    dy: i32,
}

#[derive(Clone, Copy, Debug)]
struct BlockPair {
    curr_x: usize,
    curr_y: usize,
    prev_x: usize,
    prev_y: usize,
}

fn sad16x16_with_cutoff(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    pair: BlockPair,
    cutoff: u32,
) -> u32 {
    let mut total: u32 = 0;
    for y in 0..16_usize {
        let curr_row = (pair.curr_y + y) * width;
        let prev_row = (pair.prev_y + y) * width;
        let curr_idx = curr_row + pair.curr_x;
        let prev_idx = prev_row + pair.prev_x;
        let row_total = sad16_bytes(&curr[curr_idx..], &prev[prev_idx..]);
        total = total.saturating_add(u32::try_from(row_total).unwrap_or(u32::MAX));
        if total > cutoff {
            return total;
        }
    }
    total
}

fn find_best_match_for_block(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    height: usize,
    block_x: usize,
    block_y: usize,
    search_radius: i32,
) -> MotionMatch {
    let block_origin_x = i32::try_from(block_x).unwrap_or(i32::MAX);
    let block_origin_y = i32::try_from(block_y).unwrap_or(i32::MAX);
    let max_search_x = width.saturating_sub(BLOCK_EDGE);
    let max_search_y = height.saturating_sub(BLOCK_EDGE);

    let min_offset_x = (-search_radius).max(-block_origin_x);
    let max_offset_x =
        search_radius.min(i32::try_from(max_search_x.saturating_sub(block_x)).unwrap_or(i32::MAX));
    let min_offset_y = (-search_radius).max(-block_origin_y);
    let max_offset_y =
        search_radius.min(i32::try_from(max_search_y.saturating_sub(block_y)).unwrap_or(i32::MAX));

    let mut best = MotionMatch {
        sad: u32::MAX,
        dx: 0,
        dy: 0,
    };

    for dy in min_offset_y..=max_offset_y {
        let candidate_row = apply_search_offset(block_y, dy);
        for dx in min_offset_x..=max_offset_x {
            let candidate_col = apply_search_offset(block_x, dx);
            let sad = sad16x16_with_cutoff(
                prev,
                curr,
                width,
                BlockPair {
                    curr_x: block_x,
                    curr_y: block_y,
                    prev_x: candidate_col,
                    prev_y: candidate_row,
                },
                best.sad,
            );
            if sad < best.sad {
                best.sad = sad;
                best.dx = dx;
                best.dy = dy;
            }
        }
    }

    best
}

#[expect(
    clippy::cast_sign_loss,
    reason = "search offsets are bounded by frame dimensions before this hot-loop helper"
)]
fn apply_search_offset(origin: usize, offset: i32) -> usize {
    if offset < 0 {
        origin.saturating_sub(offset.unsigned_abs() as usize)
    } else {
        origin.saturating_add(offset as usize)
    }
}

fn dev_block(
    curr: &[u8],
    width: usize,
    curr_x: usize,
    curr_y: usize,
    block_w: usize,
    block_h: usize,
) -> u32 {
    let area = block_w.saturating_mul(block_h);
    if area == 0 {
        return 0;
    }

    #[cfg(target_arch = "x86_64")]
    if block_w == BLOCK_EDGE && block_h == BLOCK_EDGE {
        // SAFETY: motion-analysis callers only request in-bounds 16x16 blocks.
        return unsafe { dev_block_16x16_sse2(curr, width, curr_x, curr_y) };
    }

    let mut sum: u32 = 0;
    for y in 0..block_h {
        let row = (curr_y + y).saturating_mul(width);
        for x in 0..block_w {
            sum = sum.saturating_add(u32::from(curr[row + curr_x + x]));
        }
    }

    let mean = sum / u32::try_from(area).unwrap_or(u32::MAX);
    let mut dev = 0_u32;
    for y in 0..block_h {
        let row = (curr_y + y).saturating_mul(width);
        for x in 0..block_w {
            dev = dev.saturating_add(u32::from(curr[row + curr_x + x]).abs_diff(mean));
        }
    }
    dev
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[expect(
    clippy::cast_ptr_alignment,
    reason = "_mm_loadu_si128 performs an unaligned load"
)]
unsafe fn dev_block_16x16_sse2(curr: &[u8], width: usize, curr_x: usize, curr_y: usize) -> u32 {
    let zero = _mm_setzero_si128();
    let mut sum = 0_u32;
    for y in 0..BLOCK_EDGE {
        let idx = (curr_y + y).saturating_mul(width) + curr_x;
        let row = unsafe { _mm_loadu_si128(curr.as_ptr().add(idx).cast::<__m128i>()) };
        let sad = _mm_sad_epu8(row, zero);
        sum = sum.saturating_add(unsafe { sum_sad_lanes_u32(sad) });
    }

    let mean = sum / 256;
    let mean_byte = u8::try_from(mean).unwrap_or(u8::MAX);
    let mean_vector = _mm_set1_epi8(i8::from_ne_bytes([mean_byte]));
    let mut dev = 0_u32;
    for y in 0..BLOCK_EDGE {
        let idx = (curr_y + y).saturating_mul(width) + curr_x;
        let row = unsafe { _mm_loadu_si128(curr.as_ptr().add(idx).cast::<__m128i>()) };
        let sad = _mm_sad_epu8(row, mean_vector);
        dev = dev.saturating_add(unsafe { sum_sad_lanes_u32(sad) });
    }

    dev
}

fn superblock_indices(macroblock_count: usize) -> Vec<usize> {
    match macroblock_count {
        0..=2 => Vec::new(),
        3 => vec![1],
        _ => (1..macroblock_count.saturating_sub(2)).step_by(2).collect(),
    }
}

#[derive(Clone, Copy, Debug)]
struct SearchWindow {
    min_offset_x: i32,
    max_offset_x: i32,
    min_offset_y: i32,
    max_offset_y: i32,
}

#[derive(Clone, Copy, Debug)]
struct SuperblockSearchResults {
    sad: [u32; 4],
    dx: [i32; 4],
    dy: [i32; 4],
}

fn empty_me_analysis() -> MeAnalysis {
    MeAnalysis {
        score: 0.0,
        intra_blocks: 0,
        total_blocks: 0,
        mc_sad: 0,
    }
}

fn available_workers() -> usize {
    static AVAILABLE_WORKERS: OnceLock<usize> = OnceLock::new();
    *AVAILABLE_WORKERS
        .get_or_init(|| std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
}

fn superblock_budget(superblock_count: usize) -> u64 {
    let count_u64 = u64::try_from(superblock_count).unwrap_or(u64::MAX);
    10_u64.saturating_add(10_u64.saturating_mul(count_u64))
}

#[derive(Clone, Copy, Debug)]
struct SuperblockEvalContext<'a> {
    prev: &'a [u8],
    curr: &'a [u8],
    width: usize,
    search_radius: i32,
    intra_thresh_u32: u32,
    max_horizontal_limit: usize,
    max_vertical_limit: usize,
}

fn analyze_superblock_row(
    context: &SuperblockEvalContext<'_>,
    x_indices: &[usize],
    macroblock_y: usize,
) -> (usize, u64, u64) {
    let base_y = macroblock_y.saturating_mul(BLOCK_EDGE);
    let mut row_intra = 0_usize;
    let mut row_ssad = 0_u64;
    let mut row_complexity = 0_u64;

    for macroblock_x in x_indices.iter().copied() {
        let base_x = macroblock_x.saturating_mul(BLOCK_EDGE);
        let search_window = superblock_search_window(
            base_x,
            base_y,
            context.search_radius,
            context.max_horizontal_limit,
            context.max_vertical_limit,
        );
        let search_results = search_superblock_matches(
            context.prev,
            context.curr,
            context.width,
            base_x,
            base_y,
            search_window,
        );
        let (intra_delta, ssad_delta, complexity_delta) = accumulate_superblock_metrics(
            context.curr,
            context.width,
            base_x,
            base_y,
            context.intra_thresh_u32,
            search_results,
        );
        row_intra = row_intra.saturating_add(intra_delta);
        row_ssad = row_ssad.saturating_add(ssad_delta);
        row_complexity = row_complexity.saturating_add(complexity_delta);
    }

    (row_intra, row_ssad, row_complexity)
}

fn superblock_search_window(
    base_x: usize,
    base_y: usize,
    search_radius: i32,
    max_horizontal_limit: usize,
    max_vertical_limit: usize,
) -> SearchWindow {
    let base_col_i32 = i32::try_from(base_x).unwrap_or(i32::MAX);
    let base_row_i32 = i32::try_from(base_y).unwrap_or(i32::MAX);
    let min_offset_x = (-search_radius).max(-base_col_i32);
    let max_offset_x = search_radius
        .min(i32::try_from(max_horizontal_limit.saturating_sub(base_x)).unwrap_or(i32::MAX));
    let min_offset_y = (-search_radius).max(-base_row_i32);
    let max_offset_y = search_radius
        .min(i32::try_from(max_vertical_limit.saturating_sub(base_y)).unwrap_or(i32::MAX));

    SearchWindow {
        min_offset_x,
        max_offset_x,
        min_offset_y,
        max_offset_y,
    }
}

fn search_superblock_matches(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    base_x: usize,
    base_y: usize,
    search_window: SearchWindow,
) -> SuperblockSearchResults {
    let mut result = SuperblockSearchResults {
        sad: [u32::MAX; 4],
        dx: [0_i32; 4],
        dy: [0_i32; 4],
    };

    for offset_y in search_window.min_offset_y..=search_window.max_offset_y {
        let candidate_row = apply_search_offset(base_y, offset_y);
        for offset_x in search_window.min_offset_x..=search_window.max_offset_x {
            let candidate_col = apply_search_offset(base_x, offset_x);
            let block_sad_0 = sad16x16_with_cutoff(
                prev,
                curr,
                width,
                BlockPair {
                    curr_x: base_x,
                    curr_y: base_y,
                    prev_x: candidate_col,
                    prev_y: candidate_row,
                },
                result.sad[0],
            );
            if block_sad_0 < result.sad[0] {
                result.sad[0] = block_sad_0;
                result.dx[0] = offset_x;
                result.dy[0] = offset_y;
            }

            let block_sad_1 = sad16x16_with_cutoff(
                prev,
                curr,
                width,
                BlockPair {
                    curr_x: base_x + BLOCK_EDGE,
                    curr_y: base_y,
                    prev_x: candidate_col + BLOCK_EDGE,
                    prev_y: candidate_row,
                },
                result.sad[1],
            );
            if block_sad_1 < result.sad[1] {
                result.sad[1] = block_sad_1;
                result.dx[1] = offset_x;
                result.dy[1] = offset_y;
            }

            let block_sad_2 = sad16x16_with_cutoff(
                prev,
                curr,
                width,
                BlockPair {
                    curr_x: base_x,
                    curr_y: base_y + BLOCK_EDGE,
                    prev_x: candidate_col,
                    prev_y: candidate_row + BLOCK_EDGE,
                },
                result.sad[2],
            );
            if block_sad_2 < result.sad[2] {
                result.sad[2] = block_sad_2;
                result.dx[2] = offset_x;
                result.dy[2] = offset_y;
            }

            let block_sad_3 = sad16x16_with_cutoff(
                prev,
                curr,
                width,
                BlockPair {
                    curr_x: base_x + BLOCK_EDGE,
                    curr_y: base_y + BLOCK_EDGE,
                    prev_x: candidate_col + BLOCK_EDGE,
                    prev_y: candidate_row + BLOCK_EDGE,
                },
                result.sad[3],
            );
            if block_sad_3 < result.sad[3] {
                result.sad[3] = block_sad_3;
                result.dx[3] = offset_x;
                result.dy[3] = offset_y;
            }
        }
    }

    result
}

fn accumulate_superblock_metrics(
    curr: &[u8],
    width: usize,
    base_x: usize,
    base_y: usize,
    intra_thresh_u32: u32,
    search_results: SuperblockSearchResults,
) -> (usize, u64, u64) {
    let mut intra_delta = 0_usize;
    let mut ssad_delta = 0_u64;
    let mut complexity_delta = 0_u64;

    for (index, (offset_x, offset_y)) in [
        (0_usize, 0_usize),
        (BLOCK_EDGE, 0),
        (0, BLOCK_EDGE),
        (BLOCK_EDGE, BLOCK_EDGE),
    ]
    .into_iter()
    .enumerate()
    {
        let block_x = base_x + offset_x;
        let block_y = base_y + offset_y;
        let dev = dev_block(curr, width, block_x, block_y, BLOCK_EDGE, BLOCK_EDGE);
        complexity_delta = complexity_delta.saturating_add(u64::from(dev.max(MIN_DEV)));

        let sad = search_results.sad[index];
        if dev.saturating_add(intra_thresh_u32) < sad {
            intra_delta = intra_delta.saturating_add(1);
        }

        if search_results.dx[index] == 0
            && search_results.dy[index] == 0
            && dev > DEV_STATIC
            && sad < SAD_STATIC
        {
            ssad_delta = ssad_delta.saturating_add(u64::from(DEV_BIAS));
        }

        let adjusted_sad = if dev < DEV_HALF { sad } else { sad / 2 };
        ssad_delta = ssad_delta.saturating_add(u64::from(adjusted_sad));
    }

    (intra_delta, ssad_delta, complexity_delta)
}

fn small_frame_meanalysis(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    height: usize,
    search_radius: i32,
    intra_thresh_u32: u32,
) -> MeAnalysis {
    let mb_width = width.div_ceil(BLOCK_EDGE);
    let mb_height = height.div_ceil(BLOCK_EDGE);
    let mut total_blocks = 0_usize;
    let mut intra_blocks = 0_usize;
    let mut ssad = 0_u64;
    let mut complexity = 0_u64;
    let mut blocks = 10_u64;

    for macroblock_y in 1..=mb_height.saturating_sub(2) {
        let block_y = macroblock_y.saturating_mul(BLOCK_EDGE);
        for macroblock_x in 1..=mb_width.saturating_sub(2) {
            let block_x = macroblock_x.saturating_mul(BLOCK_EDGE);
            if block_x.saturating_add(BLOCK_EDGE) > width
                || block_y.saturating_add(BLOCK_EDGE) > height
            {
                continue;
            }

            blocks = blocks.saturating_add(10);
            let motion_match = find_best_match_for_block(
                prev,
                curr,
                width,
                height,
                block_x,
                block_y,
                search_radius,
            );
            let dev = dev_block(curr, width, block_x, block_y, BLOCK_EDGE, BLOCK_EDGE);
            complexity = complexity.saturating_add(u64::from(dev.max(MIN_DEV)));
            if dev.saturating_add(intra_thresh_u32) < motion_match.sad {
                intra_blocks = intra_blocks.saturating_add(1);
            }
            if motion_match.dx == 0
                && motion_match.dy == 0
                && dev > DEV_STATIC
                && motion_match.sad < SAD_STATIC
            {
                ssad = ssad.saturating_add(u64::from(DEV_BIAS));
            }
            let adjusted_sad = if dev < DEV_HALF {
                motion_match.sad
            } else {
                motion_match.sad / 2
            };
            ssad = ssad.saturating_add(u64::from(adjusted_sad));
            total_blocks = total_blocks.saturating_add(1);
        }
    }

    if total_blocks == 0 {
        return empty_me_analysis();
    }

    let complexity_scaled = complexity >> 7;
    let denom = complexity_scaled.saturating_add(4_u64.saturating_mul(blocks));
    let score = if denom == 0 {
        0.0
    } else {
        u64_to_f64_exact(ssad / denom)
    };

    MeAnalysis {
        score,
        intra_blocks,
        total_blocks,
        mc_sad: ssad,
    }
}

fn build_motion_analysis_plan(
    width: usize,
    height: usize,
    search_radius: i32,
) -> Option<MotionAnalysisPlan> {
    let mb_width = width.div_ceil(BLOCK_EDGE);
    let mb_height = height.div_ceil(BLOCK_EDGE);
    let interior_width = mb_width.saturating_sub(2);
    let interior_height = mb_height.saturating_sub(2);
    let total_blocks = interior_width.saturating_mul(interior_height);
    if total_blocks == 0 {
        return None;
    }

    let x_indices = superblock_indices(mb_width);
    let y_indices = superblock_indices(mb_height);
    if x_indices.is_empty() || y_indices.is_empty() {
        return None;
    }

    let superblock_count = x_indices.len().saturating_mul(y_indices.len());

    Some(MotionAnalysisPlan {
        width,
        height,
        search_radius,
        total_blocks,
        blocks: superblock_budget(superblock_count),
        should_parallelize: superblock_count >= PARALLEL_SUPERBLOCK_THRESHOLD
            && available_workers() > 1,
        x_indices,
        y_indices,
        max_horizontal_limit: width.saturating_sub(32),
        max_vertical_limit: height.saturating_sub(32),
    })
}

fn with_cached_motion_analysis_plan<T, F>(
    width: usize,
    height: usize,
    search_radius: i32,
    f: F,
) -> T
where
    F: FnOnce(Option<&MotionAnalysisPlan>) -> T,
{
    MOTION_PLAN_CACHE.with(|cache| {
        {
            let mut cache_ref = cache.borrow_mut();
            let needs_refresh = cache_ref.as_ref().is_none_or(|plan| {
                plan.width != width || plan.height != height || plan.search_radius != search_radius
            });
            if needs_refresh {
                *cache_ref = build_motion_analysis_plan(width, height, search_radius);
            }
        }

        let cache_ref = cache.borrow();
        f(cache_ref.as_ref())
    })
}

fn run_superblock_motion_analysis(
    prev: &[u8],
    curr: &[u8],
    intra_thresh_u32: u32,
    plan: &MotionAnalysisPlan,
) -> MeAnalysis {
    let context = SuperblockEvalContext {
        prev,
        curr,
        width: plan.width,
        search_radius: plan.search_radius,
        intra_thresh_u32,
        max_horizontal_limit: plan.max_horizontal_limit,
        max_vertical_limit: plan.max_vertical_limit,
    };

    let (intra_blocks, ssad, complexity) = if plan.should_parallelize {
        plan.y_indices
            .par_iter()
            .copied()
            .map(|macroblock_y| analyze_superblock_row(&context, &plan.x_indices, macroblock_y))
            .reduce(
                || (0_usize, 0_u64, 0_u64),
                |left, right| {
                    (
                        left.0.saturating_add(right.0),
                        left.1.saturating_add(right.1),
                        left.2.saturating_add(right.2),
                    )
                },
            )
    } else {
        let mut intra_blocks = 0_usize;
        let mut ssad = 0_u64;
        let mut complexity = 0_u64;

        for macroblock_y in plan.y_indices.iter().copied() {
            let (row_intra, row_ssad, row_complexity) =
                analyze_superblock_row(&context, &plan.x_indices, macroblock_y);
            intra_blocks = intra_blocks.saturating_add(row_intra);
            ssad = ssad.saturating_add(row_ssad);
            complexity = complexity.saturating_add(row_complexity);
        }

        (intra_blocks, ssad, complexity)
    };

    let complexity_scaled = complexity >> 7;
    let denom = complexity_scaled.saturating_add(4_u64.saturating_mul(plan.blocks));
    let score = if denom == 0 {
        0.0
    } else {
        u64_to_f64_exact(ssad / denom)
    };

    MeAnalysis {
        score,
        intra_blocks,
        total_blocks: plan.total_blocks,
        mc_sad: ssad,
    }
}

#[must_use]
pub fn meanalysis_xvid_like(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    height: usize,
    search_radius: i32,
    intra_thresh: i32,
) -> MeAnalysis {
    if width < 32 || height < 32 {
        return empty_me_analysis();
    }
    let needed = width.saturating_mul(height);
    if prev.len() < needed || curr.len() < needed {
        return empty_me_analysis();
    }

    let mb_width = width.div_ceil(BLOCK_EDGE);
    let mb_height = height.div_ceil(BLOCK_EDGE);
    let interior_width = mb_width.saturating_sub(2);
    let interior_height = mb_height.saturating_sub(2);
    let total_blocks = interior_width.saturating_mul(interior_height);
    if total_blocks == 0 {
        return empty_me_analysis();
    }

    let intra_thresh_u32 = u32::try_from(intra_thresh).unwrap_or(0);
    if mb_width == 3 || mb_height == 3 {
        return small_frame_meanalysis(prev, curr, width, height, search_radius, intra_thresh_u32);
    }

    with_cached_motion_analysis_plan(width, height, search_radius, |plan| {
        plan.map_or_else(empty_me_analysis, |plan| {
            run_superblock_motion_analysis(prev, curr, intra_thresh_u32, plan)
        })
    })
}

/// Sum of absolute differences over the first 16 bytes of each slice.
///
/// Callers must pass slices with at least 16 bytes.
#[inline]
fn sad16_bytes(prev: &[u8], curr: &[u8]) -> u64 {
    debug_assert!(prev.len() >= 16);
    debug_assert!(curr.len() >= 16);

    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of the x86_64 baseline target features.
        unsafe { sad16_bytes_sse2(prev.as_ptr(), curr.as_ptr()) }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let a: [u8; 16] = prev[..16].try_into().expect("sliced to 16 bytes");
        let b: [u8; 16] = curr[..16].try_into().expect("sliced to 16 bytes");
        let a = u8x16::new(a);
        let b = u8x16::new(b);
        let diff = a.max(b) - a.min(b);
        sum_u16x8(u16x8::from_u8x16_low(diff)) + sum_u16x8(u16x8::from_u8x16_high(diff))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[expect(
    clippy::cast_ptr_alignment,
    reason = "_mm_loadu_si128 performs an unaligned load"
)]
unsafe fn sad16_bytes_sse2(a: *const u8, b: *const u8) -> u64 {
    let a = unsafe { _mm_loadu_si128(a.cast::<__m128i>()) };
    let b = unsafe { _mm_loadu_si128(b.cast::<__m128i>()) };
    let sad = _mm_sad_epu8(a, b);
    unsafe { sum_sad_lanes_u64(sad) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn sum_sad_lanes_u64(sad: __m128i) -> u64 {
    let low = _mm_cvtsi128_si64(sad).cast_unsigned();
    let high = _mm_cvtsi128_si64(_mm_srli_si128::<8>(sad)).cast_unsigned();
    low + high
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn sum_sad_lanes_u32(sad: __m128i) -> u32 {
    u32::try_from(unsafe { sum_sad_lanes_u64(sad) }).unwrap_or(u32::MAX)
}

#[cfg(not(target_arch = "x86_64"))]
fn sum_u16x8(v: u16x8) -> u64 {
    v.to_array().into_iter().map(u64::from).sum()
}

fn sum_u32x8(v: u32x8) -> u64 {
    v.to_array().into_iter().map(u64::from).sum()
}

fn u64_to_f64_exact(value: u64) -> f64 {
    let high = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low_mask = u64::from(u32::MAX);
    let low = u32::try_from(value & low_mask).unwrap_or(u32::MAX);
    (f64::from(high) * TWO_POW_32_F64) + f64::from(low)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_histogram_distance_16(prev: &[u8], curr: &[u8]) -> f64 {
        if prev.len() != curr.len() || prev.is_empty() {
            return 0.0;
        }

        let prev_hist = histogram_16(prev);
        let curr_hist = histogram_16(curr);
        let intersection: u64 = prev_hist
            .into_iter()
            .zip(curr_hist)
            .map(|(a, b)| u64::from(a.min(b)))
            .sum();

        let total = u64::try_from(prev.len()).unwrap_or(u64::MAX);
        if total == 0 {
            return 0.0;
        }
        let similarity = u64_to_f64_exact(intersection) / u64_to_f64_exact(total);
        (1.0 - similarity).clamp(0.0, 1.0)
    }

    fn make_frame(len: usize, offset: usize) -> Vec<u8> {
        (0..len)
            .map(|idx| u8::try_from((idx + offset) % 256).unwrap_or(0))
            .collect()
    }

    fn meanalysis_xvid_like_fresh_plan(
        prev: &[u8],
        curr: &[u8],
        width: usize,
        height: usize,
        search_radius: i32,
        intra_thresh: i32,
    ) -> MeAnalysis {
        if width < 32 || height < 32 {
            return empty_me_analysis();
        }

        let mb_width = width.div_ceil(BLOCK_EDGE);
        let mb_height = height.div_ceil(BLOCK_EDGE);
        let intra_thresh_u32 = u32::try_from(intra_thresh).unwrap_or(0);
        if mb_width == 3 || mb_height == 3 {
            return small_frame_meanalysis(
                prev,
                curr,
                width,
                height,
                search_radius,
                intra_thresh_u32,
            );
        }

        let Some(plan) = build_motion_analysis_plan(width, height, search_radius) else {
            return empty_me_analysis();
        };

        run_superblock_motion_analysis(prev, curr, intra_thresh_u32, &plan)
    }

    #[test]
    fn histogram_distance_matches_cached_path() {
        let prev = make_frame(4096, 0);
        let curr = make_frame(4096, 7);

        let direct = scalar_histogram_distance_16(&prev, &curr);
        let prev_hist = histogram_16(&prev);
        let curr_hist = histogram_16(&curr);
        let cached = histogram_distance_16_from_hists(&prev_hist, &curr_hist, prev.len());

        assert!((direct - cached).abs() < 1e-12);
    }

    #[test]
    fn histogram_distance_zero_on_identical() {
        let frame = make_frame(2048, 0);
        let hist = histogram_16(&frame);
        assert!(scalar_histogram_distance_16(&frame, &frame).abs() < f64::EPSILON);
        assert!(histogram_distance_16_from_hists(&hist, &hist, frame.len()).abs() < f64::EPSILON);
    }

    #[test]
    fn grid_histogram_distance_zero_on_identical() {
        let width: usize = 64;
        let height: usize = 48;
        let frame = make_frame(width * height, 0);
        let prev = grid_histogram_16(&frame, width, height);
        let curr = grid_histogram_16(&frame, width, height);
        let distance = grid_histogram_distance_16_from_hists(&prev, &curr, frame.len());
        assert!(distance.abs() < f64::EPSILON);
    }

    #[test]
    fn sad_u8_matches_scalar_reference() {
        // 1003 is not a multiple of 16, so the tail path is covered too.
        let prev = make_frame(1003, 3);
        let curr: Vec<u8> = (0..1003_usize)
            .map(|idx| u8::try_from((idx * 7 + 129) % 256).unwrap_or(0))
            .collect();

        let reference: u64 = prev
            .iter()
            .zip(&curr)
            .map(|(&a, &b)| u64::from(a.abs_diff(b)))
            .sum();

        assert_eq!(sad_u8(&prev, &curr), reference);
    }

    #[test]
    fn planar16_scaling_maps_range_endpoints() {
        let encode = |values: &[u16], le: bool| -> Vec<u8> {
            values
                .iter()
                .flat_map(|v| if le { v.to_le_bytes() } else { v.to_be_bytes() })
                .collect()
        };
        let limited = Planar16Params {
            bit_depth: 10,
            little_endian: true,
            full_range: false,
        };
        let values = [0_u16, 64, 502, 512, 940, 1023];
        let mut dst = Vec::new();

        // Limited range: 64 is black, 940 is white; the divide truncates, so
        // the exact midpoint 502 (127.5) lands on 127.
        extract_planar16_packed8(&encode(&values, true), 12, 6, 1, limited, &mut dst);
        assert_eq!(dst, [0, 0, 127, 130, 255, 255]);

        // Big-endian variant decodes to the same output.
        let be = Planar16Params {
            little_endian: false,
            ..limited
        };
        extract_planar16_packed8(&encode(&values, false), 12, 6, 1, be, &mut dst);
        assert_eq!(dst, [0, 0, 127, 130, 255, 255]);

        // Full range: linear mapping over the whole code range.
        let full = Planar16Params {
            full_range: true,
            ..limited
        };
        extract_planar16_packed8(&encode(&values, true), 12, 6, 1, full, &mut dst);
        assert_eq!(dst, [0, 16, 125, 128, 234, 255]);
    }

    #[test]
    fn fused_grid_distances_match_scalar_reference() {
        let width: usize = 96;
        let height: usize = 54;
        let prev_frame = make_frame(width * height, 5);
        let curr_frame = make_frame(width * height, 41);
        let prev = grid_histogram_16(&prev_frame, width, height);
        let curr = grid_histogram_16(&curr_frame, width, height);
        let total = width * height;

        // Independent scalar reference for both metrics.
        let intersection: u64 = prev
            .iter()
            .zip(&curr)
            .map(|(&a, &b)| u64::from(a.min(b)))
            .sum();
        let expected_global =
            1.0 - u64_to_f64_exact(intersection) / u64_to_f64_exact(u64::try_from(total).unwrap());
        let mut cell_distances: Vec<f64> = (0..GRID_HIST_X * GRID_HIST_Y)
            .filter_map(|cell| {
                let base = cell * 16;
                let cell_intersection: u64 = (0..16)
                    .map(|bin| u64::from(prev[base + bin].min(curr[base + bin])))
                    .sum();
                let cell_total: u64 = (0..16).map(|bin| u64::from(prev[base + bin])).sum();
                (cell_total > 0).then(|| {
                    1.0 - u64_to_f64_exact(cell_intersection) / u64_to_f64_exact(cell_total)
                })
            })
            .collect();
        cell_distances.sort_by(f64::total_cmp);
        let mid = cell_distances.len() / 2;
        let expected_median = if cell_distances.len() % 2 == 1 {
            cell_distances[mid]
        } else {
            (cell_distances[mid - 1] + cell_distances[mid]) * 0.5
        };

        let (global, median) = grid_histogram_distances_16_from_hists(&prev, &curr, total);
        assert!((global - expected_global).abs() < 1e-12);
        assert!((median - expected_median).abs() < 1e-12);

        // The single-metric wrappers must agree with the fused pair.
        let wrapper_global = grid_histogram_distance_16_from_hists(&prev, &curr, total);
        let wrapper_median = grid_histogram_median_cell_distance_16_from_hists(&prev, &curr);
        assert!((global - wrapper_global).abs() < 1e-12);
        assert!((median - wrapper_median).abs() < 1e-12);

        // Zero pixels: no global distance, median still defined.
        let (zero_global, zero_median) = grid_histogram_distances_16_from_hists(&prev, &curr, 0);
        assert!(zero_global.abs() < f64::EPSILON);
        assert!((zero_median - expected_median).abs() < 1e-12);
    }

    #[test]
    fn downscale_plan_matches_direct_path() {
        let src_width: usize = 64;
        let src_height: usize = 48;
        let dst_width: usize = 17;
        let dst_height: usize = 11;
        let frame = make_frame(src_width * src_height, 3);

        let mut direct = Vec::new();
        downscale_gray8_nearest(
            &frame,
            src_width,
            src_height,
            dst_width,
            dst_height,
            &mut direct,
        );

        let mut planned = Vec::new();
        DownscalePlan::new(src_width, src_height, dst_width, dst_height).run(&frame, &mut planned);

        assert_eq!(planned, direct);
    }

    #[test]
    fn exact_decimation_downscale_matches_nearest_mapping() {
        let src_width: usize = 192;
        let src_height: usize = 108;
        let dst_width: usize = 16;
        let dst_height: usize = 9;
        let frame = make_frame(src_width * src_height, 37);
        let mut expected = Vec::with_capacity(dst_width * dst_height);
        for y in 0..dst_height {
            let src_y = y * src_height / dst_height;
            for x in 0..dst_width {
                let src_x = x * src_width / dst_width;
                expected.push(frame[src_y * src_width + src_x]);
            }
        }

        let mut planned = Vec::new();
        DownscalePlan::new(src_width, src_height, dst_width, dst_height).run(&frame, &mut planned);

        assert_eq!(planned, expected);
    }

    #[test]
    fn exact_decimation_planar16_downscale_matches_nearest_mapping() {
        let src_width: usize = 8;
        let src_height: usize = 4;
        let dst_width: usize = 4;
        let dst_height: usize = 2;
        let params = Planar16Params {
            bit_depth: 10,
            little_endian: true,
            full_range: true,
        };
        let values: Vec<u16> = (0..src_width * src_height)
            .map(|idx| u16::try_from(idx * 23).expect("fits in u16"))
            .collect();
        let src: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();

        let mut expected = Vec::with_capacity(dst_width * dst_height);
        for y in 0..dst_height {
            let src_y = y * src_height / dst_height;
            for x in 0..dst_width {
                let src_x = x * src_width / dst_width;
                expected.push(scale_luma16_to_u8(
                    values[src_y * src_width + src_x],
                    params,
                ));
            }
        }

        let mut planned = Vec::new();
        DownscalePlan::new(src_width, src_height, dst_width, dst_height).run_planar16_packed8(
            &src,
            src_width * 2,
            params,
            &mut planned,
        );

        assert_eq!(planned, expected);
    }

    #[test]
    fn grid_histogram_plan_matches_direct_path() {
        let width: usize = 96;
        let height: usize = 54;
        let frame = make_frame(width * height, 11);

        let direct = grid_histogram_16(&frame, width, height);
        let planned = GridHistogramPlan::new(width, height).run(&frame);

        assert_eq!(planned, direct);
    }

    #[test]
    fn accumulated_frame_metrics_match_separate_paths() {
        let width: usize = 96;
        let height: usize = 54;
        let prev = make_frame(width * height, 5);
        let curr = make_frame(width * height, 17);
        let plan = GridHistogramPlan::new(width, height);

        let combined = plan.accumulate_frame_metrics(Some(&prev), &curr);

        assert_eq!(combined.sad, sad_u8(&prev, &curr));
        assert_eq!(combined.hist, histogram_16(&curr));
        assert_eq!(combined.grid_hist, plan.run(&curr));
    }

    #[test]
    fn cutoff_sad_matches_full_sad_when_not_cut() {
        let width: usize = 64;
        let height: usize = 64;
        let prev = make_frame(width * height, 0);
        let curr = make_frame(width * height, 9);

        let pair = BlockPair {
            curr_x: 16,
            curr_y: 16,
            prev_x: 16,
            prev_y: 16,
        };
        let full = sad16x16_with_cutoff(&prev, &curr, width, pair, u32::MAX);
        let cutoff = sad16x16_with_cutoff(&prev, &curr, width, pair, full);

        assert_eq!(cutoff, full);
    }

    #[test]
    fn dev_block_histogram_matches_scalar_baseline() {
        let width: usize = 64;
        let height: usize = 64;
        let curr = make_frame(width * height, 13);

        let scalar = {
            let mut sum: u32 = 0;
            for y in 0..16_usize {
                let row = (8 + y) * width;
                for x in 0..16_usize {
                    sum = sum.saturating_add(u32::from(curr[row + 12 + x]));
                }
            }

            let mean = sum / 256;
            let mut dev = 0_u32;
            for y in 0..16_usize {
                let row = (8 + y) * width;
                for x in 0..16_usize {
                    dev = dev.saturating_add(u32::from(curr[row + 12 + x]).abs_diff(mean));
                }
            }
            dev
        };

        assert_eq!(dev_block(&curr, width, 12, 8, 16, 16), scalar);
    }

    #[test]
    fn meanalysis_handles_three_macroblock_dimensions() {
        let width: usize = 33;
        let height: usize = 33;
        let prev = vec![0_u8; width * height];
        let curr = vec![255_u8; width * height];

        let analysis = meanalysis_xvid_like(&prev, &curr, width, height, 4, 2000);
        assert!(analysis.total_blocks > 0);
        assert!(analysis.intra_blocks <= analysis.total_blocks);
        assert!(analysis.score > 0.0 || analysis.intra_blocks > 0);
    }

    #[test]
    fn meanalysis_rejects_short_input_slices() {
        let analysis = meanalysis_xvid_like(&[0_u8; 16], &[0_u8; 16], 64, 64, 4, 2000);
        assert_eq!(analysis.total_blocks, 0);
        assert_eq!(analysis.intra_blocks, 0);
        assert_eq!(analysis.mc_sad, 0);
        assert!(analysis.score.abs() < f64::EPSILON);
    }

    #[test]
    fn meanalysis_cached_plan_matches_fresh_plan() {
        let width: usize = 160;
        let height: usize = 96;
        let prev = make_frame(width * height, 7);
        let curr = make_frame(width * height, 19);

        let cached = meanalysis_xvid_like(&prev, &curr, width, height, 4, 2000);
        let fresh = meanalysis_xvid_like_fresh_plan(&prev, &curr, width, height, 4, 2000);

        assert_eq!(cached.total_blocks, fresh.total_blocks);
        assert_eq!(cached.intra_blocks, fresh.intra_blocks);
        assert_eq!(cached.mc_sad, fresh.mc_sad);
        assert!((cached.score - fresh.score).abs() < f64::EPSILON);
    }
}
