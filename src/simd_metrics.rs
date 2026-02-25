use wide::{u8x16, u16x8, u32x8};

pub const GRID_HIST_X: usize = 4;
pub const GRID_HIST_Y: usize = 4;
pub const GRID_HIST_LEN: usize = 16 * GRID_HIST_X * GRID_HIST_Y;

#[must_use]
pub fn sad_u8(prev: &[u8], curr: &[u8]) -> u64 {
    if prev.len() != curr.len() {
        return 0;
    }

    let mut total: u64 = 0;
    let mut i: usize = 0;
    let len = prev.len();

    while i + 16 <= len {
        let a = load_u8x16(&prev[i..]);
        let b = load_u8x16(&curr[i..]);
        let diff = a.max(b) - a.min(b);

        let low = u16x8::from_u8x16_low(diff);
        let high = u16x8::from_u8x16_high(diff);

        total += sum_u16x8(low);
        total += sum_u16x8(high);

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
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        dst.clear();
        return;
    }

    let needed = dst_width.saturating_mul(dst_height);
    dst.resize(needed, 0);

    let x_indices: Vec<usize> = (0..dst_width)
        .map(|x_out| x_out.saturating_mul(src_width) / dst_width)
        .collect();
    let src_row_offsets: Vec<usize> = (0..dst_height)
        .map(|y_out| {
            let y_in = y_out.saturating_mul(src_height) / dst_height;
            y_in.saturating_mul(src_width)
        })
        .collect();

    for (y_out, row_in) in src_row_offsets.iter().copied().enumerate() {
        let row_out = y_out.saturating_mul(dst_width);
        for (x_out, x_in) in x_indices.iter().copied().enumerate() {
            dst[row_out + x_out] = src[row_in + x_in];
        }
    }
}

#[must_use]
pub fn grid_histogram_16(frame: &[u8], width: usize, height: usize) -> [u32; GRID_HIST_LEN] {
    let mut hist = [0_u32; GRID_HIST_LEN];
    if width == 0 || height == 0 || frame.len() < width.saturating_mul(height) {
        return hist;
    }

    for y in 0..height {
        let cell_y = (y * GRID_HIST_Y) / height;
        let row = y * width;
        for x in 0..width {
            let cell_x = (x * GRID_HIST_X) / width;
            let cell = (cell_y * GRID_HIST_X) + cell_x;
            let bin = (frame[row + x] >> 4) as usize;
            hist[(cell * 16) + bin] += 1;
        }
    }

    hist
}

#[must_use]
pub fn grid_histogram_median_cell_distance_16_from_hists(
    prev: &[u32; GRID_HIST_LEN],
    curr: &[u32; GRID_HIST_LEN],
) -> f64 {
    let cells = GRID_HIST_X.saturating_mul(GRID_HIST_Y);
    if cells == 0 {
        return 0.0;
    }

    let mut distances: [f64; GRID_HIST_X * GRID_HIST_Y] = [0.0; GRID_HIST_X * GRID_HIST_Y];
    let mut count: usize = 0;

    for cell in 0..cells {
        let base = cell.saturating_mul(16);
        let mut intersection: u64 = 0;
        let mut total: u64 = 0;

        for bin in 0..16 {
            let a = prev[base + bin];
            let b = curr[base + bin];
            intersection += u64::from(a.min(b));
            total += u64::from(a);
        }

        if total == 0 {
            continue;
        }

        let similarity = u64_to_f64_exact(intersection) / u64_to_f64_exact(total);
        distances[count] = (1.0 - similarity).clamp(0.0, 1.0);
        count = count.saturating_add(1);
    }

    if count == 0 {
        return 0.0;
    }

    let slice = &mut distances[..count];
    slice.sort_by(f64::total_cmp);
    let mid = count / 2;
    if count % 2 == 1 {
        slice[mid]
    } else if mid == 0 {
        slice[0]
    } else {
        (slice[mid - 1] + slice[mid]) * 0.5
    }
}

#[must_use]
pub fn grid_histogram_distance_16_from_hists(
    prev: &[u32; GRID_HIST_LEN],
    curr: &[u32; GRID_HIST_LEN],
    total_pixels: usize,
) -> f64 {
    if total_pixels == 0 {
        return 0.0;
    }

    let intersection: u64 = prev
        .iter()
        .zip(curr.iter())
        .map(|(&a, &b)| u64::from(a.min(b)))
        .sum();

    let total_u64 = u64::try_from(total_pixels).unwrap_or(u64::MAX);
    if total_u64 == 0 {
        return 0.0;
    }

    let similarity = u64_to_f64_exact(intersection) / u64_to_f64_exact(total_u64);
    (1.0 - similarity).clamp(0.0, 1.0)
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

const BLOCK_EDGE: usize = 16;
const MIN_DEV: u32 = 300;
const DEV_STATIC: u32 = 1000;
const SAD_STATIC: u32 = 1000;
const DEV_BIAS: u32 = 512;
const DEV_HALF: u32 = 3000;
const TWO_POW_32_F64: f64 = 4_294_967_296.0;

#[derive(Clone, Copy, Debug)]
struct MotionMatch {
    sad: u32,
    dx: i32,
    dy: i32,
}

fn sad16x16(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    curr_x: usize,
    curr_y: usize,
    prev_x: usize,
    prev_y: usize,
) -> u32 {
    let mut total: u32 = 0;
    for y in 0..16_usize {
        let curr_row = (curr_y + y) * width;
        let prev_row = (prev_y + y) * width;
        for x in 0..16_usize {
            total += u32::from(curr[curr_row + curr_x + x].abs_diff(prev[prev_row + prev_x + x]));
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
        let candidate_y = block_origin_y.saturating_add(dy);
        if candidate_y < 0 {
            continue;
        }
        let Ok(candidate_row) = usize::try_from(candidate_y) else {
            continue;
        };

        for dx in min_offset_x..=max_offset_x {
            let candidate_x = block_origin_x.saturating_add(dx);
            if candidate_x < 0 {
                continue;
            }
            let Ok(candidate_col) = usize::try_from(candidate_x) else {
                continue;
            };

            let sad = sad16x16(
                prev,
                curr,
                width,
                block_x,
                block_y,
                candidate_col,
                candidate_row,
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

    let mut sum: u32 = 0;
    for y in 0..block_h {
        let row = (curr_y + y).saturating_mul(width);
        for x in 0..block_w {
            sum = sum.saturating_add(u32::from(curr[row + curr_x + x]));
        }
    }

    let mean = sum / u32::try_from(area).unwrap_or(u32::MAX);
    let mut dev: u32 = 0;
    for y in 0..block_h {
        let row = (curr_y + y).saturating_mul(width);
        for x in 0..block_w {
            dev = dev.saturating_add(u32::from(curr[row + curr_x + x]).abs_diff(mean));
        }
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
    let base_col_i32 = i32::try_from(base_x).unwrap_or(i32::MAX);
    let base_row_i32 = i32::try_from(base_y).unwrap_or(i32::MAX);
    let mut result = SuperblockSearchResults {
        sad: [u32::MAX; 4],
        dx: [0_i32; 4],
        dy: [0_i32; 4],
    };

    for offset_y in search_window.min_offset_y..=search_window.max_offset_y {
        let candidate_y = base_row_i32.saturating_add(offset_y);
        if candidate_y < 0 {
            continue;
        }
        let Ok(candidate_row) = usize::try_from(candidate_y) else {
            continue;
        };

        for offset_x in search_window.min_offset_x..=search_window.max_offset_x {
            let candidate_x = base_col_i32.saturating_add(offset_x);
            if candidate_x < 0 {
                continue;
            }
            let Ok(candidate_col) = usize::try_from(candidate_x) else {
                continue;
            };

            let block_sad_0 = sad16x16(
                prev,
                curr,
                width,
                base_x,
                base_y,
                candidate_col,
                candidate_row,
            );
            if block_sad_0 < result.sad[0] {
                result.sad[0] = block_sad_0;
                result.dx[0] = offset_x;
                result.dy[0] = offset_y;
            }

            let block_sad_1 = sad16x16(
                prev,
                curr,
                width,
                base_x + BLOCK_EDGE,
                base_y,
                candidate_col + BLOCK_EDGE,
                candidate_row,
            );
            if block_sad_1 < result.sad[1] {
                result.sad[1] = block_sad_1;
                result.dx[1] = offset_x;
                result.dy[1] = offset_y;
            }

            let block_sad_2 = sad16x16(
                prev,
                curr,
                width,
                base_x,
                base_y + BLOCK_EDGE,
                candidate_col,
                candidate_row + BLOCK_EDGE,
            );
            if block_sad_2 < result.sad[2] {
                result.sad[2] = block_sad_2;
                result.dx[2] = offset_x;
                result.dy[2] = offset_y;
            }

            let block_sad_3 = sad16x16(
                prev,
                curr,
                width,
                base_x + BLOCK_EDGE,
                base_y + BLOCK_EDGE,
                candidate_col + BLOCK_EDGE,
                candidate_row + BLOCK_EDGE,
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

    let x_indices = superblock_indices(mb_width);
    let y_indices = superblock_indices(mb_height);
    if x_indices.is_empty() || y_indices.is_empty() {
        return empty_me_analysis();
    }

    let mut intra_blocks = 0_usize;
    let mut ssad = 0_u64;
    let mut complexity = 0_u64;
    let mut blocks = 10_u64;
    let max_horizontal_limit = width.saturating_sub(32);
    let max_vertical_limit = height.saturating_sub(32);

    for macroblock_y in y_indices {
        let base_y = macroblock_y.saturating_mul(BLOCK_EDGE);
        for macroblock_x in x_indices.iter().copied() {
            let base_x = macroblock_x.saturating_mul(BLOCK_EDGE);
            blocks = blocks.saturating_add(10);
            let search_window = superblock_search_window(
                base_x,
                base_y,
                search_radius,
                max_horizontal_limit,
                max_vertical_limit,
            );
            let search_results =
                search_superblock_matches(prev, curr, width, base_x, base_y, search_window);
            let (intra_delta, ssad_delta, complexity_delta) = accumulate_superblock_metrics(
                curr,
                width,
                base_x,
                base_y,
                intra_thresh_u32,
                search_results,
            );
            intra_blocks = intra_blocks.saturating_add(intra_delta);
            ssad = ssad.saturating_add(ssad_delta);
            complexity = complexity.saturating_add(complexity_delta);
        }
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

fn load_u8x16(bytes: &[u8]) -> u8x16 {
    let mut arr = [0_u8; 16];
    arr.copy_from_slice(&bytes[..16]);
    u8x16::new(arr)
}

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
}
