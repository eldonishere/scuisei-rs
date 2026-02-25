const HIST_BLEND_THRESHOLD: f64 = 0.10;
const STRUCTURE_HIST_MAX: f64 = 0.05;
const STRUCTURE_GRID_HIST_MIN: f64 = 0.22;
const FRAMES_SCORE_MIN: f64 = 70.0;
const CANDIDATE_PEAK_RADIUS: usize = 1;
const CLUSTER_GAP_FRAMES: usize = 12;
const MIN_POSTPROCESS_FRAMES: usize = 24;
const DENSE_RECOVERY_MIN_GAP_FRAMES: usize = 24;
const DENSE_RECOVERY_MIN_CANDIDATES: usize = 4;
const DENSE_RECOVERY_MARGIN_FRAMES: usize = 12;
const DENSE_RECOVERY_BLEND_MIN: f64 = 0.18;
const DENSE_RECOVERY_SCORE_MIN: f64 = 80.0;
const DENSE_RECOVERY_TARGET_SPAN_FRAMES: usize = 26;
const TEMPORAL_BURST_WINDOW_FRAMES: usize = 12;
const TEMPORAL_BURST_ACTIVITY_BLEND_MIN: f64 = 0.08;
const TEMPORAL_BURST_MIN_COUNT: usize = 6;
const TEMPORAL_BURST_RAW_HIST_MAX: f64 = 0.30;
const TEMPORAL_BURST_RAW_SCORE_MAX: f64 = 120.0;
const TEMPORAL_BURST_NONRAW_SCORE_MAX: f64 = 90.0;
const TEMPORAL_BURST_NONRAW_BLEND_MAX: f64 = 0.20;

#[derive(Clone, Copy, Debug)]
pub struct FrameCutStats {
    pub frame_index: usize,
    pub is_cut: bool,
    pub score: f64,
    pub hist_distance: f64,
    pub grid_hist_distance: f64,
    pub grid_hist_median: f64,
}

fn is_local_peak(values: &[f64], index: usize, radius: usize) -> bool {
    if values.is_empty() || index >= values.len() {
        return false;
    }

    let value = values[index];
    let start = index.saturating_sub(radius);
    let end = (index.saturating_add(radius)).min(values.len().saturating_sub(1));

    for (idx, other) in values[start..=end].iter().enumerate() {
        let abs_idx = start + idx;
        if abs_idx == index {
            continue;
        }
        if *other >= value {
            return false;
        }
    }

    true
}

fn initial_keyframes(stats: &[FrameCutStats]) -> Vec<usize> {
    let mut baseline = vec![0_usize];
    baseline.extend(
        stats
            .iter()
            .filter(|s| s.is_cut || s.hist_distance >= HIST_BLEND_THRESHOLD)
            .map(|s| s.frame_index),
    );
    baseline
}

fn build_blended_score(stats: &[FrameCutStats]) -> Vec<f64> {
    stats
        .iter()
        .map(|s| s.hist_distance + (0.1 * s.grid_hist_median))
        .collect()
}

fn burst_activity_prefix(blended: &[f64]) -> Vec<usize> {
    let mut prefix = vec![0; blended.len() + 1];
    for (index, value) in blended.iter().enumerate() {
        prefix[index + 1] =
            prefix[index] + usize::from(*value >= TEMPORAL_BURST_ACTIVITY_BLEND_MIN);
    }
    prefix
}

fn collect_candidates(
    stats: &[FrameCutStats],
    blended: &[f64],
    grid_hist: &[f64],
    burst_prefix: &[usize],
) -> Vec<usize> {
    let mut candidates: Vec<usize> = Vec::new();
    for (index, stat) in stats.iter().enumerate() {
        let burst_start = index.saturating_sub(TEMPORAL_BURST_WINDOW_FRAMES);
        let burst_end = index
            .saturating_add(TEMPORAL_BURST_WINDOW_FRAMES)
            .min(blended.len().saturating_sub(1));
        let burst_count = burst_prefix[burst_end + 1] - burst_prefix[burst_start];
        let in_activity_burst = burst_count >= TEMPORAL_BURST_MIN_COUNT;

        let suppress_weak_cut = stat.is_cut
            && in_activity_burst
            && stat.hist_distance < TEMPORAL_BURST_RAW_HIST_MAX
            && stat.score < TEMPORAL_BURST_RAW_SCORE_MAX;
        let suppress_weak_noncut = !stat.is_cut
            && in_activity_burst
            && stat.score < TEMPORAL_BURST_NONRAW_SCORE_MAX
            && blended[index] < TEMPORAL_BURST_NONRAW_BLEND_MAX;
        if suppress_weak_cut || suppress_weak_noncut {
            continue;
        }

        let primary = blended[index] >= HIST_BLEND_THRESHOLD
            && stat.score >= FRAMES_SCORE_MIN
            && is_local_peak(blended, index, CANDIDATE_PEAK_RADIUS);
        let structural = stat.hist_distance <= STRUCTURE_HIST_MAX
            && stat.grid_hist_distance >= STRUCTURE_GRID_HIST_MIN
            && is_local_peak(grid_hist, index, CANDIDATE_PEAK_RADIUS);
        if primary || structural {
            candidates.push(index);
        }
    }
    candidates
}

fn best_candidate_index(candidates: &[usize], blended: &[f64]) -> Option<usize> {
    candidates
        .iter()
        .copied()
        .max_by(|left, right| blended[*left].total_cmp(&blended[*right]))
}

fn select_cluster_peaks(
    stats: &[FrameCutStats],
    candidate_indices: &[usize],
    blended: &[f64],
) -> Vec<usize> {
    let mut refined: Vec<usize> = vec![0];
    let mut cluster: Vec<usize> = Vec::new();
    let mut previous_frame: Option<usize> = None;

    for candidate in candidate_indices.iter().copied() {
        let frame = stats[candidate].frame_index;
        if previous_frame.is_some_and(|prev| frame.saturating_sub(prev) < CLUSTER_GAP_FRAMES) {
            cluster.push(candidate);
            previous_frame = Some(frame);
            continue;
        }

        if let Some(best) = best_candidate_index(&cluster, blended) {
            refined.push(stats[best].frame_index);
        }
        cluster.clear();
        cluster.push(candidate);
        previous_frame = Some(frame);
    }

    if let Some(best) = best_candidate_index(&cluster, blended) {
        refined.push(stats[best].frame_index);
    }

    refined
}

fn recover_dense_candidates(
    stats: &[FrameCutStats],
    candidate_indices: &[usize],
    blended: &[f64],
    refined: &[usize],
) -> Vec<usize> {
    let mut recovered: Vec<usize> = Vec::new();

    for window in refined.windows(2) {
        let left = window[0];
        let right = window[1];
        let gap = right.saturating_sub(left);
        if gap < DENSE_RECOVERY_MIN_GAP_FRAMES {
            continue;
        }

        let interval_candidates: Vec<usize> = candidate_indices
            .iter()
            .copied()
            .filter(|idx| {
                let frame = stats[*idx].frame_index;
                frame > left && frame < right
            })
            .collect();
        if interval_candidates.len() < DENSE_RECOVERY_MIN_CANDIDATES {
            continue;
        }

        let filtered: Vec<usize> = interval_candidates
            .into_iter()
            .filter(|idx| {
                let frame = stats[*idx].frame_index;
                frame.saturating_sub(left) >= DENSE_RECOVERY_MARGIN_FRAMES
                    && right.saturating_sub(frame) >= DENSE_RECOVERY_MARGIN_FRAMES
                    && blended[*idx] >= DENSE_RECOVERY_BLEND_MIN
                    && stats[*idx].score >= DENSE_RECOVERY_SCORE_MIN
            })
            .collect();
        if filtered.is_empty() {
            continue;
        }

        let max_additions = (gap / DENSE_RECOVERY_TARGET_SPAN_FRAMES).max(1);
        if max_additions == 1 {
            if let Some(best) = best_candidate_index(&filtered, blended) {
                recovered.push(stats[best].frame_index);
            }
            continue;
        }

        let slots = max_additions + 1;
        let mut chosen: Vec<usize> = Vec::new();
        let mut chosen_mask = vec![false; stats.len()];
        for slot in 0..max_additions {
            let start = left + (gap.saturating_mul(slot) / slots);
            let end = left + (gap.saturating_mul(slot + 1) / slots);
            if let Some(best) = filtered
                .iter()
                .copied()
                .filter(|idx| {
                    let frame = stats[*idx].frame_index;
                    frame >= start && frame <= end && !chosen_mask[*idx]
                })
                .max_by(|a, b| blended[*a].total_cmp(&blended[*b]))
            {
                chosen.push(best);
                chosen_mask[best] = true;
            }
        }

        if chosen.len() < max_additions {
            let mut remaining = filtered.clone();
            remaining.sort_unstable_by(|a, b| blended[*b].total_cmp(&blended[*a]));
            for idx in remaining {
                if chosen.len() >= max_additions {
                    break;
                }
                if !chosen_mask[idx] {
                    chosen.push(idx);
                    chosen_mask[idx] = true;
                }
            }
        }

        recovered.extend(chosen.into_iter().map(|idx| stats[idx].frame_index));
    }

    recovered
}

#[must_use]
pub fn refine_frame_keyframes(stats: &[FrameCutStats]) -> Vec<usize> {
    let baseline_keyframes = initial_keyframes(stats);
    if stats.len() < MIN_POSTPROCESS_FRAMES {
        return baseline_keyframes;
    }

    let blended = build_blended_score(stats);
    let grid_hist: Vec<f64> = stats.iter().map(|s| s.grid_hist_distance).collect();
    let burst_prefix = burst_activity_prefix(&blended);
    let candidate_indices = collect_candidates(stats, &blended, &grid_hist, &burst_prefix);
    if candidate_indices.is_empty() {
        return baseline_keyframes;
    }

    let mut refined = select_cluster_peaks(stats, &candidate_indices, &blended);
    let recovered = recover_dense_candidates(stats, &candidate_indices, &blended, &refined);
    refined.extend(recovered);
    refined.sort_unstable();
    refined.dedup();
    refined
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn refined_keyframes_are_sorted_unique_and_bounded(
            rows in prop::collection::vec((any::<bool>(), 0.0_f64..200.0_f64, 0.0_f64..1.0_f64, 0.0_f64..1.0_f64, 0.0_f64..1.0_f64), 0..150)
        ) {
            let stats: Vec<FrameCutStats> = rows
                .into_iter()
                .enumerate()
                .map(|(idx, (is_cut, score, hist_distance, grid_hist_distance, grid_hist_median))| FrameCutStats {
                    frame_index: idx + 1,
                    is_cut,
                    score,
                    hist_distance,
                    grid_hist_distance,
                    grid_hist_median,
                })
                .collect();

            let keyframes = refine_frame_keyframes(&stats);
            prop_assert!(!keyframes.is_empty());
            prop_assert_eq!(keyframes[0], 0);
            prop_assert!(keyframes.windows(2).all(|window| window[0] < window[1]));
            let max_frame = stats.last().map_or(0, |item| item.frame_index);
            prop_assert!(keyframes.iter().all(|frame| *frame <= max_frame));
        }
    }
}
