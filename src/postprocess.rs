#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostprocessConfig {
    pub hist_blend_threshold: f64,
    pub structure_hist_max: f64,
    pub structure_grid_hist_min: f64,
    pub frames_score_min: f64,
    pub candidate_peak_radius: usize,
    pub cluster_gap_frames: usize,
    pub min_postprocess_frames: usize,
    pub dense_recovery_min_gap_frames: usize,
    pub dense_recovery_min_candidates: usize,
    pub dense_recovery_margin_frames: usize,
    pub dense_recovery_blend_min: f64,
    pub dense_recovery_score_min: f64,
    pub dense_recovery_target_span_frames: usize,
    pub temporal_burst_window_frames: usize,
    pub temporal_burst_activity_blend_min: f64,
    pub temporal_burst_min_count: usize,
    pub temporal_burst_raw_hist_max: f64,
    pub temporal_burst_raw_score_max: f64,
    pub temporal_burst_nonraw_score_max: f64,
    pub temporal_burst_nonraw_blend_max: f64,
    pub temporal_burst_dense_min_count: usize,
    pub temporal_burst_dense_nonraw_score_max: f64,
    pub temporal_burst_cut_fallback_hist_max: f64,
    pub temporal_burst_cut_fallback_score_min: f64,
    pub grid_hist_median_weight: f64,
}

impl Default for PostprocessConfig {
    fn default() -> Self {
        Self {
            hist_blend_threshold: 0.10,
            structure_hist_max: 0.05,
            structure_grid_hist_min: 0.22,
            frames_score_min: 70.0,
            candidate_peak_radius: 1,
            cluster_gap_frames: 12,
            min_postprocess_frames: 24,
            dense_recovery_min_gap_frames: 24,
            dense_recovery_min_candidates: 4,
            dense_recovery_margin_frames: 12,
            dense_recovery_blend_min: 0.18,
            dense_recovery_score_min: 80.0,
            dense_recovery_target_span_frames: 26,
            temporal_burst_window_frames: 12,
            temporal_burst_activity_blend_min: 0.08,
            temporal_burst_min_count: 6,
            temporal_burst_raw_hist_max: 0.30,
            temporal_burst_raw_score_max: 120.0,
            temporal_burst_nonraw_score_max: 90.0,
            temporal_burst_nonraw_blend_max: 0.20,
            temporal_burst_dense_min_count: 20,
            temporal_burst_dense_nonraw_score_max: 90.0,
            temporal_burst_cut_fallback_hist_max: 0.20,
            temporal_burst_cut_fallback_score_min: 120.0,
            grid_hist_median_weight: 0.1,
        }
    }
}

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

fn initial_keyframes(stats: &[FrameCutStats], config: &PostprocessConfig) -> Vec<usize> {
    let mut baseline = vec![0_usize];
    baseline.extend(
        stats
            .iter()
            .filter(|s| s.is_cut || s.hist_distance >= config.hist_blend_threshold)
            .map(|s| s.frame_index),
    );
    baseline
}

fn build_blended_score(stats: &[FrameCutStats], config: &PostprocessConfig) -> Vec<f64> {
    stats
        .iter()
        .map(|s| s.hist_distance + (config.grid_hist_median_weight * s.grid_hist_median))
        .collect()
}

fn burst_activity_prefix(blended: &[f64], config: &PostprocessConfig) -> Vec<usize> {
    let mut prefix = vec![0; blended.len() + 1];
    for (index, value) in blended.iter().enumerate() {
        prefix[index + 1] =
            prefix[index] + usize::from(*value >= config.temporal_burst_activity_blend_min);
    }
    prefix
}

fn burst_count_at_index(
    burst_prefix: &[usize],
    index: usize,
    len: usize,
    config: &PostprocessConfig,
) -> usize {
    if len == 0 || index >= len {
        return 0;
    }

    let burst_start = index.saturating_sub(config.temporal_burst_window_frames);
    let burst_end = index
        .saturating_add(config.temporal_burst_window_frames)
        .min(len.saturating_sub(1));
    burst_prefix[burst_end + 1] - burst_prefix[burst_start]
}

fn is_cut_fallback_candidate(
    stat: &FrameCutStats,
    in_activity_burst: bool,
    config: &PostprocessConfig,
) -> bool {
    stat.is_cut
        && in_activity_burst
        && stat.score >= config.temporal_burst_cut_fallback_score_min
        && stat.hist_distance <= config.temporal_burst_cut_fallback_hist_max
}

fn collect_candidates(
    stats: &[FrameCutStats],
    blended: &[f64],
    grid_hist: &[f64],
    burst_prefix: &[usize],
    config: &PostprocessConfig,
) -> Vec<usize> {
    let mut candidates: Vec<usize> = Vec::new();
    for (index, stat) in stats.iter().enumerate() {
        let burst_count = burst_count_at_index(burst_prefix, index, blended.len(), config);
        let in_activity_burst = burst_count >= config.temporal_burst_min_count;

        let suppress_weak_cut = stat.is_cut
            && in_activity_burst
            && stat.hist_distance < config.temporal_burst_raw_hist_max
            && stat.score < config.temporal_burst_raw_score_max;
        let suppress_weak_noncut = !stat.is_cut
            && in_activity_burst
            && stat.score < config.temporal_burst_nonraw_score_max
            && blended[index] < config.temporal_burst_nonraw_blend_max;
        let suppress_dense_noncut = !stat.is_cut
            && burst_count >= config.temporal_burst_dense_min_count
            && stat.score < config.temporal_burst_dense_nonraw_score_max;
        let suppress_followup_cut =
            stat.is_cut && !in_activity_burst && index > 0 && stats[index - 1].is_cut;
        if suppress_weak_cut
            || suppress_weak_noncut
            || suppress_dense_noncut
            || suppress_followup_cut
        {
            continue;
        }

        let primary = blended[index] >= config.hist_blend_threshold
            && stat.score >= config.frames_score_min
            && is_local_peak(blended, index, config.candidate_peak_radius);
        let structural = stat.hist_distance <= config.structure_hist_max
            && stat.grid_hist_distance >= config.structure_grid_hist_min
            && is_local_peak(grid_hist, index, config.candidate_peak_radius);
        let cut_fallback = is_cut_fallback_candidate(stat, in_activity_burst, config);
        if primary || structural || cut_fallback {
            candidates.push(index);
        }
    }
    candidates
}

fn best_candidate_index(
    candidates: &[usize],
    stats: &[FrameCutStats],
    blended: &[f64],
    burst_prefix: &[usize],
    config: &PostprocessConfig,
) -> Option<usize> {
    if let Some(fallback) = candidates
        .iter()
        .copied()
        .filter(|idx| {
            let burst_count = burst_count_at_index(burst_prefix, *idx, blended.len(), config);
            let in_activity_burst = burst_count >= config.temporal_burst_min_count;
            is_cut_fallback_candidate(&stats[*idx], in_activity_burst, config)
        })
        .min_by_key(|idx| stats[*idx].frame_index)
    {
        return Some(fallback);
    }

    candidates
        .iter()
        .copied()
        .max_by(|left, right| blended[*left].total_cmp(&blended[*right]))
}

fn select_cluster_peaks(
    stats: &[FrameCutStats],
    candidate_indices: &[usize],
    blended: &[f64],
    burst_prefix: &[usize],
    config: &PostprocessConfig,
) -> Vec<usize> {
    let mut refined: Vec<usize> = vec![0];
    let mut cluster: Vec<usize> = Vec::new();
    let mut previous_frame: Option<usize> = None;

    for candidate in candidate_indices.iter().copied() {
        let frame = stats[candidate].frame_index;
        if previous_frame.is_some_and(|prev| frame.saturating_sub(prev) < config.cluster_gap_frames)
        {
            cluster.push(candidate);
            previous_frame = Some(frame);
            continue;
        }

        if let Some(best) = best_candidate_index(&cluster, stats, blended, burst_prefix, config) {
            refined.push(stats[best].frame_index);
        }
        cluster.clear();
        cluster.push(candidate);
        previous_frame = Some(frame);
    }

    if let Some(best) = best_candidate_index(&cluster, stats, blended, burst_prefix, config) {
        refined.push(stats[best].frame_index);
    }

    refined
}

fn recover_dense_candidates(
    stats: &[FrameCutStats],
    candidate_indices: &[usize],
    blended: &[f64],
    burst_prefix: &[usize],
    refined: &[usize],
    config: &PostprocessConfig,
) -> Vec<usize> {
    let mut recovered: Vec<usize> = Vec::new();

    for window in refined.windows(2) {
        let left = window[0];
        let right = window[1];
        let gap = right.saturating_sub(left);
        if gap < config.dense_recovery_min_gap_frames {
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
        if interval_candidates.len() < config.dense_recovery_min_candidates {
            continue;
        }

        let filtered: Vec<usize> = interval_candidates
            .into_iter()
            .filter(|idx| {
                let frame = stats[*idx].frame_index;
                frame.saturating_sub(left) >= config.dense_recovery_margin_frames
                    && right.saturating_sub(frame) >= config.dense_recovery_margin_frames
                    && blended[*idx] >= config.dense_recovery_blend_min
                    && stats[*idx].score >= config.dense_recovery_score_min
            })
            .collect();
        if filtered.is_empty() {
            continue;
        }

        let max_additions = (gap / config.dense_recovery_target_span_frames).max(1);
        if max_additions == 1 {
            if let Some(best) =
                best_candidate_index(&filtered, stats, blended, burst_prefix, config)
            {
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
            let slot_candidates: Vec<usize> = filtered
                .iter()
                .copied()
                .filter(|idx| {
                    let frame = stats[*idx].frame_index;
                    frame >= start && frame <= end && !chosen_mask[*idx]
                })
                .collect();
            if let Some(best) =
                best_candidate_index(&slot_candidates, stats, blended, burst_prefix, config)
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
pub fn refine_frame_keyframes_with_config(
    stats: &[FrameCutStats],
    config: &PostprocessConfig,
) -> Vec<usize> {
    let baseline_keyframes = initial_keyframes(stats, config);
    if stats.len() < config.min_postprocess_frames {
        return baseline_keyframes;
    }

    let blended = build_blended_score(stats, config);
    let grid_hist: Vec<f64> = stats.iter().map(|s| s.grid_hist_distance).collect();
    let burst_prefix = burst_activity_prefix(&blended, config);
    let candidate_indices = collect_candidates(stats, &blended, &grid_hist, &burst_prefix, config);
    if candidate_indices.is_empty() {
        return baseline_keyframes;
    }

    let mut refined =
        select_cluster_peaks(stats, &candidate_indices, &blended, &burst_prefix, config);
    let recovered = recover_dense_candidates(
        stats,
        &candidate_indices,
        &blended,
        &burst_prefix,
        &refined,
        config,
    );
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

            let keyframes =
                refine_frame_keyframes_with_config(&stats, &PostprocessConfig::default());
            prop_assert!(!keyframes.is_empty());
            prop_assert_eq!(keyframes[0], 0);
            prop_assert!(keyframes.windows(2).all(|window| window[0] < window[1]));
            let max_frame = stats.last().map_or(0, |item| item.frame_index);
            prop_assert!(keyframes.iter().all(|frame| *frame <= max_frame));
        }
    }
}
