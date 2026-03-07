use crate::{AnalyzeOptions, PostprocessConfig, SCuiseiError, analyze_video, write_pass_log};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn to_py_err(error: &SCuiseiError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pyclass(module = "scuisei_rs", name = "PostprocessConfig")]
#[derive(Clone, Debug)]
struct PyPostprocessConfig {
    #[pyo3(get, set)]
    hist_blend_threshold: f64,
    #[pyo3(get, set)]
    structure_hist_max: f64,
    #[pyo3(get, set)]
    structure_grid_hist_min: f64,
    #[pyo3(get, set)]
    frames_score_min: f64,
    #[pyo3(get, set)]
    candidate_peak_radius: usize,
    #[pyo3(get, set)]
    cluster_gap_frames: usize,
    #[pyo3(get, set)]
    min_postprocess_frames: usize,
    #[pyo3(get, set)]
    dense_recovery_min_gap_frames: usize,
    #[pyo3(get, set)]
    dense_recovery_min_candidates: usize,
    #[pyo3(get, set)]
    dense_recovery_margin_frames: usize,
    #[pyo3(get, set)]
    dense_recovery_blend_min: f64,
    #[pyo3(get, set)]
    dense_recovery_score_min: f64,
    #[pyo3(get, set)]
    dense_recovery_target_span_frames: usize,
    #[pyo3(get, set)]
    temporal_burst_window_frames: usize,
    #[pyo3(get, set)]
    temporal_burst_activity_blend_min: f64,
    #[pyo3(get, set)]
    temporal_burst_min_count: usize,
    #[pyo3(get, set)]
    temporal_burst_raw_hist_max: f64,
    #[pyo3(get, set)]
    temporal_burst_raw_score_max: f64,
    #[pyo3(get, set)]
    temporal_burst_nonraw_score_max: f64,
    #[pyo3(get, set)]
    temporal_burst_nonraw_blend_max: f64,
    #[pyo3(get, set)]
    temporal_burst_dense_min_count: usize,
    #[pyo3(get, set)]
    temporal_burst_dense_nonraw_score_max: f64,
    #[pyo3(get, set)]
    temporal_burst_cut_fallback_hist_max: f64,
    #[pyo3(get, set)]
    temporal_burst_cut_fallback_score_min: f64,
    #[pyo3(get, set)]
    grid_hist_median_weight: f64,
}

impl Default for PyPostprocessConfig {
    fn default() -> Self {
        Self::from_rust(PostprocessConfig::default())
    }
}

impl PyPostprocessConfig {
    fn from_rust(config: PostprocessConfig) -> Self {
        Self {
            hist_blend_threshold: config.hist_blend_threshold,
            structure_hist_max: config.structure_hist_max,
            structure_grid_hist_min: config.structure_grid_hist_min,
            frames_score_min: config.frames_score_min,
            candidate_peak_radius: config.candidate_peak_radius,
            cluster_gap_frames: config.cluster_gap_frames,
            min_postprocess_frames: config.min_postprocess_frames,
            dense_recovery_min_gap_frames: config.dense_recovery_min_gap_frames,
            dense_recovery_min_candidates: config.dense_recovery_min_candidates,
            dense_recovery_margin_frames: config.dense_recovery_margin_frames,
            dense_recovery_blend_min: config.dense_recovery_blend_min,
            dense_recovery_score_min: config.dense_recovery_score_min,
            dense_recovery_target_span_frames: config.dense_recovery_target_span_frames,
            temporal_burst_window_frames: config.temporal_burst_window_frames,
            temporal_burst_activity_blend_min: config.temporal_burst_activity_blend_min,
            temporal_burst_min_count: config.temporal_burst_min_count,
            temporal_burst_raw_hist_max: config.temporal_burst_raw_hist_max,
            temporal_burst_raw_score_max: config.temporal_burst_raw_score_max,
            temporal_burst_nonraw_score_max: config.temporal_burst_nonraw_score_max,
            temporal_burst_nonraw_blend_max: config.temporal_burst_nonraw_blend_max,
            temporal_burst_dense_min_count: config.temporal_burst_dense_min_count,
            temporal_burst_dense_nonraw_score_max: config.temporal_burst_dense_nonraw_score_max,
            temporal_burst_cut_fallback_hist_max: config.temporal_burst_cut_fallback_hist_max,
            temporal_burst_cut_fallback_score_min: config.temporal_burst_cut_fallback_score_min,
            grid_hist_median_weight: config.grid_hist_median_weight,
        }
    }

    fn to_rust(&self) -> PostprocessConfig {
        PostprocessConfig {
            hist_blend_threshold: self.hist_blend_threshold,
            structure_hist_max: self.structure_hist_max,
            structure_grid_hist_min: self.structure_grid_hist_min,
            frames_score_min: self.frames_score_min,
            candidate_peak_radius: self.candidate_peak_radius,
            cluster_gap_frames: self.cluster_gap_frames,
            min_postprocess_frames: self.min_postprocess_frames,
            dense_recovery_min_gap_frames: self.dense_recovery_min_gap_frames,
            dense_recovery_min_candidates: self.dense_recovery_min_candidates,
            dense_recovery_margin_frames: self.dense_recovery_margin_frames,
            dense_recovery_blend_min: self.dense_recovery_blend_min,
            dense_recovery_score_min: self.dense_recovery_score_min,
            dense_recovery_target_span_frames: self.dense_recovery_target_span_frames,
            temporal_burst_window_frames: self.temporal_burst_window_frames,
            temporal_burst_activity_blend_min: self.temporal_burst_activity_blend_min,
            temporal_burst_min_count: self.temporal_burst_min_count,
            temporal_burst_raw_hist_max: self.temporal_burst_raw_hist_max,
            temporal_burst_raw_score_max: self.temporal_burst_raw_score_max,
            temporal_burst_nonraw_score_max: self.temporal_burst_nonraw_score_max,
            temporal_burst_nonraw_blend_max: self.temporal_burst_nonraw_blend_max,
            temporal_burst_dense_min_count: self.temporal_burst_dense_min_count,
            temporal_burst_dense_nonraw_score_max: self.temporal_burst_dense_nonraw_score_max,
            temporal_burst_cut_fallback_hist_max: self.temporal_burst_cut_fallback_hist_max,
            temporal_burst_cut_fallback_score_min: self.temporal_burst_cut_fallback_score_min,
            grid_hist_median_weight: self.grid_hist_median_weight,
        }
    }
}

#[pymethods]
impl PyPostprocessConfig {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

fn apply_postprocess_config(
    py: Python<'_>,
    options: &mut AnalyzeOptions,
    postprocess: Option<Py<PyPostprocessConfig>>,
) {
    if let Some(postprocess) = postprocess {
        let postprocess = postprocess.borrow(py);
        options.postprocess_config = postprocess.to_rust();
    }
}

#[pyfunction]
#[pyo3(signature = (input, native_res=false, hwdec=None, postprocess=None))]
fn detect_frames(
    py: Python<'_>,
    input: String,
    native_res: bool,
    hwdec: Option<String>,
    postprocess: Option<Py<PyPostprocessConfig>>,
) -> PyResult<Vec<usize>> {
    let mut options = AnalyzeOptions::defaults_for_input(input);
    options.native_res = native_res;
    options.hwdec = hwdec;
    apply_postprocess_config(py, &mut options, postprocess);
    analyze_video(&options)
        .map(|result| result.keyframes)
        .map_err(|error| to_py_err(&error))
}

#[pyfunction]
#[pyo3(signature = (input, native_res=false, hwdec=None, postprocess=None))]
fn detect_pass(
    py: Python<'_>,
    input: String,
    native_res: bool,
    hwdec: Option<String>,
    postprocess: Option<Py<PyPostprocessConfig>>,
) -> PyResult<String> {
    let mut options = AnalyzeOptions::defaults_for_input(input);
    options.native_res = native_res;
    options.hwdec = hwdec;
    apply_postprocess_config(py, &mut options, postprocess);

    let result = analyze_video(&options).map_err(|error| to_py_err(&error))?;
    let mut pass_bytes: Vec<u8> = Vec::new();
    write_pass_log(&mut pass_bytes, &result.pass_decisions).map_err(|error| to_py_err(&error))?;
    String::from_utf8(pass_bytes).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pymodule]
fn scuisei_rs(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPostprocessConfig>()?;
    module.add_function(wrap_pyfunction!(detect_frames, module)?)?;
    module.add_function(wrap_pyfunction!(detect_pass, module)?)?;
    Ok(())
}
