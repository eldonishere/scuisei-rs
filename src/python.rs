use crate::{AnalyzeOptions, SCuiseiError, analyze_video, write_pass_log};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn to_py_err(error: &SCuiseiError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pyfunction]
#[pyo3(signature = (input, native_res=false, hwdec=None))]
fn detect_frames(input: String, native_res: bool, hwdec: Option<String>) -> PyResult<Vec<usize>> {
    let mut options = AnalyzeOptions::defaults_for_input(input);
    options.native_res = native_res;
    options.hwdec = hwdec;
    analyze_video(&options)
        .map(|result| result.keyframes)
        .map_err(|error| to_py_err(&error))
}

#[pyfunction]
#[pyo3(signature = (input, native_res=false, hwdec=None))]
fn detect_pass(input: String, native_res: bool, hwdec: Option<String>) -> PyResult<String> {
    let mut options = AnalyzeOptions::defaults_for_input(input);
    options.native_res = native_res;
    options.hwdec = hwdec;

    let result = analyze_video(&options).map_err(|error| to_py_err(&error))?;
    let mut pass_bytes: Vec<u8> = Vec::new();
    write_pass_log(&mut pass_bytes, &result.pass_decisions).map_err(|error| to_py_err(&error))?;
    String::from_utf8(pass_bytes).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pymodule]
fn scuisei_rs(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(detect_frames, module)?)?;
    module.add_function(wrap_pyfunction!(detect_pass, module)?)?;
    Ok(())
}
