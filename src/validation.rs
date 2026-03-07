use crate::{SCuiseiError, SCuiseiResult};

pub fn validate_finite_nonnegative(name: &str, value: f64) -> SCuiseiResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(SCuiseiError::config(format!(
            "{name} must be a finite number greater than or equal to 0"
        )));
    }
    Ok(())
}

pub fn validate_nonnegative_i32(name: &str, value: i32) -> SCuiseiResult<()> {
    if value < 0 {
        return Err(SCuiseiError::config(format!(
            "{name} must be greater than or equal to 0"
        )));
    }
    Ok(())
}

pub fn validate_positive_usize(name: &str, value: usize) -> SCuiseiResult<()> {
    if value == 0 {
        return Err(SCuiseiError::config(format!(
            "{name} must be greater than 0"
        )));
    }
    Ok(())
}

pub fn validate_unit_interval(name: &str, value: f64) -> SCuiseiResult<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(SCuiseiError::config(format!(
            "{name} must be a finite number between 0 and 1"
        )));
    }
    Ok(())
}
