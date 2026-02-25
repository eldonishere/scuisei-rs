use thiserror::Error;

/// Public error type for `scuisei-rs` library and CLI operations.
#[derive(Debug, Error)]
pub enum SCuiseiError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("input/output error: {0}")]
    Io(String),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("internal error: {0}")]
    Internal(String),
}

pub type SCuiseiResult<T> = std::result::Result<T, SCuiseiError>;

impl SCuiseiError {
    /// Return a stable category label for this error variant.
    #[must_use]
    pub fn category(&self) -> &'static str {
        match self {
            Self::Config(_) => "config",
            Self::Io(_) => "io",
            Self::Decode(_) => "decode",
            Self::Unsupported(_) => "unsupported",
            Self::Internal(_) => "internal",
        }
    }

    #[must_use]
    pub fn io(context: &str, error: &std::io::Error) -> Self {
        Self::Io(format!("{context}: {error}"))
    }

    fn classify(message: String) -> Self {
        let lowered = message.to_ascii_lowercase();

        if lowered.contains("unknown --hwdec") || lowered.contains("invalid --hwdec") {
            return Self::Config(message);
        }

        if lowered.contains("failed to open input")
            || lowered.contains("failed to create output file")
            || lowered.contains("failed to flush output")
        {
            return Self::Io(message);
        }

        if lowered.contains("hardware decoding")
            || lowered.contains("no video stream found")
            || lowered.contains("unsupported")
        {
            return Self::Unsupported(message);
        }

        if lowered.contains("ffmpeg")
            || lowered.contains("decoder")
            || lowered.contains("frame")
            || lowered.contains("luma")
            || lowered.contains("scaler")
            || lowered.contains("video")
        {
            return Self::Decode(message);
        }

        Self::Internal(message)
    }
}

impl From<anyhow::Error> for SCuiseiError {
    fn from(value: anyhow::Error) -> Self {
        Self::classify(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::SCuiseiError;

    #[test]
    fn classifies_hwdec_as_config_error() {
        let error = SCuiseiError::from(anyhow::anyhow!("unknown --hwdec: nope"));
        assert!(matches!(error, SCuiseiError::Config(_)));
    }

    #[test]
    fn classifies_input_open_as_io_error() {
        let error = SCuiseiError::from(anyhow::anyhow!("failed to open input: foo.mp4"));
        assert!(matches!(error, SCuiseiError::Io(_)));
        assert_eq!(error.category(), "io");
    }
}
