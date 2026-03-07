use std::fmt::Display;
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
    #[must_use]
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }

    #[must_use]
    pub fn io_message(message: impl Into<String>) -> Self {
        Self::Io(message.into())
    }

    #[must_use]
    pub fn decode(message: impl Into<String>) -> Self {
        Self::Decode(message.into())
    }

    #[must_use]
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }

    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

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
        Self::io_with(context, error)
    }

    #[must_use]
    pub fn config_with(context: &str, error: &impl Display) -> Self {
        Self::config(Self::contextual_message(context, error))
    }

    #[must_use]
    pub fn io_with(context: &str, error: &impl Display) -> Self {
        Self::io_message(Self::contextual_message(context, error))
    }

    #[must_use]
    pub fn decode_with(context: &str, error: &impl Display) -> Self {
        Self::decode(Self::contextual_message(context, error))
    }

    #[must_use]
    pub fn unsupported_with(context: &str, error: &impl Display) -> Self {
        Self::unsupported(Self::contextual_message(context, error))
    }

    #[must_use]
    pub fn internal_with(context: &str, error: &impl Display) -> Self {
        Self::internal(Self::contextual_message(context, error))
    }

    fn contextual_message(context: &str, error: &impl Display) -> String {
        format!("{context}: {error}")
    }
}

impl From<anyhow::Error> for SCuiseiError {
    fn from(value: anyhow::Error) -> Self {
        Self::internal(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::SCuiseiError;

    #[test]
    fn from_anyhow_defaults_to_internal_error() {
        let error = SCuiseiError::from(anyhow::anyhow!("opaque failure"));
        assert!(matches!(error, SCuiseiError::Internal(_)));
    }

    #[test]
    fn io_helper_preserves_io_category() {
        let error = SCuiseiError::io_message("failed to open input: foo.mp4");
        assert!(matches!(error, SCuiseiError::Io(_)));
        assert_eq!(error.category(), "io");
    }

    #[test]
    fn config_helper_preserves_config_category() {
        let error = SCuiseiError::config("unknown --hwdec: nope");
        assert!(matches!(error, SCuiseiError::Config(_)));
        assert_eq!(error.category(), "config");
    }
}
