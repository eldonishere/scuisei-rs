#![deny(clippy::pedantic)]

use clap::Parser;
use std::fs::File;
use std::io::{self, BufWriter, Write as _};

fn main() -> scuisei_rs::SCuiseiResult<()> {
    let cli = scuisei_rs::cli::Cli::parse();

    let writer: Box<dyn io::Write> = match cli.output.as_ref() {
        Some(path) => Box::new(File::create(path).map_err(|error| {
            scuisei_rs::SCuiseiError::io(
                &format!("failed to create output file: {}", path.display()),
                &error,
            )
        })?),
        None => Box::new(io::stdout().lock()),
    };
    let mut writer = BufWriter::new(writer);

    scuisei_rs::run_cli(&cli, &mut writer)?;
    writer
        .flush()
        .map_err(|error| scuisei_rs::SCuiseiError::io("failed to flush output", &error))?;
    Ok(())
}
