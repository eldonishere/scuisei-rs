use std::io::{Result, Write};

pub struct ScxvidPassFormatter<'a, W: Write> {
    writer: &'a mut W,
    wrote_header: bool,
}

impl<'a, W: Write> ScxvidPassFormatter<'a, W> {
    pub fn new(writer: &'a mut W) -> Result<Self> {
        let mut this = Self {
            writer,
            wrote_header: false,
        };
        this.write_header()?;
        Ok(this)
    }

    fn write_header(&mut self) -> Result<()> {
        if self.wrote_header {
            return Ok(());
        }
        writeln!(
            self.writer,
            "# XviD 2pass stat file (core version scuisei-rs {})",
            env!("CARGO_PKG_VERSION")
        )?;
        self.writer
            .write_all(b"# Please do not modify this file\n\n")?;
        self.wrote_header = true;
        Ok(())
    }

    pub fn write_first_frame(&mut self) -> Result<()> {
        self.write_frame(true)
    }

    pub fn write_frame(&mut self, is_cut: bool) -> Result<()> {
        let ch = if is_cut { b'i' } else { b'p' };
        self.writer.write_all(&[ch, b'\n'])?;
        Ok(())
    }
}
