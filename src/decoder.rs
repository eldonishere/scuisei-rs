use anyhow::{Context as _, Result};
use ffmpeg::format::Pixel;
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context as ScalingContext, flag::Flags as ScalingFlags};
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use std::ffi::c_char;
use std::path::Path;
use std::ptr;

struct HwCtx {
    pix_fmt: ffmpeg::ffi::AVPixelFormat,
}

unsafe extern "C" fn get_hw_format(
    s: *mut ffmpeg::ffi::AVCodecContext,
    fmt: *const ffmpeg::ffi::AVPixelFormat,
) -> ffmpeg::ffi::AVPixelFormat {
    let hw = unsafe { (*s).opaque as *const HwCtx };
    if hw.is_null() {
        return ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE;
    }
    let desired = unsafe { (*hw).pix_fmt };

    let mut p = fmt;
    loop {
        let v = unsafe { *p };
        if v == desired {
            return v;
        }
        if v == ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            return v;
        }
        p = unsafe { p.add(1) };
    }
}

fn enable_hwdec(
    ctx: &mut ffmpeg::codec::context::Context,
    codec: ffmpeg::Codec,
    device_type: ffmpeg::ffi::AVHWDeviceType,
) -> Result<Box<HwCtx>> {
    unsafe {
        let codec = codec.as_ptr();
        let avctx = ctx.as_mut_ptr();

        for i in 0_i32.. {
            let cfg = ffmpeg::ffi::avcodec_get_hw_config(codec, i);
            if cfg.is_null() {
                break;
            }
            if ((*cfg).methods & (ffmpeg::ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32)) == 0
            {
                continue;
            }
            if (*cfg).device_type != device_type {
                continue;
            }

            let mut device_ctx: *mut ffmpeg::ffi::AVBufferRef = ptr::null_mut();
            if ffmpeg::ffi::av_hwdevice_ctx_create(
                &raw mut device_ctx,
                (*cfg).device_type,
                ptr::null(),
                ptr::null_mut(),
                0,
            ) < 0
            {
                continue;
            }

            (*avctx).hw_device_ctx = device_ctx;
            (*avctx).get_format = Some(get_hw_format);

            let hw = Box::new(HwCtx {
                pix_fmt: (*cfg).pix_fmt,
            });
            (*avctx).opaque = std::ptr::from_ref::<HwCtx>(hw.as_ref()) as *mut _;
            return Ok(hw);
        }
    }

    anyhow::bail!("failed to enable hardware decoding")
}

#[derive(Clone, Copy, Debug)]
pub struct FrameInfo {
    pub width: usize,
    pub height: usize,
}

pub struct Decoder {
    ictx: ffmpeg::format::context::Input,
    video_stream_index: usize,
    video: ffmpeg::decoder::Video,
    scaler: Option<ScalingContext>,
    scaler_input: Option<ScalerInput>,
    hw: Option<Box<HwCtx>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScalerInput {
    format: Pixel,
    width: u32,
    height: u32,
}

struct DecodeFrameBuffers {
    decoded: Video,
    gray: Video,
    transferred: Video,
    luma: Vec<u8>,
}

impl DecodeFrameBuffers {
    fn new() -> Self {
        Self {
            decoded: Video::empty(),
            gray: Video::empty(),
            transferred: Video::empty(),
            luma: Vec::new(),
        }
    }
}

impl Decoder {
    pub fn open(path: &Path, hwdev: Option<&str>) -> Result<Self> {
        let ictx = ffmpeg::format::input(path)
            .with_context(|| format!("failed to open input: {}", path.display()))?;

        let input = ictx
            .streams()
            .best(Type::Video)
            .context("no video stream found")?;
        let video_stream_index = input.index();

        let mut context_decoder =
            ffmpeg::codec::context::Context::from_parameters(input.parameters())
                .context("failed to create decoder context")?;
        let codec = ffmpeg::codec::decoder::find(context_decoder.id())
            .context("failed to find video decoder")?;
        let hw = hwdev
            .map(|name| {
                let ty = parse_hw_device_type(name)?;
                enable_hwdec(&mut context_decoder, codec, ty)
                    .with_context(|| format!("failed to enable hardware decoding (--hwdec {name})"))
            })
            .transpose()?;
        let video = context_decoder
            .decoder()
            .open_as(codec)
            .context("failed to open video decoder")?
            .video()
            .context("failed to open video decoder")?;

        Ok(Self {
            ictx,
            video_stream_index,
            video,
            scaler: None,
            scaler_input: None,
            hw,
        })
    }

    pub fn decode_luma_frames<F>(&mut self, mut on_frame: F) -> Result<()>
    where
        F: FnMut(&mut Vec<u8>, FrameInfo) -> Result<()>,
    {
        let mut buffers = DecodeFrameBuffers::new();

        for (stream, packet) in self.ictx.packets() {
            if stream.index() != self.video_stream_index {
                continue;
            }
            self.video
                .send_packet(&packet)
                .context("failed to send packet to decoder")?;
            receive_and_process_frames(
                &mut self.video,
                &mut self.scaler,
                &mut self.scaler_input,
                &mut buffers,
                self.hw.is_some(),
                &mut on_frame,
            )?;
        }

        self.video
            .send_eof()
            .context("failed to signal EOF to decoder")?;
        receive_and_process_frames(
            &mut self.video,
            &mut self.scaler,
            &mut self.scaler_input,
            &mut buffers,
            self.hw.is_some(),
            &mut on_frame,
        )?;

        Ok(())
    }
}

fn parse_hw_device_type(name: &str) -> Result<ffmpeg::ffi::AVHWDeviceType> {
    let raw = name;
    let name = std::ffi::CString::new(raw).context("invalid --hwdec value")?;
    let ty = unsafe { ffmpeg::ffi::av_hwdevice_find_type_by_name(name.as_ptr().cast::<c_char>()) };
    if ty == ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
        anyhow::bail!("unknown --hwdec: {raw}");
    }
    Ok(ty)
}

fn receive_and_process_frames<F>(
    decoder: &mut ffmpeg::decoder::Video,
    scaler: &mut Option<ScalingContext>,
    scaler_input: &mut Option<ScalerInput>,
    buffers: &mut DecodeFrameBuffers,
    require_hw: bool,
    on_frame: &mut F,
) -> Result<()>
where
    F: FnMut(&mut Vec<u8>, FrameInfo) -> Result<()>,
{
    loop {
        match decoder.receive_frame(&mut buffers.decoded) {
            Ok(()) => {}
            Err(ffmpeg::Error::Other { errno })
                if errno == ffmpeg::util::error::EAGAIN
                    || errno == ffmpeg::util::error::EWOULDBLOCK =>
            {
                break;
            }
            Err(ffmpeg::Error::Eof) => break,
            Err(e) => return Err(anyhow::Error::new(e)).context("failed to receive frame"),
        }

        let is_hw = unsafe { !(*buffers.decoded.as_ptr()).hw_frames_ctx.is_null() };
        if require_hw && !is_hw {
            anyhow::bail!("--hwdec set but decoder produced software frames");
        }
        let frame: &Video = if is_hw {
            unsafe {
                ffmpeg::ffi::av_frame_unref(buffers.transferred.as_mut_ptr());
                let rc = ffmpeg::ffi::av_hwframe_transfer_data(
                    buffers.transferred.as_mut_ptr(),
                    buffers.decoded.as_ptr(),
                    0,
                );
                if rc < 0 {
                    anyhow::bail!("failed to transfer hardware frame");
                }
            }
            &buffers.transferred
        } else {
            &buffers.decoded
        };

        let info = FrameInfo {
            width: frame.width() as usize,
            height: frame.height() as usize,
        };

        if is_direct_luma_format(frame.format()) {
            copy_luma_plane(frame, &mut buffers.luma).context("failed to copy luma plane")?;
        } else {
            scale_to_gray8(scaler, scaler_input, frame, &mut buffers.gray)?;
            copy_luma_plane(&buffers.gray, &mut buffers.luma)
                .context("failed to copy luma plane")?;
        }

        on_frame(&mut buffers.luma, info)?;
    }

    Ok(())
}

fn scale_to_gray8(
    scaler: &mut Option<ScalingContext>,
    scaler_input: &mut Option<ScalerInput>,
    decoded: &Video,
    gray: &mut Video,
) -> Result<()> {
    let current_input = ScalerInput {
        format: decoded.format(),
        width: decoded.width(),
        height: decoded.height(),
    };

    let scaler_ref = ensure_gray8_scaler(scaler, scaler_input, gray, current_input)
        .context("scaler unexpectedly missing")?;

    match scaler_ref.run(decoded, gray) {
        Ok(()) => Ok(()),
        Err(ffmpeg::Error::InputChanged | ffmpeg::Error::OutputChanged) => {
            *gray = Video::empty();
            *scaler = Some(create_gray8_scaler(current_input).with_context(|| {
                format!(
                    "failed to recreate scaler after input change ({:?} {}x{})",
                    current_input.format, current_input.width, current_input.height
                )
            })?);
            *scaler_input = Some(current_input);

            let scaler_ref = scaler.as_mut().context("scaler unexpectedly missing")?;
            scaler_ref
                .run(decoded, gray)
                .context("failed to scale frame to GRAY8 after recreation")?;
            Ok(())
        }
        Err(e) => Err(anyhow::Error::new(e)).context("failed to scale frame to GRAY8"),
    }
}

fn ensure_gray8_scaler<'a>(
    scaler: &'a mut Option<ScalingContext>,
    scaler_input: &mut Option<ScalerInput>,
    gray: &mut Video,
    current_input: ScalerInput,
) -> Result<&'a mut ScalingContext> {
    if scaler.is_none()
        || scaler_input
            .as_ref()
            .is_none_or(|prev| *prev != current_input)
    {
        *gray = Video::empty();
        *scaler = Some(create_gray8_scaler(current_input).with_context(|| {
            format!(
                "failed to create scaler ({:?} {}x{})",
                current_input.format, current_input.width, current_input.height
            )
        })?);
        *scaler_input = Some(current_input);
    }

    scaler.as_mut().context("scaler unexpectedly missing")
}

fn create_gray8_scaler(input: ScalerInput) -> Result<ScalingContext> {
    ScalingContext::get(
        input.format,
        input.width,
        input.height,
        Pixel::GRAY8,
        input.width,
        input.height,
        ScalingFlags::FAST_BILINEAR,
    )
    .context("failed to create GRAY8 scaler")
}

fn is_direct_luma_format(format: Pixel) -> bool {
    matches!(
        format,
        Pixel::GRAY8
            | Pixel::YUV420P
            | Pixel::YUV422P
            | Pixel::YUV444P
            | Pixel::YUV410P
            | Pixel::YUV411P
            | Pixel::YUV440P
            | Pixel::YUVJ420P
            | Pixel::YUVJ422P
            | Pixel::YUVJ444P
            | Pixel::NV12
            | Pixel::NV21
    )
}

fn copy_luma_plane(frame: &Video, out: &mut Vec<u8>) -> Result<()> {
    let width = frame.width() as usize;
    let height = frame.height() as usize;
    let stride = frame.stride(0);

    let needed = width.checked_mul(height).context("frame size overflow")?;

    out.resize(needed, 0);

    if stride < width {
        anyhow::bail!("unexpected luma stride ({stride}) for width ({width})");
    }

    let data = frame.data(0);
    let src_needed = stride
        .checked_mul(height)
        .context("stride multiplication overflow")?;
    if data.len() < src_needed {
        anyhow::bail!(
            "insufficient luma data (have {}, need {src_needed})",
            data.len()
        );
    }

    if stride == width {
        let src = data
            .get(..needed)
            .context("insufficient luma data for contiguous copy")?;
        out.copy_from_slice(src);
        return Ok(());
    }

    for row in 0..height {
        let src_start = row
            .checked_mul(stride)
            .context("stride multiplication overflow")?;
        let src_end = src_start
            .checked_add(width)
            .context("stride range overflow")?;
        let dst_start = row
            .checked_mul(width)
            .context("width multiplication overflow")?;
        let dst_end = dst_start.checked_add(width).context("dst range overflow")?;

        out[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
    }

    Ok(())
}
