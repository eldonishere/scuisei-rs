use crate::{SCuiseiError, SCuiseiResult};
use anyhow::{Context as _, Result as AnyResult};
use ffmpeg::format::Pixel;
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context as ScalingContext, flag::Flags as ScalingFlags};
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use std::ffi::c_char;
use std::path::Path;

mod hwaccel {
    use super::{AnyResult, SCuiseiError, SCuiseiResult, Video, ffmpeg};
    use std::ptr;

    struct HwCtx {
        pix_fmt: ffmpeg::ffi::AVPixelFormat,
    }

    /// Hardware-decoder state pinned beside the `FFmpeg` decoder context.
    ///
    /// Safety invariants:
    /// - `selector` must outlive the `AVCodecContext` `opaque` pointer that references it.
    /// - `get_hw_format` may only read `selector`; it must not mutate or free FFmpeg-owned memory.
    /// - `transfer_to_cpu` always unrefs `transferred` before asking `FFmpeg` to write into it.
    pub(super) struct Binding {
        selector: Box<HwCtx>,
    }

    impl Binding {
        pub(super) fn attach(
            ctx: &mut ffmpeg::codec::context::Context,
            codec: ffmpeg::Codec,
            device_type: ffmpeg::ffi::AVHWDeviceType,
        ) -> AnyResult<Self> {
            unsafe {
                let codec = codec.as_ptr();
                let avctx = ctx.as_mut_ptr();

                for i in 0_i32.. {
                    let cfg = ffmpeg::ffi::avcodec_get_hw_config(codec, i);
                    if cfg.is_null() {
                        break;
                    }
                    if ((*cfg).methods
                        & (ffmpeg::ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32))
                        == 0
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

                    let selector = Box::new(HwCtx {
                        pix_fmt: (*cfg).pix_fmt,
                    });
                    (*avctx).opaque = std::ptr::from_ref::<HwCtx>(selector.as_ref()) as *mut _;
                    return Ok(Self { selector });
                }
            }

            anyhow::bail!("failed to enable hardware decoding")
        }

        #[must_use]
        pub(super) fn is_hardware_frame(frame: &Video) -> bool {
            unsafe { !(*frame.as_ptr()).hw_frames_ctx.is_null() }
        }

        pub(super) fn transfer_to_cpu(
            decoded: &Video,
            transferred: &mut Video,
        ) -> SCuiseiResult<()> {
            unsafe {
                ffmpeg::ffi::av_frame_unref(transferred.as_mut_ptr());
                let rc = ffmpeg::ffi::av_hwframe_transfer_data(
                    transferred.as_mut_ptr(),
                    decoded.as_ptr(),
                    0,
                );
                if rc < 0 {
                    return Err(SCuiseiError::decode("failed to transfer hardware frame"));
                }
            }

            Ok(())
        }

        #[must_use]
        pub(super) fn pixel_format(&self) -> ffmpeg::ffi::AVPixelFormat {
            self.selector.pix_fmt
        }
    }

    unsafe extern "C" fn get_hw_format(
        s: *mut ffmpeg::ffi::AVCodecContext,
        fmt: *const ffmpeg::ffi::AVPixelFormat,
    ) -> ffmpeg::ffi::AVPixelFormat {
        let selector = unsafe { (*s).opaque as *const HwCtx };
        if selector.is_null() {
            return ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE;
        }
        let desired = unsafe { (*selector).pix_fmt };

        let mut p = fmt;
        loop {
            let value = unsafe { *p };
            if value == desired {
                return value;
            }
            if value == ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE {
                return value;
            }
            p = unsafe { p.add(1) };
        }
    }
}

fn configure_decoder_threading(ctx: &mut ffmpeg::codec::context::Context) {
    // Enable FFmpeg's built-in frame threading for software decode.
    // count=0 lets FFmpeg auto-pick an appropriate worker count.
    let mut config = ffmpeg::codec::threading::Config::kind(ffmpeg::codec::threading::Type::Frame);
    config.count = 0;
    ctx.set_threading(config);
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
    hw: Option<hwaccel::Binding>,
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
    pub fn open(path: &Path, hwdev: Option<&str>) -> SCuiseiResult<Self> {
        let ictx = ffmpeg::format::input(path).map_err(|error| {
            SCuiseiError::io_message(format!("failed to open input: {}: {error}", path.display()))
        })?;

        let input = ictx
            .streams()
            .best(Type::Video)
            .ok_or_else(|| SCuiseiError::unsupported("no video stream found"))?;
        let video_stream_index = input.index();

        let mut context_decoder = ffmpeg::codec::context::Context::from_parameters(
            input.parameters(),
        )
        .map_err(|error| SCuiseiError::decode_with("failed to create decoder context", &error))?;
        let codec = ffmpeg::codec::decoder::find(context_decoder.id())
            .ok_or_else(|| SCuiseiError::unsupported("failed to find video decoder"))?;
        configure_decoder_threading(&mut context_decoder);
        let hw = hwdev
            .map(|name| -> SCuiseiResult<hwaccel::Binding> {
                let ty = parse_hw_device_type(name)?;
                let binding =
                    hwaccel::Binding::attach(&mut context_decoder, codec, ty).map_err(|error| {
                        SCuiseiError::unsupported_with(
                            &format!("failed to enable hardware decoding (--hwdec {name})"),
                            &error,
                        )
                    })?;
                debug_assert_ne!(
                    binding.pixel_format(),
                    ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE
                );
                Ok(binding)
            })
            .transpose()?;
        let video = context_decoder
            .decoder()
            .open_as(codec)
            .map_err(|error| SCuiseiError::decode_with("failed to open video decoder", &error))?
            .video()
            .map_err(|error| SCuiseiError::decode_with("failed to open video decoder", &error))?;

        Ok(Self {
            ictx,
            video_stream_index,
            video,
            scaler: None,
            scaler_input: None,
            hw,
        })
    }

    pub fn decode_luma_frames<F>(&mut self, mut on_frame: F) -> SCuiseiResult<()>
    where
        F: FnMut(&mut Vec<u8>, FrameInfo) -> SCuiseiResult<()>,
    {
        let mut buffers = DecodeFrameBuffers::new();

        for (stream, packet) in self.ictx.packets() {
            if stream.index() != self.video_stream_index {
                continue;
            }
            self.video.send_packet(&packet).map_err(|error| {
                SCuiseiError::decode_with("failed to send packet to decoder", &error)
            })?;
            receive_and_process_frames(
                &mut self.video,
                &mut self.scaler,
                &mut self.scaler_input,
                &mut buffers,
                self.hw.is_some(),
                &mut on_frame,
            )?;
        }

        self.video.send_eof().map_err(|error| {
            SCuiseiError::decode_with("failed to signal EOF to decoder", &error)
        })?;
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

fn parse_hw_device_type(name: &str) -> SCuiseiResult<ffmpeg::ffi::AVHWDeviceType> {
    let raw = name;
    let name = std::ffi::CString::new(raw)
        .map_err(|error| SCuiseiError::config_with("invalid --hwdec value", &error))?;
    let ty = unsafe { ffmpeg::ffi::av_hwdevice_find_type_by_name(name.as_ptr().cast::<c_char>()) };
    if ty == ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
        return Err(SCuiseiError::config(format!("unknown --hwdec: {raw}")));
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
) -> SCuiseiResult<()>
where
    F: FnMut(&mut Vec<u8>, FrameInfo) -> SCuiseiResult<()>,
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
            Err(error) => {
                return Err(SCuiseiError::decode_with("failed to receive frame", &error));
            }
        }

        let is_hw = hwaccel::Binding::is_hardware_frame(&buffers.decoded);
        validate_hw_frame_requirement(require_hw, is_hw)?;
        let frame: &Video = if is_hw {
            hwaccel::Binding::transfer_to_cpu(&buffers.decoded, &mut buffers.transferred)?;
            &buffers.transferred
        } else {
            &buffers.decoded
        };

        let info = FrameInfo {
            width: frame.width() as usize,
            height: frame.height() as usize,
        };

        if is_direct_luma_format(frame.format()) {
            copy_luma_plane(frame, &mut buffers.luma).map_err(|error| decode_anyhow(&error))?;
        } else {
            scale_to_gray8(scaler, scaler_input, frame, &mut buffers.gray)
                .map_err(|error| decode_anyhow(&error))?;
            copy_luma_plane(&buffers.gray, &mut buffers.luma)
                .map_err(|error| decode_anyhow(&error))?;
        }

        on_frame(&mut buffers.luma, info)?;
    }

    Ok(())
}

fn decode_anyhow(error: &anyhow::Error) -> SCuiseiError {
    SCuiseiError::decode(error.to_string())
}

fn validate_hw_frame_requirement(require_hw: bool, is_hw: bool) -> SCuiseiResult<()> {
    if require_hw && !is_hw {
        return Err(SCuiseiError::unsupported(
            "--hwdec set but decoder produced software frames",
        ));
    }

    Ok(())
}

fn scale_to_gray8(
    scaler: &mut Option<ScalingContext>,
    scaler_input: &mut Option<ScalerInput>,
    decoded: &Video,
    gray: &mut Video,
) -> AnyResult<()> {
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
) -> AnyResult<&'a mut ScalingContext> {
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

fn create_gray8_scaler(input: ScalerInput) -> AnyResult<ScalingContext> {
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

fn copy_luma_plane(frame: &Video, out: &mut Vec<u8>) -> AnyResult<()> {
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

#[cfg(test)]
mod tests {
    use super::{parse_hw_device_type, validate_hw_frame_requirement};
    use crate::SCuiseiError;

    #[test]
    fn invalid_hwdec_name_is_config_error() {
        let error = parse_hw_device_type("not-a-device").expect_err("invalid device should fail");
        assert!(matches!(error, SCuiseiError::Config(_)));
    }

    #[test]
    fn nul_in_hwdec_name_is_config_error() {
        let error = parse_hw_device_type("bad\0device").expect_err("nul device should fail");
        assert!(matches!(error, SCuiseiError::Config(_)));
    }

    #[test]
    fn requiring_hw_frames_rejects_software_output() {
        let error = validate_hw_frame_requirement(true, false)
            .expect_err("software frames should be rejected when --hwdec is required");
        assert!(matches!(error, SCuiseiError::Unsupported(_)));
    }

    #[test]
    fn software_frames_are_allowed_without_hw_requirement() {
        assert!(validate_hw_frame_requirement(false, false).is_ok());
        assert!(validate_hw_frame_requirement(true, true).is_ok());
    }
}
