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
    // Enable FFmpeg's built-in frame threading for software decode. FFmpeg's
    // auto pick (count=0) caps at 16 workers; request the real core count so
    // wider machines keep scaling. FFmpeg clamps values beyond the codec
    // limit, and decoder thread count never changes decoded output.
    let mut config = ffmpeg::codec::threading::Config::kind(ffmpeg::codec::threading::Type::Frame);
    config.count = std::thread::available_parallelism().map_or(0, std::num::NonZero::get);
    ctx.set_threading(config);

    // Demote this context's log chatter (e.g. the >16-thread advisory) below
    // the default level while leaving real errors visible.
    unsafe {
        (*ctx.as_mut_ptr()).log_level_offset = 16;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FrameInfo {
    pub width: usize,
    pub height: usize,
}

/// Borrowed view of an 8-bit luma plane with a row stride in bytes.
#[derive(Clone, Copy, Debug)]
pub struct LumaView<'a> {
    pub data: &'a [u8],
    pub stride: usize,
}

pub struct Decoder {
    ictx: ffmpeg::format::context::Input,
    video_stream_index: usize,
    video: ffmpeg::decoder::Video,
    hw: Option<hwaccel::Binding>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScalerInput {
    format: Pixel,
    width: u32,
    height: u32,
}

/// Reusable frame slots for the decode loop.
struct SpareFrames {
    /// Target for `receive_frame`.
    recv: Video,
    /// Target for hardware-frame transfers to CPU memory.
    cpu: Video,
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
            hw,
        })
    }

    /// Decode the video stream and hand each frame (already transferred to CPU
    /// memory) to `on_frame` in decode order. The callback returns a frame
    /// whose buffers may be reused for subsequent decoding.
    pub fn decode_frames<F>(&mut self, mut on_frame: F) -> SCuiseiResult<()>
    where
        F: FnMut(Video) -> SCuiseiResult<Video>,
    {
        let mut spare = SpareFrames {
            recv: Video::empty(),
            cpu: Video::empty(),
        };

        for (stream, packet) in self.ictx.packets() {
            if stream.index() != self.video_stream_index {
                continue;
            }
            self.video.send_packet(&packet).map_err(|error| {
                SCuiseiError::decode_with("failed to send packet to decoder", &error)
            })?;
            receive_and_forward_frames(
                &mut self.video,
                &mut spare,
                self.hw.is_some(),
                &mut on_frame,
            )?;
        }

        self.video.send_eof().map_err(|error| {
            SCuiseiError::decode_with("failed to signal EOF to decoder", &error)
        })?;
        receive_and_forward_frames(
            &mut self.video,
            &mut spare,
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

fn receive_and_forward_frames<F>(
    decoder: &mut ffmpeg::decoder::Video,
    spare: &mut SpareFrames,
    require_hw: bool,
    on_frame: &mut F,
) -> SCuiseiResult<()>
where
    F: FnMut(Video) -> SCuiseiResult<Video>,
{
    loop {
        match decoder.receive_frame(&mut spare.recv) {
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

        let is_hw = hwaccel::Binding::is_hardware_frame(&spare.recv);
        validate_hw_frame_requirement(require_hw, is_hw)?;
        let outgoing = if is_hw {
            hwaccel::Binding::transfer_to_cpu(&spare.recv, &mut spare.cpu)?;
            std::mem::replace(&mut spare.cpu, Video::empty())
        } else {
            std::mem::replace(&mut spare.recv, Video::empty())
        };

        let recycled = on_frame(outgoing)?;
        if is_hw {
            spare.cpu = recycled;
        } else {
            spare.recv = recycled;
        }
    }

    Ok(())
}

/// Converts decoded frames into borrowable 8-bit luma planes, reusing scaler
/// state and the GRAY8 scratch frame across calls.
pub struct LumaExtractor {
    scaler: Option<ScalingContext>,
    scaler_input: Option<ScalerInput>,
    gray: Video,
}

impl LumaExtractor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scaler: None,
            scaler_input: None,
            gray: Video::empty(),
        }
    }

    /// Borrow the frame's luma plane, converting via `swscale` to GRAY8 first
    /// when the pixel format has no directly usable 8-bit luma plane.
    ///
    /// # Errors
    /// Returns an error if the scaler fails or the plane layout is invalid.
    pub fn luma_view<'a>(&'a mut self, frame: &'a Video) -> SCuiseiResult<LumaView<'a>> {
        if is_direct_luma_format(frame.format()) {
            return borrow_packed8(frame).map_err(|error| decode_anyhow(&error));
        }

        scale_to_gray8(
            &mut self.scaler,
            &mut self.scaler_input,
            frame,
            &mut self.gray,
        )
        .map_err(|error| decode_anyhow(&error))?;
        borrow_packed8(&self.gray).map_err(|error| decode_anyhow(&error))
    }
}

impl Default for LumaExtractor {
    fn default() -> Self {
        Self::new()
    }
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
            | Pixel::NV16
            | Pixel::NV24
            | Pixel::NV42
    )
}

fn borrow_packed8(frame: &Video) -> AnyResult<LumaView<'_>> {
    let width = frame.width() as usize;
    let height = frame.height() as usize;
    let stride = frame.stride(0);
    validate_plane(frame.data(0), stride, width, height)?;
    Ok(LumaView {
        data: frame.data(0),
        stride,
    })
}

fn validate_plane(data: &[u8], stride: usize, row_bytes: usize, height: usize) -> AnyResult<()> {
    if stride < row_bytes {
        anyhow::bail!("unexpected luma stride ({stride}) for row size ({row_bytes})");
    }
    let needed = stride
        .checked_mul(height.saturating_sub(1))
        .and_then(|bytes| bytes.checked_add(row_bytes))
        .context("stride multiplication overflow")?;
    if data.len() < needed {
        anyhow::bail!(
            "insufficient luma data (have {}, need {needed})",
            data.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        LumaExtractor, ScalerInput, borrow_packed8, parse_hw_device_type, scale_to_gray8,
        validate_hw_frame_requirement,
    };
    use crate::SCuiseiError;
    use crate::simd_metrics;
    use ffmpeg_next::format::Pixel;
    use ffmpeg_next::util::frame::video::Video;

    fn make_video(format: Pixel, width: u32, height: u32) -> Video {
        let mut frame = Video::new(format, width, height);
        for plane in 0..frame.planes() {
            let fill = if plane == 0 { 32_u8 } else { 128_u8 };
            frame.data_mut(plane).fill(fill);
        }
        frame
    }

    fn fill_plane0_random_u16(frame: &mut Video, mask: u16, seed: u32) {
        let mut state = seed;
        for chunk in frame.data_mut(0).chunks_exact_mut(2) {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = u16::try_from((state >> 8) & 0xFFFF).expect("masked to 16 bits") & mask;
            chunk.copy_from_slice(&value.to_le_bytes());
        }
    }

    #[test]
    fn extractor_high_bit_output_matches_direct_swscale() {
        ffmpeg_next::init().expect("ffmpeg init should succeed");
        let mut src = Video::new(Pixel::YUV420P10LE, 64, 48);
        fill_plane0_random_u16(&mut src, 0x03FF, 0xC0FF_EE00);

        let mut scaler = None;
        let mut scaler_input = None;
        let mut gray = Video::empty();
        scale_to_gray8(&mut scaler, &mut scaler_input, &src, &mut gray)
            .expect("swscale conversion should succeed");

        let mut extractor = LumaExtractor::new();
        let view = extractor
            .luma_view(&src)
            .expect("extractor should convert high-bit frames");
        assert_eq!(view.data, gray.data(0));
        assert_eq!(view.stride, gray.stride(0));
    }

    #[test]
    fn packed8_strided_extraction_matches_plane() {
        ffmpeg_next::init().expect("ffmpeg init should succeed");
        let mut src = Video::new(Pixel::YUV420P, 61, 29);
        for (index, byte) in src.data_mut(0).iter_mut().enumerate() {
            *byte = u8::try_from(index % 251).expect("fits");
        }

        let view = borrow_packed8(&src).expect("borrow should not fail");
        let (data, stride) = (view.data, view.stride);

        let mut extracted = Vec::new();
        simd_metrics::extract_packed8(data, stride, 61, 29, &mut extracted);
        for y in 0..29_usize {
            assert_eq!(
                &extracted[y * 61..(y + 1) * 61],
                &data[y * stride..y * stride + 61],
            );
        }

        let plan = simd_metrics::DownscalePlan::new(61, 29, 17, 11);
        let mut sampled_strided = Vec::new();
        plan.run_packed8(data, stride, &mut sampled_strided);
        let mut sampled_contiguous = Vec::new();
        plan.run(&extracted, &mut sampled_contiguous);
        assert_eq!(sampled_strided, sampled_contiguous);
    }

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

    #[test]
    fn scaler_recreates_on_format_change() {
        ffmpeg_next::init().expect("ffmpeg init should succeed");

        let mut scaler = None;
        let mut scaler_input = None;
        let mut gray = Video::empty();
        let first = make_video(Pixel::YUV420P, 32, 32);
        scale_to_gray8(&mut scaler, &mut scaler_input, &first, &mut gray)
            .expect("first scale should succeed");
        assert_eq!(
            scaler_input,
            Some(ScalerInput {
                format: Pixel::YUV420P,
                width: 32,
                height: 32,
            })
        );

        let second = make_video(Pixel::NV12, 32, 32);
        scale_to_gray8(&mut scaler, &mut scaler_input, &second, &mut gray)
            .expect("second scale should succeed");
        assert_eq!(
            scaler_input,
            Some(ScalerInput {
                format: Pixel::NV12,
                width: 32,
                height: 32,
            })
        );
        assert_eq!(gray.format(), Pixel::GRAY8);
        assert_eq!(gray.width(), 32);
        assert_eq!(gray.height(), 32);
    }

    #[test]
    fn scaler_recreates_on_dimension_change() {
        ffmpeg_next::init().expect("ffmpeg init should succeed");

        let mut scaler = None;
        let mut scaler_input = None;
        let mut gray = Video::empty();
        let first = make_video(Pixel::YUV420P, 32, 32);
        scale_to_gray8(&mut scaler, &mut scaler_input, &first, &mut gray)
            .expect("first scale should succeed");

        let second = make_video(Pixel::YUV420P, 64, 48);
        scale_to_gray8(&mut scaler, &mut scaler_input, &second, &mut gray)
            .expect("second scale should succeed");
        assert_eq!(
            scaler_input,
            Some(ScalerInput {
                format: Pixel::YUV420P,
                width: 64,
                height: 48,
            })
        );
        assert_eq!(gray.format(), Pixel::GRAY8);
        assert_eq!(gray.width(), 64);
        assert_eq!(gray.height(), 48);
    }
}
