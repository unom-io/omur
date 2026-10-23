//! omur — an AC-3 and E-AC-3 encoder (ATSC A/52:2018) in safe Rust for a real-time audio
//! thread: interleaved f32 at 48 kHz in, whole syncframes out, nothing allocated after
//! [`Encoder::new`].
//!
//! ```no_run
//! # fn send(_: &[u8]) {}
//! # fn main() -> Result<(), omur::Error> {
//! use omur::{Codec, Config, Encoder, Layout};
//! let mut enc = Encoder::new(Config::new(Codec::Ac3, Layout::Surround51, 448)?)?;
//! let chunk = vec![0.0f32; 6 * 240]; // 5 ms of 5.1 in WAV order
//! enc.push(&chunk)?;
//! while let Some(frame) = enc.pull() {
//!     send(frame.bytes); // one syncframe; 7.1 E-AC-3 carries two in one access unit
//! }
//! # Ok(()) }
//! ```
//!
//! Object audio (ETSI TS 103 420) is the [`joc`] module, behind the `joc` feature.
//!
//! Status: AC-3 2.0 and 5.1, E-AC-3 2.0, 5.1 and 7.1 at every frame length, with coupling
//! and block switching (`docs/design.md`).

mod ac3;
mod bitalloc;
mod bits;
mod eac3;
mod exponents;
#[cfg(feature = "joc")]
pub mod joc;
mod mantissas;
mod mdct;
mod pack;
mod tables;
mod tables_gen;
mod tuner;

use core::f32::consts::FRAC_1_SQRT_2;
use core::fmt;
use core::ops::Range;

/// Samples per channel in one audio block. Every frame is a whole number of them.
pub const BLOCK_SAMPLES: usize = 256;
/// The only sample rate. The consumer's audio runs at 48 kHz and nothing else.
pub const SAMPLE_RATE_HZ: u32 = 48_000;

/// Which bitstream to write. E-AC-3 chooses its frame length; AC-3 is always six blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    /// A/52 §5: six blocks, 1536 samples, 32 ms. What optical S/PDIF carries, 5.1 at most.
    Ac3,
    /// A/52 Annex E, `bsid` 16. HDMI ARC/eARC. 7.1 takes two substreams.
    Eac3 { blocks: Blocks },
}

/// Blocks per E-AC-3 syncframe (`numblkscod`). Fewer blocks: less delay and a smaller
/// frame, more header and fresh exponents per sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Blocks {
    One = 1,
    Two = 2,
    Three = 3,
    #[default]
    Six = 6,
}

impl Blocks {
    pub fn count(self) -> usize {
        self as usize
    }
}

/// Speaker layout. Input is interleaved in WAV order: FL FR FC LFE BL BR, then SL SR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Stereo,
    Surround51,
    Surround71,
}

impl Layout {
    pub fn channels(self) -> usize {
        match self {
            Layout::Stereo => 2,
            Layout::Surround51 => 6,
            Layout::Surround71 => 8,
        }
    }
}

/// Everything the encoder needs. [`Config::new`] validates; the queries need no encoder, so
/// a host can size its datagrams and its A/V sync before it builds one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    codec: Codec,
    layout: Layout,
    bitrate_kbps: u32,
}

impl Config {
    pub fn new(codec: Codec, layout: Layout, bitrate_kbps: u32) -> Result<Config, Error> {
        let ladder = Config::bitrates(codec, layout);
        if ladder.is_empty() {
            return Err(Error::Layout);
        }
        if !ladder.contains(&bitrate_kbps) {
            return Err(Error::Bitrate { kbps: bitrate_kbps });
        }
        Ok(Config {
            codec,
            layout,
            bitrate_kbps,
        })
    }

    pub fn codec(&self) -> Codec {
        self.codec
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn bitrate_kbps(&self) -> u32 {
        self.bitrate_kbps
    }

    /// The bitrates a (codec, layout) pair accepts, in kbit/s. Empty when the pair cannot
    /// be coded: AC-3 stops at 5.1. The ladder is a proposal until P1 and P2 measure it.
    pub fn bitrates(codec: Codec, layout: Layout) -> &'static [u32] {
        match (codec, layout) {
            (_, Layout::Stereo) => &[128, 160, 192, 224, 256, 320],
            (Codec::Ac3, Layout::Surround51) => &[256, 320, 384, 448, 512, 576, 640],
            (Codec::Ac3, Layout::Surround71) => &[],
            (Codec::Eac3 { .. }, Layout::Surround51) => &[256, 320, 384, 448, 512, 640, 768, 1024],
            (Codec::Eac3 { .. }, Layout::Surround71) => &[640, 768, 1024, 1280, 1536],
        }
    }

    /// Samples per channel in one frame: 1536 on AC-3, 256 per block on E-AC-3.
    pub fn frame_samples(&self) -> usize {
        match self.codec {
            Codec::Ac3 => 6 * BLOCK_SAMPLES,
            Codec::Eac3 { blocks } => blocks.count() * BLOCK_SAMPLES,
        }
    }

    /// Bytes in one access unit: one syncframe, or the pair a 7.1 E-AC-3 stream writes.
    /// AC-3 frames are fixed by `frmsizecod`; an E-AC-3 syncframe is whole 16-bit words, so
    /// its share of the rate rounds to the nearest word (1.5 kbit/s at most). The 7.1 pair
    /// is the sum of its two syncframes, each rounded on its own.
    pub fn frame_bytes(&self) -> usize {
        let [independent, dependent] = self.syncframe_bytes();
        independent + dependent
    }

    /// Bytes of the independent syncframe, then of the dependent one (0 but on 7.1). 7.1
    /// splits the rate 7:5 by coded channels: 5.1 and a coupling channel, against four
    /// channels and one. Of 2:1, 3:2, 5:3 and 7:5, 7:5 keeps the probe signal's weakest
    /// channel highest at 768 and 1,024 kbit/s. Each is 4,096 bytes at most on the ladder.
    fn syncframe_bytes(&self) -> [usize; 2] {
        // words = kbps · 1000 · samples / 48000 / 16 of the share num/den, to the nearest
        let words = |num: usize, den: usize| {
            (self.bitrate_kbps as usize * self.frame_samples() * num + 384 * den) / (768 * den)
        };
        match self.layout {
            Layout::Surround71 => [2 * words(7, 12), 2 * words(5, 12)],
            _ => [2 * words(1, 1), 0],
        }
    }

    /// The cores that code the stream: AC-3 or E-AC-3's independent substream, and 7.1's
    /// dependent substream. Each picks its coupling range by its share of the rate.
    pub(crate) fn cores(&self) -> (ac3::Core, Option<ac3::Core>) {
        let blocks = match self.codec {
            Codec::Ac3 => return (ac3::Core::ac3(self.layout, self.bitrate_kbps), None),
            Codec::Eac3 { blocks } => blocks.count(),
        };
        let [independent, dependent] = self.syncframe_bytes();
        let kbps = self.bitrate_kbps;
        let core = |acmod, lfe, frame_bytes, kbps, chanmap| ac3::Core {
            acmod,
            lfe,
            kbps,
            frame_bytes,
            blocks,
            syntax: pack::Syntax::Eac3 { chanmap },
        };
        match self.layout {
            Layout::Stereo => (core(2, false, independent, kbps, None), None),
            Layout::Surround51 => (core(7, true, independent, kbps, None), None),
            Layout::Surround71 => (
                core(7, true, independent, kbps * 7 / 12, None),
                Some(core(
                    6,
                    false,
                    dependent,
                    kbps * 5 / 12,
                    Some(eac3::CHANMAP_71),
                )),
            ),
        }
    }

    /// Samples from a sample entering `push` to the frame holding it leaving `pull`: the
    /// frame plus the 256 the last block's window reaches past it. The receiver's
    /// overlap-add adds another 256 on its side.
    pub fn delay_samples(&self) -> usize {
        self.frame_samples() + BLOCK_SAMPLES
    }
}

/// Why a configuration or a push is refused. Construction validates; encoding never panics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Not on [`Config::bitrates`] for this codec and layout.
    Bitrate { kbps: u32 },
    /// AC-3 carries 5.1 at most; 7.1 needs E-AC-3.
    Layout,
    /// A frame completed while the previous one was still unread. Pull after every push.
    Backlog,
    /// The pushed slice is not a whole number of interleaved samples, or an extension does
    /// not fit the stream's frames.
    Length,
    /// An object index at or past `ObjectConfig::objects`.
    #[cfg(feature = "joc")]
    Object { index: u8 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Bitrate { kbps } => write!(f, "bitrate {kbps} kbit/s not in the ladder"),
            Error::Layout => f.write_str("AC-3 carries 5.1 at most"),
            Error::Backlog => f.write_str("frame ready and unread"),
            Error::Length => f.write_str("pcm is not whole samples, or extension too long"),
            #[cfg(feature = "joc")]
            Error::Object { index } => write!(f, "object {index} is past the configured count"),
        }
    }
}

impl std::error::Error for Error {}

/// One encoded access unit, borrowed from the encoder until the next [`Encoder::push`].
#[derive(Debug)]
pub struct Frame<'a> {
    /// Whole syncframes, back to back.
    pub bytes: &'a [u8],
    /// Where each syncframe lies in `bytes`: one range, or two for 7.1 E-AC-3 (independent
    /// substream, then dependent). A datagram carries one or both.
    pub syncframes: &'a [Range<usize>],
    /// Index of the frame's first sample, counted from construction.
    pub pts: u64,
}

/// Where a coded channel's samples come from in a WAV-order sample.
#[derive(Clone, Copy)]
enum Source {
    Wav(usize),
    /// A side and a back channel at −3 dB each: 7.1's surround as a 5.1 receiver plays it.
    Fold(usize, usize),
}

/// `sample[slot]` clamped to full scale; a non-finite or missing sample reads as 0.
fn read(sample: &[f32], slot: usize) -> f32 {
    match sample.get(slot) {
        Some(v) if v.is_finite() => v.clamp(-1.0, 1.0),
        _ => 0.0,
    }
}

/// Copies `chunk`'s samples of `src` into `dst`.
fn copy_in(dst: &mut [f32], chunk: &[f32], channels: usize, src: Source) {
    match channels {
        2 => copy_from::<2>(dst, chunk, src),
        6 => copy_from::<6>(dst, chunk, src),
        8 => copy_from::<8>(dst, chunk, src),
        _ => {}
    }
}

/// `copy_in` for `C` channels a sample.
fn copy_from<const C: usize>(dst: &mut [f32], chunk: &[f32], src: Source) {
    let samples = chunk.as_chunks::<C>().0;
    match src {
        Source::Wav(slot) => {
            for (d, s) in dst.iter_mut().zip(samples) {
                *d = read(s, slot);
            }
        }
        Source::Fold(side, back) => {
            for (d, s) in dst.iter_mut().zip(samples) {
                *d = ((read(s, side) + read(s, back)) * FRAC_1_SQRT_2).clamp(-1.0, 1.0);
            }
        }
    }
}

/// The encoder. Allocates everything in [`Encoder::new`]; after that `push` and `pull` take
/// no lock, spawn nothing, allocate nothing and never panic.
pub struct Encoder {
    config: Config,
    core: ac3::Ac3,
    /// 7.1's dependent substream.
    dep: Option<ac3::Ac3>,
    /// For each coded channel of `core`, then of `dep`, in bitstream order, its samples.
    sources: [&'static [Source]; 2],
    /// Samples of the current frame's span that have arrived.
    fill: usize,
    /// Index of the current frame's first sample.
    pts: u64,
    out: Box<[u8]>,
    out_pts: u64,
    ready: bool,
    syncframes: [Range<usize>; 2],
}

impl Encoder {
    /// Allocates everything the stream needs. A [`Config`] is valid once built, so every
    /// one constructs.
    pub fn new(config: Config) -> Result<Encoder, Error> {
        use Source::{Fold, Wav};
        let (core, dep) = config.cores();
        let (core, dep) = (ac3::Ac3::new(core), dep.map(ac3::Ac3::new));
        let first = core.frame_bytes();
        let bytes = config.frame_bytes();
        Ok(Encoder {
            config,
            core,
            dep,
            sources: match config.layout {
                Layout::Stereo => [&[Wav(0), Wav(1)], &[]],
                // L C R Ls Rs LFE from FL FR FC LFE BL BR
                Layout::Surround51 => [&[Wav(0), Wav(2), Wav(1), Wav(4), Wav(5), Wav(3)], &[]],
                // the fold, then Ls Rs Lrs Rrs from SL SR BL BR
                Layout::Surround71 => [
                    &[Wav(0), Wav(2), Wav(1), Fold(6, 4), Fold(7, 5), Wav(3)],
                    &[Wav(6), Wav(7), Wav(4), Wav(5)],
                ],
            },
            fill: 0,
            pts: 0,
            out: vec![0; bytes].into_boxed_slice(),
            out_pts: 0,
            ready: false,
            syncframes: [0..first, first..bytes],
        })
    }

    pub fn config(&self) -> Config {
        self.config
    }

    /// Queue interleaved f32 in WAV order, at most one frame ([`Config::frame_samples`],
    /// 1536 on AC-3) per push. Samples are clamped to ±1.0; non-finite ones read as 0. A
    /// frame is encoded as soon as [`Config::delay_samples`] have arrived past its start,
    /// and every frame comes out. [`Error::Backlog`] refuses, and consumes nothing of, a
    /// push that would complete a frame while the last one is unread, or complete two —
    /// which a push longer than a frame can.
    pub fn push(&mut self, pcm: &[f32]) -> Result<(), Error> {
        let channels = self.config.layout.channels();
        if !pcm.len().is_multiple_of(channels) {
            return Err(Error::Length);
        }
        let n = pcm.len() / channels;
        let span = self.core.span();
        let need = span - self.fill;
        if n >= need && (self.ready || n - need >= self.config.frame_samples()) {
            return Err(Error::Backlog);
        }
        let mut rest = pcm;
        while !rest.is_empty() {
            let take = (rest.len() / channels).min(span - self.fill);
            let (chunk, after) = rest.split_at(take * channels);
            rest = after;
            let fill = self.fill;
            let cores = core::iter::once(&mut self.core).chain(self.dep.as_mut());
            for (core, sources) in cores.zip(self.sources) {
                for (dst, src) in core.pcm_mut().iter_mut().zip(sources) {
                    if let Some(dst) = dst.get_mut(fill..fill + take) {
                        copy_in(dst, chunk, channels, *src);
                    }
                }
            }
            self.fill += take;
            if self.fill == span {
                if let Some((first, second)) = self.out.split_at_mut_checked(self.syncframes[0].end)
                {
                    self.core.encode(first);
                    if let Some(dep) = self.dep.as_mut() {
                        dep.encode(second);
                    }
                }
                if let Some(dep) = self.dep.as_mut() {
                    dep.advance();
                }
                self.ready = true;
                self.out_pts = self.pts;
                self.pts += self.config.frame_samples() as u64;
                self.core.advance();
                self.fill = BLOCK_SAMPLES;
            }
        }
        Ok(())
    }

    /// Carries `bytes` in the skip fields of the next frame a `push` completes, spread over
    /// its blocks in order: the hook for object audio's container (docs/design.md §8.3).
    /// The frame's mantissas get what is left. A second call before that frame replaces
    /// the first. E-AC-3 2.0 and 5.1 only: [`Error::Length`] on AC-3 or 7.1, or when
    /// `bytes` passes what the stream's busiest frame leaves free.
    pub fn set_extension(&mut self, bytes: &[u8]) -> Result<(), Error> {
        match (self.config.codec, self.config.layout) {
            (Codec::Ac3, _) | (_, Layout::Surround71) => Err(Error::Length),
            _ => self.core.set_extension(bytes),
        }
    }

    /// The frame the last `push` completed, once.
    pub fn pull(&mut self) -> Option<Frame<'_>> {
        if !self.ready {
            return None;
        }
        self.ready = false;
        Some(Frame {
            bytes: &self.out,
            syncframes: &self.syncframes[..1 + usize::from(self.dep.is_some())],
            pts: self.out_pts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(codec: Codec, layout: Layout, kbps: u32) -> Config {
        Config::new(codec, layout, kbps).unwrap()
    }

    #[test]
    fn ac3_frames_are_four_bytes_per_kbps() {
        for (kbps, bytes) in [(320, 1280), (448, 1792), (640, 2560)] {
            let c = cfg(Codec::Ac3, Layout::Surround51, kbps);
            assert_eq!(c.frame_bytes(), bytes);
            assert_eq!(c.frame_samples(), 1536);
            assert_eq!(c.delay_samples(), 1792);
        }
    }

    #[test]
    fn eac3_short_frames_round_to_words() {
        let one = cfg(
            Codec::Eac3 {
                blocks: Blocks::One,
            },
            Layout::Surround51,
            640,
        );
        assert_eq!(one.frame_bytes(), 426); // 213.3 words
        assert_eq!(one.delay_samples(), 512);
        let two = cfg(
            Codec::Eac3 {
                blocks: Blocks::Two,
            },
            Layout::Surround51,
            448,
        );
        assert_eq!(two.frame_bytes(), 598); // 298.7 words
        let six = cfg(
            Codec::Eac3 {
                blocks: Blocks::Six,
            },
            Layout::Surround51,
            1024,
        );
        assert_eq!(six.frame_bytes(), 4096); // frmsiz's ceiling
    }

    /// 7.1's pair splits the access unit 7:5, each syncframe whole words and within
    /// `frmsiz`'s 4,096 bytes; every ladder entry builds an encoder.
    #[test]
    fn every_ladder_entry_fits_its_syncframes() {
        let six = Codec::Eac3 {
            blocks: Blocks::Six,
        };
        let c = cfg(six, Layout::Surround71, 1024);
        assert_eq!(c.syncframe_bytes(), [2390, 1706]); // 1194.7 and 853.3 words
        assert_eq!(c.frame_bytes(), 4096);
        for codec in [Blocks::One, Blocks::Two, Blocks::Three, Blocks::Six]
            .map(|blocks| Codec::Eac3 { blocks })
            .into_iter()
            .chain([Codec::Ac3])
        {
            for layout in [Layout::Stereo, Layout::Surround51, Layout::Surround71] {
                for &kbps in Config::bitrates(codec, layout) {
                    let c = cfg(codec, layout, kbps);
                    for bytes in c.syncframe_bytes() {
                        assert!(bytes % 2 == 0 && bytes <= 4096, "{c:?}");
                    }
                    let enc = Encoder::new(c).unwrap();
                    assert_eq!(enc.out.len(), c.frame_bytes());
                }
            }
        }
    }

    #[test]
    fn construction_refuses_what_the_stream_cannot_carry() {
        assert_eq!(
            Config::new(Codec::Ac3, Layout::Surround71, 640),
            Err(Error::Layout)
        );
        assert_eq!(
            Config::new(Codec::Ac3, Layout::Surround51, 400),
            Err(Error::Bitrate { kbps: 400 })
        );
    }
}
