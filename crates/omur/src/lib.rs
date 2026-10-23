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
//! Status: AC-3 and E-AC-3 (every frame length) 2.0 and 5.1, with coupling and block
//! switching; E-AC-3 7.1 is P2 (`docs/design.md`).

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
    /// AC-3 frames are fixed by `frmsizecod`; an E-AC-3 frame is whole 16-bit words, so the
    /// rate rounds to the nearest word (1.5 kbit/s at most).
    pub fn frame_bytes(&self) -> usize {
        // bits = kbps * 1000 * samples / 48000; words = bits / 16; rounded to nearest.
        let words = (self.bitrate_kbps as usize * self.frame_samples() + 384) / 768;
        2 * words
    }

    /// The core that codes the stream: AC-3, or E-AC-3's independent substream.
    pub(crate) fn core(&self) -> ac3::Core {
        let (acmod, lfe) = match self.layout {
            Layout::Stereo => (2, false),
            _ => (7, true),
        };
        match self.codec {
            Codec::Ac3 => ac3::Core::ac3(self.layout, self.bitrate_kbps),
            Codec::Eac3 { blocks } => ac3::Core {
                acmod,
                lfe,
                kbps: self.bitrate_kbps,
                frame_bytes: self.frame_bytes(),
                blocks: blocks.count(),
                syntax: pack::Syntax::Eac3 { chanmap: None },
            },
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
    /// The pushed slice is not a whole number of interleaved samples.
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
            Error::Length => f.write_str("pcm length is not whole samples"),
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

/// The encoder. Allocates everything in [`Encoder::new`]; after that `push` and `pull` take
/// no lock, spawn nothing, allocate nothing and never panic.
pub struct Encoder {
    config: Config,
    core: ac3::Ac3,
    /// For each coded channel, in bitstream order, its slot in a WAV-order sample.
    wav_slot: &'static [usize],
    /// Samples of the current frame's span that have arrived.
    fill: usize,
    /// Index of the current frame's first sample.
    pts: u64,
    out: Box<[u8]>,
    out_pts: u64,
    ready: bool,
    syncframes: [Range<usize>; 1],
}

impl Encoder {
    /// Panics on 7.1, which is not built yet.
    pub fn new(config: Config) -> Result<Encoder, Error> {
        if config.layout == Layout::Surround71 {
            todo!("7.1")
        }
        let core = ac3::Ac3::new(config.core());
        let bytes = core.frame_bytes();
        Ok(Encoder {
            config,
            core,
            wav_slot: match config.layout {
                Layout::Stereo => &[0, 1],
                // L C R Ls Rs LFE from FL FR FC LFE BL BR
                _ => &[0, 2, 1, 4, 5, 3],
            },
            fill: 0,
            pts: 0,
            out: vec![0; bytes].into_boxed_slice(),
            out_pts: 0,
            ready: false,
            syncframes: [Range {
                start: 0,
                end: bytes,
            }],
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
            for (dst, slot) in self.core.pcm_mut().iter_mut().zip(self.wav_slot) {
                let dst = dst.iter_mut().skip(fill).take(take);
                for (d, sample) in dst.zip(chunk.chunks_exact(channels)) {
                    let v = sample.get(*slot).copied().unwrap_or(0.0);
                    *d = if v.is_finite() {
                        v.clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                }
            }
            self.fill += take;
            if self.fill == span {
                self.core.encode(&mut self.out);
                self.ready = true;
                self.out_pts = self.pts;
                self.pts += self.config.frame_samples() as u64;
                self.core.advance();
                self.fill = BLOCK_SAMPLES;
            }
        }
        Ok(())
    }

    /// The frame the last `push` completed, once.
    pub fn pull(&mut self) -> Option<Frame<'_>> {
        if !self.ready {
            return None;
        }
        self.ready = false;
        Some(Frame {
            bytes: &self.out,
            syncframes: &self.syncframes,
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
