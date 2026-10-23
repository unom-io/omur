//! Random interleavings of `push`, `pull`, `set_extension` and `Config::new` against the
//! documented contract: `Length` for a partial sample, `Backlog` for a second frame or one
//! past an unread frame, nothing consumed by a refusal, each frame pulled once, `Layout` and
//! `Bitrate` from `Config::new`, `Length` for an extension on AC-3, on 7.1 or past the
//! frame. A shadow encoder takes only the accepted calls; its frames must match.

#![no_main]

mod common;

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use omur::{Codec, Config, Encoder, Error, Layout};

struct Host {
    config: Config,
    enc: Encoder,
    shadow: Encoder,
    /// Samples per channel `enc` accepted.
    pushed: usize,
    /// The frame `enc` holds unread, as the shadow made it.
    unread: Option<(u64, Vec<u8>)>,
    seed: u32,
}

impl Host {
    fn new(config: Config) -> Host {
        Host {
            config,
            enc: Encoder::new(config).expect("an encoder"),
            shadow: Encoder::new(config).expect("an encoder"),
            pushed: 0,
            unread: None,
            seed: 1,
        }
    }

    fn push(&mut self, len: usize) {
        let pcm: Vec<f32> = (0..len)
            .map(|_| {
                self.seed = self
                    .seed
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                self.seed as i32 as f32 / i32::MAX as f32
            })
            .collect();
        let got = self.enc.push(&pcm);
        let ch = self.config.layout().channels();
        if !len.is_multiple_of(ch) {
            assert_eq!(got, Err(Error::Length), "push of {len} values");
            return;
        }
        let n = len / ch;
        let done =
            common::frames(self.config, self.pushed + n) - common::frames(self.config, self.pushed);
        if done > 1 || (done == 1 && self.unread.is_some()) {
            assert_eq!(
                got,
                Err(Error::Backlog),
                "push of {n} samples after {}",
                self.pushed
            );
            return;
        }
        assert_eq!(got, Ok(()), "push of {n} samples after {}", self.pushed);
        self.shadow.push(&pcm).expect("shadow push");
        self.pushed += n;
        let made = self.shadow.pull().map(|f| (f.pts, f.bytes.to_vec()));
        assert_eq!(
            made.is_some(),
            done == 1,
            "shadow frame after {}",
            self.pushed
        );
        if made.is_some() {
            self.unread = made;
        }
    }

    /// An extension of `len` bytes: refused on AC-3 and 7.1, taken up to what the frame
    /// leaves (a frame holds 34 bytes or more, the least on the ladder).
    fn extension(&mut self, len: usize) {
        let bytes: Vec<u8> = (0..len).map(|i| i as u8).collect();
        let got = self.enc.set_extension(&bytes);
        let carried = matches!(self.config.codec(), Codec::Eac3 { .. })
            && self.config.layout() != Layout::Surround71;
        if !carried {
            assert_eq!(got, Err(Error::Length), "extension of {len} bytes");
            return;
        }
        if len <= 34 {
            assert_eq!(got, Ok(()), "extension of {len} bytes");
        }
        assert!(matches!(got, Ok(()) | Err(Error::Length)), "{got:?}");
        if got.is_ok() {
            self.shadow.set_extension(&bytes).expect("shadow extension");
        }
    }

    fn pull(&mut self) {
        let got = self.enc.pull().map(|f| (f.pts, f.bytes.to_vec()));
        if let Some(f) = self.enc.pull() {
            panic!("a second pull of the frame at {}", f.pts);
        }
        if let Some((pts, _)) = &got {
            let first =
                (common::frames(self.config, self.pushed) - 1) * self.config.frame_samples();
            assert_eq!(*pts, first as u64, "pts after {}", self.pushed);
        }
        assert!(
            got == self.unread.take(),
            "pulled frame after {}",
            self.pushed
        );
    }
}

/// `Config::new` on the triple these pick; `kbps` overrides the ladder entry `pick` names.
/// Returns the config when it is valid.
fn config(codec: u8, layout: u8, pick: u8, kbps: Option<u32>) -> Option<Config> {
    let codec = common::CODECS[usize::from(codec) % common::CODECS.len()];
    let layout = [Layout::Stereo, Layout::Surround51, Layout::Surround71][usize::from(layout % 3)];
    let ladder = Config::bitrates(codec, layout);
    let entry = ladder.get(usize::from(pick) % ladder.len().max(1));
    let kbps = kbps.or(entry.copied()).unwrap_or(u32::from(pick));
    let got = Config::new(codec, layout, kbps);
    if ladder.is_empty() {
        assert_eq!(got, Err(Error::Layout), "{codec:?} {layout:?} {kbps}");
        return None;
    }
    if !ladder.contains(&kbps) {
        assert_eq!(got, Err(Error::Bitrate { kbps }), "{codec:?} {layout:?}");
        return None;
    }
    let c = got.expect("a ladder entry");
    assert_eq!(
        (c.codec(), c.layout(), c.bitrate_kbps()),
        (codec, layout, kbps)
    );
    let samples = match codec {
        Codec::Ac3 => 1536,
        Codec::Eac3 { blocks } => 256 * blocks.count(),
    };
    assert_eq!(
        (c.frame_samples(), c.delay_samples()),
        (samples, samples + 256)
    );
    // Whole 16-bit words, each syncframe within half a word of its share of
    // kbps × samples / 384 bytes: within a word for 7.1's pair.
    let bytes = c.frame_bytes() as u64;
    let exact = u64::from(kbps) * samples as u64;
    let syncframes = 1 + u64::from(layout == Layout::Surround71);
    assert!(
        bytes.is_multiple_of(2) && (384 * bytes).abs_diff(exact) <= 384 * syncframes,
        "{c:?}: {bytes} bytes"
    );
    if codec == Codec::Ac3 {
        assert_eq!(bytes, 4 * u64::from(kbps), "{c:?}");
    }
    Some(c)
}

fuzz_target!(|data: &[u8]| {
    let _ = run(&mut Unstructured::new(data));
});

fn run(u: &mut Unstructured) -> arbitrary::Result<()> {
    let Some(start) = common::config(u)? else {
        return Ok(());
    };
    let mut host = Host::new(start);
    while !u.is_empty() {
        match u.int_in_range(0..=3)? {
            0 => host.push(usize::from(u.arbitrary::<u16>()?)),
            1 => host.pull(),
            2 => host.extension(usize::from(u.arbitrary::<u16>()? % 4200)),
            _ => {
                let c = config(
                    u.arbitrary()?,
                    u.arbitrary()?,
                    u.arbitrary()?,
                    u.arbitrary()?,
                );
                if let Some(c) = c {
                    host = Host::new(c);
                }
            }
        }
    }
    Ok(())
}
