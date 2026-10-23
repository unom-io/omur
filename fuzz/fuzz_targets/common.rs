//! What the targets share: a config, a push schedule and a signal drawn from the fuzz input,
//! and the host's loop of push, pull and split-on-`Backlog`.

// Each target uses part of this module.
#![allow(dead_code)]

use arbitrary::{Result, Unstructured};
use omur::{Blocks, Codec, Config, Encoder, Error, Layout};

/// Values a game can hand the encoder that a tone never would.
const SPECIAL: [f32; 14] = [
    0.0,
    -0.0,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    1e30,
    -1e30,
    1.0,
    -1.0,
    f32::MAX,
    f32::MIN,
    f32::MIN_POSITIVE,
    1e-40,
    -1e-40,
];

/// Frames one input encodes past the first; bounds the time an input takes.
const MAX_FRAMES: usize = 4;

/// Every codec: AC-3 and E-AC-3 at each frame length.
pub const CODECS: [Codec; 5] = [
    Codec::Ac3,
    Codec::Eac3 {
        blocks: Blocks::One,
    },
    Codec::Eac3 {
        blocks: Blocks::Two,
    },
    Codec::Eac3 {
        blocks: Blocks::Three,
    },
    Codec::Eac3 {
        blocks: Blocks::Six,
    },
];

/// A config: any codec and layout, any entry of its ladder. `None` for AC-3 7.1, which
/// `Config::new` refuses.
pub fn config(u: &mut Unstructured) -> Result<Option<Config>> {
    let codec = *u.choose(&CODECS)?;
    let layout = *u.choose(&[Layout::Stereo, Layout::Surround51, Layout::Surround71])?;
    let ladder = Config::bitrates(codec, layout);
    if ladder.is_empty() {
        assert_eq!(Config::new(codec, layout, 640), Err(Error::Layout));
        return Ok(None);
    }
    let kbps = *u.choose(ladder)?;
    Ok(Some(
        Config::new(codec, layout, kbps).expect("a ladder entry"),
    ))
}

/// One to eight push lengths in samples per channel, each 1 to 4096, used in a cycle.
pub fn schedule(u: &mut Unstructured) -> Result<Vec<usize>> {
    let n = u.int_in_range(1..=8)?;
    (0..n).map(|_| u.int_in_range(1..=4096)).collect()
}

/// Interleaved whole samples, at least one frame's worth: runs of the `SPECIAL` values,
/// square waves between them, a per-value mix, noise at four scales, denormals and raw bits.
pub fn signal(u: &mut Unstructured, config: Config) -> Result<Vec<f32>> {
    let ch = config.layout().channels();
    let delay = config.delay_samples();
    let cap = ch * (delay + MAX_FRAMES * config.frame_samples());
    let mut out = Vec::with_capacity(cap);
    let mut seed: u32 = u.arbitrary()?;
    let mut next = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    while !u.is_empty() && out.len() < cap {
        let kind = u.int_in_range(0..=SPECIAL.len() + 4)?;
        let len = u.int_in_range(1..=2048)?.min(cap - out.len());
        match kind.saturating_sub(SPECIAL.len() - 1) {
            0 => out.extend(std::iter::repeat_n(SPECIAL[kind], len)),
            1 => {
                let v = *u.choose(&SPECIAL)?;
                let period = ch * u.int_in_range(1..=512)?;
                out.extend((0..len).map(|i| {
                    if (i / period).is_multiple_of(2) {
                        v
                    } else {
                        -v
                    }
                }));
            }
            2 => out.extend((0..len).map(|_| SPECIAL[next() as usize % SPECIAL.len()])),
            3 => {
                let scale = *u.choose(&[1.0, 1e-3, 1e-38, 1e30])?;
                out.extend((0..len).map(|_| scale * (next() as i32 as f32 / i32::MAX as f32)));
            }
            4 => out.extend((0..len).map(|_| f32::from_bits(next() & 0x807f_ffff))),
            _ => {
                for _ in 0..len {
                    out.push(f32::from_bits(u.arbitrary()?));
                }
            }
        }
    }
    out.resize(out.len().max(ch * delay), 0.0);
    out.truncate(out.len() / ch * ch);
    Ok(out)
}

/// Pushes `pcm` in `schedule`'s lengths, pulling after every push; a push refused with
/// `Backlog` goes in as two halves. Returns every frame with its pts, and asserts one frame
/// per `frame_samples` past the first `delay_samples`, in order.
pub fn encode(config: Config, pcm: &[f32], schedule: &[usize]) -> Vec<(u64, Vec<u8>)> {
    let ch = config.layout().channels();
    let mut enc = Encoder::new(config).expect("an encoder");
    let mut out = Vec::new();
    let mut rest = pcm;
    for len in schedule.iter().cycle() {
        if rest.is_empty() {
            break;
        }
        let (chunk, tail) = rest.split_at((len * ch).min(rest.len()));
        feed(&mut enc, chunk, ch, &mut out);
        rest = tail;
    }
    let samples = pcm.len() / ch;
    assert_eq!(
        out.len(),
        frames(config, samples),
        "frames from {samples} samples"
    );
    for (k, (pts, _)) in out.iter().enumerate() {
        assert_eq!(
            *pts,
            (k * config.frame_samples()) as u64,
            "pts of frame {k}"
        );
    }
    out
}

/// Frames complete once `samples` per channel have arrived.
pub fn frames(config: Config, samples: usize) -> usize {
    match samples.checked_sub(config.delay_samples()) {
        Some(past) => 1 + past / config.frame_samples(),
        None => 0,
    }
}

/// The syncframes of a frame cover its bytes back to back: one, or 7.1's pair.
pub fn assert_syncframes(f: &omur::Frame) {
    let ok = match f.syncframes {
        [a] => *a == (0..f.bytes.len()),
        [a, b] => a.start == 0 && a.end == b.start && b.end == f.bytes.len(),
        _ => false,
    };
    assert!(
        ok,
        "syncframes {:?} of {} bytes",
        f.syncframes,
        f.bytes.len()
    );
}

fn feed(enc: &mut Encoder, pcm: &[f32], ch: usize, out: &mut Vec<(u64, Vec<u8>)>) {
    match enc.push(pcm) {
        Ok(()) => {
            if let Some(f) = enc.pull() {
                assert_syncframes(&f);
                out.push((f.pts, f.bytes.to_vec()));
            }
        }
        Err(Error::Backlog) => {
            let half = pcm.len() / ch / 2 * ch;
            feed(enc, &pcm[..half], ch, out);
            feed(enc, &pcm[half..], ch, out);
        }
        Err(e) => panic!("push of {} values: {e}", pcm.len()),
    }
}
