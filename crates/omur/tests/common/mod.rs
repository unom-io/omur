//! What the integration tests share: the test signal and the host's push/pull loop.

// Each test binary uses part of this module.
#![allow(dead_code)]

use omur::{Blocks, Codec, Config, Encoder, Error, Layout};

const RATE: f32 = 48_000.0;

/// Every configuration the ladder allows: each codec and frame length, layout and rate.
pub fn ladder() -> Vec<Config> {
    let eac3 =
        [Blocks::One, Blocks::Two, Blocks::Three, Blocks::Six].map(|blocks| Codec::Eac3 { blocks });
    let mut out = vec![];
    for codec in [Codec::Ac3].into_iter().chain(eac3) {
        for layout in [Layout::Stereo, Layout::Surround51, Layout::Surround71] {
            for &kbps in Config::bitrates(codec, layout) {
                out.push(Config::new(codec, layout, kbps).unwrap());
            }
        }
    }
    out
}

/// Tones per channel in WAV order, plus noise on the back right and slow tremolo on the
/// side left: `frames` interleaved samples of `layout`.
pub fn signal(layout: Layout, frames: usize) -> Vec<f32> {
    let ch = layout.channels();
    let mut seed = 0x9e37_79b9u32;
    let mut out = Vec::with_capacity(frames * ch);
    for i in 0..frames {
        let t = i as f32 / RATE;
        let tone = |f: f32, a: f32| a * (2.0 * std::f32::consts::PI * f * t).sin();
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
        let wav = [
            tone(440.0, 0.3) + tone(1320.0, 0.1),
            tone(660.0, 0.3) + tone(3300.0, 0.05),
            tone(250.0, 0.25) + tone(2000.0, 0.08),
            tone(50.0, 0.4),
            tone(880.0, 0.2) + tone(5000.0, 0.03),
            0.2 * noise,
            tone(3000.0, 0.2) * (0.6 + 0.4 * tone(2.0, 1.0)),
            tone(1100.0, 0.25) + tone(7000.0, 0.02),
        ];
        out.extend_from_slice(&wav[..ch]);
    }
    out
}

/// A tone as the recurrence `s[n] = k·s[n−1] − s[n−2]`, `k = 2·cos ω`, started at
/// `s[1] = A·sin ω`: a sine with no libm call, so every platform computes the same bits.
struct Tone {
    k: f32,
    s1: f32,
    s2: f32,
}

impl Tone {
    fn new(k: f32, s1: f32) -> Tone {
        Tone { k, s1, s2: 0.0 }
    }

    fn next(&mut self) -> f32 {
        let s = self.k * self.s1 - self.s2;
        self.s2 = self.s1;
        self.s1 = s;
        self.s2
    }
}

/// `signal`'s tones and noise, from constants alone: byte-identical on every platform, so
/// the golden hashes hold on every CI runner. 7.1's seventh channel is a plain 3 kHz tone.
pub fn pinned_signal(layout: Layout, frames: usize) -> Vec<f32> {
    let ch = layout.channels();
    let mut tones: [Vec<Tone>; 8] = [
        vec![
            Tone::new(1.9966836, 0.017269207),
            Tone::new(1.9702187, 0.01719291),
        ],
        vec![
            Tone::new(1.9925407, 0.02588591),
            Tone::new(1.8162863, 0.020932987),
        ],
        vec![
            Tone::new(1.9989291, 0.008179771),
            Tone::new(1.9318516, 0.020705523),
        ],
        vec![Tone::new(1.9999572, 0.0026179752)],
        vec![
            Tone::new(1.9867455, 0.02298743),
            Tone::new(1.5867066, 0.018262843),
        ],
        vec![],
        vec![Tone::new(1.847759, 0.076536685)],
        vec![
            Tone::new(1.9793028, 0.035873156),
            Tone::new(1.2175229, 0.015867068),
        ],
    ];
    let mut seed = 0x9e37_79b9u32;
    let mut out = Vec::with_capacity(frames * ch);
    for _ in 0..frames {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
        for (c, t) in tones.iter_mut().enumerate().take(ch) {
            let v = if c == 5 {
                0.2 * noise
            } else {
                t.iter_mut().map(Tone::next).sum()
            };
            out.push(v);
        }
    }
    out
}

/// Pushes `pcm` in `chunk`-sample pushes, pulling after each; a refused push is split in
/// two and retried. Returns the frames with their pts.
pub fn encode(config: Config, pcm: &[f32], chunk: usize) -> Vec<(u64, Vec<u8>)> {
    fn feed(enc: &mut Encoder, pcm: &[f32], ch: usize, out: &mut Vec<(u64, Vec<u8>)>) {
        match enc.push(pcm) {
            Ok(()) => {
                if let Some(f) = enc.pull() {
                    // one syncframe, or 7.1's pair, back to back
                    let ends: Vec<usize> = f.syncframes.iter().map(|r| r.end).collect();
                    let starts: Vec<usize> = f.syncframes.iter().map(|r| r.start).collect();
                    assert!(matches!(ends.len(), 1 | 2));
                    assert_eq!(starts[0], 0);
                    assert_eq!(starts[1..], ends[..ends.len() - 1]);
                    assert_eq!(ends.last(), Some(&f.bytes.len()));
                    out.push((f.pts, f.bytes.to_vec()));
                }
            }
            Err(Error::Backlog) => {
                let half = pcm.len() / ch / 2 * ch;
                feed(enc, &pcm[..half], ch, out);
                feed(enc, &pcm[half..], ch, out);
            }
            Err(e) => panic!("{e}"),
        }
    }
    let ch = config.layout().channels();
    let mut enc = Encoder::new(config).unwrap();
    let mut out = Vec::new();
    for c in pcm.chunks(chunk * ch) {
        feed(&mut enc, c, ch, &mut out);
    }
    out
}

/// ffmpeg, where it is on `PATH`, decodes `stream` without a message. `name` ends in the
/// stream's extension.
pub fn ffmpeg_decodes(stream: &[u8], name: &str) {
    let path = std::env::temp_dir().join(format!("omur-{}-{name}", std::process::id()));
    std::fs::write(&path, stream).unwrap();
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-f", "null", "-"])
        .output();
    let _ = std::fs::remove_file(&path);
    if let Ok(out) = out {
        assert!(out.status.success(), "{name}");
        assert_eq!(String::from_utf8_lossy(&out.stderr), "", "{name}");
    }
}
