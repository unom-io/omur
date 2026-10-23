//! Every frame through oxideav-ac3: CRCs, the header fields, and the decoded audio against
//! the input. The behaviour of `push` and `pull` the host relies on.

use omur::{Codec, Config, Encoder, Error, Layout};
use oxideav_ac3::{audblk, bsi, decoder, syncinfo};

const RATE: f32 = 48_000.0;

/// Tones per channel in WAV order, plus noise on the back right: `frames` interleaved
/// samples of `layout`.
fn signal(layout: Layout, frames: usize) -> Vec<f32> {
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
        ];
        out.extend_from_slice(&wav[..ch]);
    }
    out
}

/// Pushes `pcm` in `chunk`-sample pushes, pulling after each; a refused push is split in
/// two and retried. Returns the frames with their pts.
fn encode(config: Config, pcm: &[f32], chunk: usize) -> Vec<(u64, Vec<u8>)> {
    fn feed(enc: &mut Encoder, pcm: &[f32], ch: usize, out: &mut Vec<(u64, Vec<u8>)>) {
        match enc.push(pcm) {
            Ok(()) => {
                if let Some(f) = enc.pull() {
                    assert_eq!(f.syncframes.len(), 1);
                    assert_eq!(f.syncframes[0], 0..f.bytes.len());
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

/// Decodes the frames with oxideav into one f32 stream per channel, bitstream order.
fn decode(frames: &[(u64, Vec<u8>)], nch: usize) -> Vec<Vec<f32>> {
    let mut state = audblk::Ac3State::new();
    let mut out = vec![Vec::new(); nch];
    let mut pcm = vec![0f32; 1536 * nch];
    for (_, f) in frames {
        let si = syncinfo::parse(f).unwrap();
        let b = bsi::parse(&f[5..]).unwrap();
        audblk::decode_frame(&mut state, &si, &b, f, &mut pcm).unwrap();
        for (i, s) in pcm.iter().enumerate() {
            out[i % nch].push(*s);
        }
    }
    out
}

fn snr_db(want: &[f32], got: &[f32]) -> f64 {
    let (mut s, mut e) = (0f64, 0f64);
    for (w, g) in want.iter().zip(got) {
        s += (*w as f64).powi(2);
        e += (*w as f64 - *g as f64).powi(2);
    }
    10.0 * (s / e).log10()
}

#[test]
fn every_rate_decodes_with_both_crcs_and_its_header() {
    // (layout, kbps, SNR floor per channel in bitstream order: L C R Ls Rs LFE). Rs is the
    // noise channel; above the coded bandwidth (15.8 kHz at 320, 20.3 kHz at 448 and 640)
    // its energy is lost.
    let cases: [(Layout, u32, &[f64]); 5] = [
        (Layout::Stereo, 192, &[78.0, 70.0]),
        (Layout::Stereo, 320, &[85.0, 82.0]),
        (
            Layout::Surround51,
            320,
            &[38.0, 39.0, 36.0, 35.0, 3.0, 36.0],
        ),
        (
            Layout::Surround51,
            448,
            &[50.0, 49.0, 45.0, 43.0, 6.0, 48.0],
        ),
        (
            Layout::Surround51,
            640,
            &[66.0, 66.0, 61.0, 60.0, 7.0, 62.0],
        ),
    ];
    for (layout, kbps, floor) in cases {
        let config = Config::new(Codec::Ac3, layout, kbps).unwrap();
        let pcm = signal(layout, 12 * 1536);
        let frames = encode(config, &pcm, 240);
        assert_eq!(frames.len(), 11);
        for (k, (pts, f)) in frames.iter().enumerate() {
            assert_eq!(*pts, 1536 * k as u64);
            assert_eq!(f.len(), config.frame_bytes());
            let crc = decoder::verify_packet_crc(f).unwrap();
            assert_eq!((crc.crc1_ok, crc.crc2_ok), (Some(true), Some(true)));
            let si = syncinfo::parse(f).unwrap();
            assert_eq!(si.frame_length as usize, config.frame_bytes());
            assert_eq!(si.sample_rate, 48_000);
            let b = bsi::parse(&f[5..]).unwrap();
            let (acmod, lfe) = if layout == Layout::Stereo {
                (2, false)
            } else {
                (7, true)
            };
            assert_eq!((b.bsid, b.bsmod, b.acmod, b.lfeon), (8, 0, acmod, lfe));
            assert_eq!(b.dialnorm, 31);
        }
        // WAV slot of each bitstream channel
        let wav: &[usize] = if layout == Layout::Stereo {
            &[0, 1]
        } else {
            &[0, 2, 1, 4, 5, 3]
        };
        let decoded = decode(&frames, wav.len());
        let ch = layout.channels();
        for (c, slot) in wav.iter().enumerate() {
            let want: Vec<f32> = pcm.iter().skip(*slot).step_by(ch).copied().collect();
            // the first frame fades in from no history
            let range = 1536..11 * 1536;
            let snr = snr_db(&want[range.clone()], &decoded[c][range]);
            assert!(
                snr > floor[c],
                "{layout:?} {kbps}: channel {c} at {snr:.1} dB"
            );
        }
    }
}

#[test]
fn output_does_not_depend_on_the_chunking() {
    let config = Config::new(Codec::Ac3, Layout::Surround51, 384).unwrap();
    let pcm = signal(Layout::Surround51, 10 * 1536);
    let whole = encode(config, &pcm, 240);
    assert_eq!(whole.len(), 9);
    for chunk in [1, 1000, 1536, 4096] {
        assert!(encode(config, &pcm, chunk) == whole, "chunk {chunk}");
    }
}

#[test]
fn a_frame_completes_with_its_lookahead() {
    let config = Config::new(Codec::Ac3, Layout::Stereo, 192).unwrap();
    let mut enc = Encoder::new(config).unwrap();
    let pcm = signal(Layout::Stereo, 4000);
    let delay = config.delay_samples();
    enc.push(&pcm[..2 * (delay - 1)]).unwrap();
    assert!(enc.pull().is_none());
    enc.push(&pcm[2 * (delay - 1)..2 * delay]).unwrap();
    let f = enc.pull().unwrap();
    assert_eq!((f.pts, f.bytes.len()), (0, 768));
    assert!(enc.pull().is_none());
}

#[test]
fn backlog_refuses_without_consuming() {
    let config = Config::new(Codec::Ac3, Layout::Stereo, 128).unwrap();
    let pcm = signal(Layout::Stereo, 8000);
    let mut enc = Encoder::new(config).unwrap();
    // two frames' worth in one push: refused
    assert_eq!(enc.push(&pcm[..2 * 3328]), Err(Error::Backlog));
    enc.push(&pcm[..2 * 1792]).unwrap();
    // the frame is unread: a push that completes the next is refused, a short one is not
    assert_eq!(enc.push(&pcm[2 * 1792..2 * 3328]), Err(Error::Backlog));
    enc.push(&pcm[2 * 1792..2 * 2000]).unwrap();
    assert_eq!(enc.push(&pcm[2 * 2000..2 * 3328]), Err(Error::Backlog));
    let first = enc.pull().unwrap().bytes.to_vec();
    enc.push(&pcm[2 * 2000..2 * 3328]).unwrap();
    let second = enc.pull().unwrap().pts;
    assert_eq!(second, 1536);
    assert_eq!(enc.push(&pcm[..3]), Err(Error::Length));
    // the refused pushes left no trace
    let reference = encode(config, &pcm[..2 * 3328], 1792);
    assert_eq!(reference[0].1, first);
}

#[test]
fn non_finite_input_decodes_as_silence() {
    let config = Config::new(Codec::Ac3, Layout::Surround51, 448).unwrap();
    let mut pcm = vec![0f32; 6 * 6 * 1536];
    for (i, v) in pcm.iter_mut().enumerate() {
        *v = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0][i % 4];
    }
    let frames = encode(config, &pcm, 240);
    assert_eq!(frames.len(), 5);
    for (_, f) in &frames {
        assert!(decoder::verify_packet_crc(f).unwrap().all_ok());
    }
    for ch in decode(&frames, 6) {
        assert!(ch.iter().all(|s| s.abs() < 1e-3));
    }
    // and full scale either side of it
    let loud: Vec<f32> = (0..6 * 4 * 1536)
        .map(|i| if i % 12 < 6 { 1e30 } else { -1e30 })
        .collect();
    for (_, f) in encode(config, &loud, 240) {
        assert!(decoder::verify_packet_crc(&f).unwrap().all_ok());
    }
}

#[test]
fn samples_past_full_scale_clamp_to_it() {
    let config = Config::new(Codec::Ac3, Layout::Surround51, 320).unwrap();
    let mut pcm = signal(Layout::Surround51, 4 * 1536);
    let mut clamped = pcm.clone();
    for (i, (v, c)) in pcm.iter_mut().zip(clamped.iter_mut()).enumerate() {
        if i % 997 == 0 {
            (*v, *c) = (-1e30, -1.0);
        } else if i % 991 == 0 {
            (*v, *c) = (3.5, 1.0);
        }
    }
    assert!(encode(config, &pcm, 240) == encode(config, &clamped, 240));
}

/// Tone bursts that start at full level out of silence: the frames switch to short
/// blocks, oxideav decodes them close to the input, and ffmpeg, where it is on `PATH`,
/// decodes them without a message.
#[test]
fn short_blocks_decode_cleanly() {
    let config = Config::new(Codec::Ac3, Layout::Surround51, 448).unwrap();
    let pcm: Vec<f32> = (0..8 * 1536 * 6)
        .map(|i| {
            let (t, c) = ((i / 6) % 2500, i % 6);
            let f = [0.31, 0.2, 0.43, 0.05, 0.37, 0.27][c];
            if t < 900 {
                0.5 * (t as f32 * f).sin() * (-(t as f32) / 300.0).exp()
            } else {
                0.0
            }
        })
        .collect();
    let frames = encode(config, &pcm, 240);
    let mut switched = 0;
    for (_, f) in &frames {
        let si = syncinfo::parse(f).unwrap();
        let b = bsi::parse(&f[5..]).unwrap();
        let side = audblk::parse_frame_side_info(&si, &b, f).unwrap();
        switched += side.iter().filter(|s| s.blksw.iter().any(|w| *w)).count();
        assert!(decoder::verify_packet_crc(f).unwrap().all_ok());
    }
    assert!(switched >= 4, "{switched} blocks switched");
    let decoded = decode(&frames, 6);
    let want: Vec<f32> = pcm.iter().step_by(6).copied().collect();
    let snr = snr_db(&want[1536..6 * 1536], &decoded[0][1536..6 * 1536]);
    assert!(snr > 15.0, "{snr:.1} dB");
    let path = std::env::temp_dir().join(format!("omur-short-{}.ac3", std::process::id()));
    let stream: Vec<u8> = frames.iter().flat_map(|(_, f)| f.iter().copied()).collect();
    std::fs::write(&path, stream).unwrap();
    let ffmpeg = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-f", "null", "-"])
        .output();
    let _ = std::fs::remove_file(&path);
    if let Ok(out) = ffmpeg {
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stderr), "");
    }
}
