// AC-3 encoder probe: omur or a reference Rust encoder vs ffmpeg's ac3, decoded by ffmpeg.
// usage: probe gen <out.s16> | enc <in.s16> <out.ac3> <kbps> | cmp <ref.s16> <dec.f32>
//        | dump <stream.ac3> [summary]
// `enc` and `cmp` read `CH` channels (6, or 2 for FL FR); `enc` feeds `CHUNK` samples a push.
mod dump;

use std::f32::consts::PI;
use std::time::Instant;

#[cfg(all(not(feature = "omur"), not(feature = "vuio")))]
use {oxideav_ac3 as ac3, oxideav_core as codec_core};
#[cfg(all(not(feature = "omur"), feature = "vuio"))]
use {vuio_codec_ac3 as ac3, vuio_codec_core as codec_core};

#[cfg(not(feature = "omur"))]
use codec_core::{AudioFrame, CodecId, CodecParameters, Error, Frame, SampleFormat};

const SR: usize = 48_000;
const CH: usize = 6; // WAV order: FL FR FC LFE SL SR

/// Channels `enc` and `cmp` read: `CH` from the environment, 6 by default.
fn channels() -> usize {
    std::env::var("CH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CH)
}

/// Samples per push: `CHUNK` from the environment, 5 ms by default.
fn chunk() -> usize {
    std::env::var("CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(SR / 200)
}

fn generate(path: &str) {
    let n = SR * 10;
    let mut seed = 0x1234_5678u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    };
    let mut pink = [0f32; 3];
    let mut out = Vec::with_capacity(n * CH);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        // FL: harmonic tone with tremolo (music-like)
        let fl = (1..8)
            .map(|h| (2.0 * PI * 220.0 * h as f32 * t).sin() / h as f32)
            .sum::<f32>()
            * 0.25
            * (0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin());
        // FR: log sweep 50 Hz -> 16 kHz over 10 s
        let k = (16_000f32 / 50.0).ln() / 10.0;
        let fr = 0.3 * (2.0 * PI * 50.0 * ((k * t).exp() - 1.0) / k).sin();
        // FC: speech-like band noise bursts, 180 ms on / 120 ms off
        let w = rnd();
        let fc = if (t * 1000.0) as u32 % 300 < 180 {
            0.3 * w * (2.0 * PI * 900.0 * t).sin()
        } else {
            0.0
        };
        // LFE: 40 + 80 Hz
        let lfe = 0.3 * (2.0 * PI * 40.0 * t).sin() + 0.15 * (2.0 * PI * 80.0 * t).sin();
        // SL: pinkish noise
        let wn = rnd();
        pink[0] = 0.997 * pink[0] + 0.029 * wn;
        pink[1] = 0.985 * pink[1] + 0.032 * wn;
        pink[2] = 0.950 * pink[2] + 0.048 * wn;
        let sl = (pink[0] + pink[1] + pink[2] + 0.02 * wn) * 2.0;
        // SR: gunshot-like decaying noise bursts every 250 ms, silence between
        let ph = t % 0.25;
        let sr = if ph < 0.08 {
            0.8 * rnd() * (-ph * 60.0).exp()
        } else {
            0.0
        };
        for s in [fl, fr, fc, lfe, sl, sr] {
            out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
    }
    std::fs::write(path, out).unwrap();
}

#[cfg(feature = "omur")]
fn enc(inp: &str, outp: &str, kbps: u64) {
    let ch = channels();
    let pcm: Vec<f32> = std::fs::read(inp)
        .unwrap()
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();
    let layout = if ch == 2 {
        omur::Layout::Stereo
    } else {
        omur::Layout::Surround51
    };
    let config = omur::Config::new(omur::Codec::Ac3, layout, kbps as u32).unwrap();
    let mut e = omur::Encoder::new(config).unwrap();
    // the tail: silence until the frame holding the last sample is out
    let mut input = pcm;
    input.resize(input.len() + config.delay_samples() * ch, 0.0);
    let frames = input.len() / ch;
    let step = chunk();
    let mut out = Vec::new();
    let (mut slowest, mut slow_at) = (0f64, 0u64);
    let t0 = Instant::now();
    for c in (0..frames).step_by(step) {
        let n = step.min(frames - c);
        // a push may complete one frame, so it carries one frame's samples at most
        for piece in input[c * ch..(c + n) * ch].chunks(config.frame_samples() * ch) {
            let t = Instant::now();
            e.push(piece).unwrap();
            let dt = t.elapsed().as_secs_f64();
            if let Some(frame) = e.pull() {
                if dt > slowest {
                    (slowest, slow_at) = (dt, frame.pts);
                }
                out.extend_from_slice(frame.bytes);
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    std::fs::write(outp, &out).unwrap();
    let secs = (frames - config.delay_samples()) as f64 / SR as f64;
    println!(
        "encoded {secs:.1} s of {ch} ch in {dt:.3} s = {:.0}x realtime, {} bytes; slowest frame \
         {:.0} us (pts {slow_at})",
        secs / dt,
        out.len(),
        slowest * 1e6,
    );
}

#[cfg(not(feature = "omur"))]
fn enc(inp: &str, outp: &str, kbps: u64) {
    let raw = std::fs::read(inp).unwrap();
    let wav: Vec<i16> = raw
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    let frames = wav.len() / CH;
    // WAV (FL FR FC LFE SL SR) -> AC-3 bitstream order (L C R Ls Rs LFE)
    let map = [0usize, 2, 1, 4, 5, 3];
    let mut bytes = Vec::with_capacity(raw.len());
    for f in 0..frames {
        for &c in &map {
            bytes.extend_from_slice(&wav[f * CH + c].to_le_bytes());
        }
    }
    let mut p = CodecParameters::audio(CodecId::new("ac3"));
    p.sample_rate = Some(SR as u32);
    p.channels = Some(6);
    p.sample_format = Some(SampleFormat::S16);
    p.bit_rate = Some(kbps * 1000);
    let mut e = ac3::encoder::make_encoder(&p).unwrap();
    let mut out = Vec::new();
    let t0 = Instant::now();
    // feed in 5 ms chunks, like the audio plane would
    let chunk = chunk();
    for c in (0..frames).step_by(chunk) {
        let n = chunk.min(frames - c);
        let data = bytes[c * CH * 2..(c + n) * CH * 2].to_vec();
        e.send_frame(&Frame::Audio(AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![data],
        }))
        .unwrap();
        loop {
            match e.receive_packet() {
                Ok(pk) => out.extend_from_slice(&pk.data),
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(x) => panic!("{x}"),
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    let _ = e.flush();
    while let Ok(pk) = e.receive_packet() {
        out.extend_from_slice(&pk.data);
    }
    std::fs::write(outp, &out).unwrap();
    println!(
        "encoded {:.1} s of 5.1 in {:.3} s = {:.0}x realtime, {} bytes",
        frames as f64 / SR as f64,
        dt,
        frames as f64 / SR as f64 / dt,
        out.len()
    );
}

fn cmp(refp: &str, decp: &str) {
    let ch_n = channels();
    let r: Vec<f32> = std::fs::read(refp)
        .unwrap()
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();
    let d: Vec<f32> = std::fs::read(decp)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let (nr, nd) = (r.len() / ch_n, d.len() / ch_n);
    let ch = |v: &[f32], c: usize, n: usize| (0..n).map(|i| v[i * ch_n + c]).collect::<Vec<f32>>();
    let names = ["FL", "FR", "FC", "LFE", "SL", "SR"];
    // one lag for the whole stream: search on FL
    let rf = ch(&r, 0, nr);
    let df = ch(&d, 0, nd);
    let win = SR * 4;
    let (mut best, mut lag) = (f64::MIN, 0usize);
    for l in 0..4096 {
        if l + SR + win > nd {
            break;
        }
        let c: f64 = (0..win).map(|i| (rf[SR + i] * df[SR + i + l]) as f64).sum();
        if c > best {
            best = c;
            lag = l;
        }
    }
    println!("lag {lag} samples ({:.1} ms)", lag as f64 / 48.0);
    // channel mapping: which decoded channel correlates best with each ref channel
    let n = (nr.min(nd - lag)) - SR; // skip the last second's tail
    for c in 0..ch_n {
        let rc = ch(&r, c, nr);
        let mut bestd = (0usize, f64::MIN);
        for dc in 0..ch_n {
            let dd = ch(&d, dc, nd);
            let x: f64 = (SR / 2..n).map(|i| (rc[i] * dd[i + lag]) as f64).sum();
            if x > bestd.1 {
                bestd = (dc, x);
            }
        }
        let dd = ch(&d, c, nd);
        let (mut sxx, mut sxy, mut syy) = (0f64, 0f64, 0f64);
        for i in SR / 2..n {
            let (x, y) = (rc[i] as f64, dd[i + lag] as f64);
            sxx += x * x;
            sxy += x * y;
            syy += y * y;
        }
        let g = sxy / sxx; // least-squares gain decoded vs ref
        let err = syy - 2.0 * sxy + sxx; // |d - r|^2
        let errg = syy - 2.0 * g * sxy + g * g * sxx; // |d - g r|^2
        let snr = 10.0 * (sxx / err).log10();
        let snrg = 10.0 * (g * g * sxx / errg).log10();
        println!(
            "{:>3}: best-match ch {:>3}  gain {:+.2} dB  SNR {:5.1} dB  gain-corrected SNR {:5.1} dB",
            names[c],
            names[bestd.0],
            20.0 * g.abs().log10(),
            snr,
            snrg
        );
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "gen" => generate(&a[2]),
        "enc" => enc(&a[2], &a[3], a[4].parse().unwrap()),
        "cmp" => cmp(&a[2], &a[3]),
        "dump" => dump::dump(&a[2], a.get(3).is_some_and(|s| s == "summary")),
        _ => panic!("gen | enc | cmp | dump"),
    }
}
