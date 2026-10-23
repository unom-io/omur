//! `scene <dir> [kbps] [seconds]`: the synthetic object-audio scenes of docs/design.md §8.5,
//! written as raw inputs and encoded through omur's ObjectEncoder, for the receiver on glass.
//! Per scene: `<name>.bed.f32` (interleaved, BedChannel order), `<name>.obj<i>.f32` (mono),
//! `<name>.updates.csv`, the stream `<name>.ec3`, its rendered core `<name>.core.s16` (5.1,
//! WAV order) for `cmp`, and that core through the plain 5.1 encoder: `<name>.p2.ec3` at the
//! same rate, `<name>.p2x.ec3` with each frame's container bytes taken from it as zeros.
//! `SCENE=<name>` keeps one: objects, bed, crowd or mix.
use std::f64::consts::PI;
use std::fmt::Write as _;
use std::time::Instant;

use omur::joc::{Bed, BedChannel, ObjectConfig, ObjectEncoder, ObjectUpdate};
use oxideav_ac3::audblk::Ac3State;
use oxideav_ac3::eac3::{audfrm, bsi, dsp};
use oxideav_core::bits::BitReader;

const SR: usize = 48_000;

struct Scene {
    name: &'static str,
    bed: Bed,
    bed_pcm: Vec<f32>,
    obj: Vec<Vec<f32>>,
    updates: Vec<Vec<(usize, ObjectUpdate)>>,
}

fn noise(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

fn pink(p: &mut [f32; 3], w: f32) -> f32 {
    p[0] = 0.997 * p[0] + 0.029 * w;
    p[1] = 0.985 * p[1] + 0.032 * w;
    p[2] = 0.950 * p[2] + 0.048 * w;
    (p[0] + p[1] + p[2] + 0.02 * w) * 2.0
}

fn update(position: [f32; 3]) -> ObjectUpdate {
    ObjectUpdate {
        position,
        gain: 1.0,
        size: 0.0,
        at_sample: 0,
    }
}

fn bed_714() -> Bed {
    use BedChannel::*;
    Bed::new(&[
        FrontLeft,
        FrontRight,
        FrontCenter,
        Lfe,
        SideLeft,
        SideRight,
        BackLeft,
        BackRight,
        TopFrontLeft,
        TopFrontRight,
        TopBackLeft,
        TopBackRight,
    ])
}

/// Eleven formant-like voices, each a vibrato tone in a QMF subband of its own, swelling
/// at a syllable rate of its own, and 40 Hz on the LFE: 7.1.4 in BedChannel order.
fn voices(n: usize, level: f64) -> Vec<f32> {
    let mut pcm = Vec::with_capacity(n * 12);
    for i in 0..n {
        let t = i as f64 / SR as f64;
        let mut k = 0.0;
        for c in 0..12 {
            if c == 3 {
                pcm.push((0.3 * (2.0 * PI * 40.0 * t).sin()) as f32);
                continue;
            }
            let f = (1.0 + 2.0 * k + 0.5) * 375.0;
            let syllable = 0.5 - 0.5 * (2.0 * PI * (3.0 + 0.4 * k) * t + k).cos();
            let phase = 2.0 * PI * f * t - 3.0 * (2.0 * PI * 5.0 * t + k).cos();
            pcm.push((level * syllable * phase.sin()) as f32);
            k += 1.0;
        }
    }
    pcm
}

fn scenes(seconds: f64) -> Vec<Scene> {
    let n = (seconds * SR as f64) as usize;
    let only = std::env::var("SCENE").ok();
    let want = |name: &str| only.as_deref().is_none_or(|o| o == name);
    let mut out = Vec::new();

    // a pink-noise object circling every four seconds, a second jumping every half second
    // to the corner farthest from it, floor and ceiling in turn
    if want("objects") {
        let (mut s1, mut s2) = (0x1111_2222u32, 0x3333_4444u32);
        let (mut p1, mut p2) = ([0f32; 3], [0f32; 3]);
        let a = (0..n)
            .map(|_| 0.5 * pink(&mut p1, noise(&mut s1)))
            .collect();
        let b = (0..n)
            .map(|_| 0.5 * pink(&mut p2, noise(&mut s2)))
            .collect();
        let at = |i: usize| {
            let phi = 2.0 * PI * i as f64 / (4.0 * SR as f64);
            [phi.sin() as f32, phi.cos() as f32]
        };
        let circle = (0..n)
            .step_by(480)
            .map(|i| {
                let [x, y] = at(i);
                (i, update([x, y, 0.0]))
            })
            .collect();
        let jumps = (0..n)
            .step_by(SR / 2)
            .enumerate()
            .map(|(k, i)| {
                let [x, y] = at(i + SR / 4);
                (i + 100, update([-x.signum(), -y.signum(), (k % 2) as f32]))
            })
            .collect();
        out.push(Scene {
            name: "objects",
            bed: Bed::new(&[]),
            bed_pcm: vec![],
            obj: vec![a, b],
            updates: vec![circle, jumps],
        });
    }
    if want("bed") {
        out.push(Scene {
            name: "bed",
            bed: bed_714(),
            bed_pcm: voices(n, 0.25),
            obj: vec![],
            updates: vec![],
        });
    }

    // 32 objects, tone `o` in QMF subband `o`, the even ones 14 dB louder, the odd drifting
    if want("crowd") {
        let mut obj = Vec::new();
        let mut updates = Vec::new();
        for o in 0..32u32 {
            let f = (f64::from(o) + 0.5) * 375.0;
            let level = if o % 2 == 0 { 0.15 } else { 0.03 };
            obj.push(
                (0..n)
                    .map(|i| (level * (2.0 * PI * f * i as f64 / SR as f64).sin()) as f32)
                    .collect(),
            );
            let base = 2.0 * PI * f64::from(o * 7 % 32) / 32.0;
            updates.push(
                (0..n)
                    .step_by(4_800)
                    .map(|i| {
                        let drift = if o % 2 == 1 { i as f64 / 96_000.0 } else { 0.0 };
                        let phi = base + drift;
                        let z = (o % 3) as f32 * 0.4;
                        (i, update([phi.sin() as f32, phi.cos() as f32, z]))
                    })
                    .collect(),
            );
        }
        out.push(Scene {
            name: "crowd",
            bed: Bed::new(&[]),
            bed_pcm: vec![],
            obj,
            updates,
        });
    }

    // the speed case: the 7.1.4 voices and five objects circling at their own heights
    if want("mix") {
        let mut obj = Vec::new();
        let mut updates = Vec::new();
        for o in 0..5u32 {
            let f = (24.0 + 6.0 * f64::from(o) + 0.5) * 375.0;
            obj.push(
                (0..n)
                    .map(|i| (0.1 * (2.0 * PI * f * i as f64 / SR as f64).sin()) as f32)
                    .collect(),
            );
            let speed = 1.0 + f64::from(o) * 0.5;
            updates.push(
                (0..n)
                    .step_by(480)
                    .map(|i| {
                        let phi = 2.0 * PI * speed * i as f64 / (4.0 * SR as f64) + f64::from(o);
                        let z = o as f32 * 0.25;
                        (i, update([phi.sin() as f32, phi.cos() as f32, z]))
                    })
                    .collect(),
            );
        }
        out.push(Scene {
            name: "mix",
            bed: bed_714(),
            bed_pcm: voices(n, 0.15),
            obj,
            updates,
        });
    }
    out
}

fn write_f32(path: &str, v: &[f32]) {
    std::fs::write(
        path,
        v.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();
}

/// Bytes of the EMDF container in the frame's skip fields.
fn container_bytes(frame: &[u8]) -> usize {
    let Ok(b) = bsi::parse(&frame[2..]) else {
        return 0;
    };
    let mut br = BitReader::new(&frame[2..]);
    if br.skip(b.bits_consumed as u32).is_err() {
        return 0;
    }
    let Ok(a) = audfrm::parse_with(&mut br, &b) else {
        return 0;
    };
    let mut out = vec![0f32; 1536 * 6];
    let mut skips = vec![];
    let _ = dsp::decode_indep_audblks(&b, &a, &mut br, &mut Ac3State::new(), &mut out, &mut skips);
    skips
        .iter()
        .filter(|s| s.starts_with(&[0x58, 0x38]))
        .map(Vec::len)
        .sum()
}

pub fn scene(dir: &str, kbps: u32, seconds: f64) {
    std::fs::create_dir_all(dir).unwrap();
    let chunk = super::chunk();
    let only = std::env::var("SCENE").ok();
    for s in scenes(seconds) {
        if only.as_deref().is_some_and(|o| o != s.name) {
            continue;
        }
        let base = format!("{dir}/{}", s.name);
        let nb = s.bed.channels();
        write_f32(&format!("{base}.bed.f32"), &s.bed_pcm);
        let mut csv = String::from("sample,object,x,y,z,gain,size\n");
        for (o, pcm) in s.obj.iter().enumerate() {
            write_f32(&format!("{base}.obj{o}.f32"), pcm);
            for (at, u) in &s.updates[o] {
                let [x, y, z] = u.position;
                writeln!(csv, "{at},{o},{x},{y},{z},{},{}", u.gain, u.size).unwrap();
            }
        }
        std::fs::write(format!("{base}.updates.csv"), csv).unwrap();

        let config = ObjectConfig {
            bed: s.bed,
            objects: s.obj.len() as u8,
            bitrate_kbps: kbps,
            blocks: omur::Blocks::Six,
        };
        let mut enc = ObjectEncoder::new(config).unwrap();
        let len = s
            .bed_pcm
            .len()
            .checked_div(nb)
            .unwrap_or_else(|| s.obj[0].len());
        let total = len + 1536 + 256;
        let (mut stream, mut core) = (Vec::new(), Vec::new());
        let (mut busy, mut slowest) = (0f64, 0f64);
        let mut c = 0;
        while c < total {
            let n = chunk.min(total - c);
            let bed: Vec<f32> = (c * nb..(c + n) * nb)
                .map(|i| s.bed_pcm.get(i).copied().unwrap_or(0.0))
                .collect();
            let pieces: Vec<(Vec<f32>, Option<ObjectUpdate>)> = s
                .obj
                .iter()
                .zip(&s.updates)
                .map(|(pcm, ups)| {
                    let piece = (c..c + n)
                        .map(|i| pcm.get(i).copied().unwrap_or(0.0))
                        .collect();
                    let u = ups
                        .iter()
                        .find(|(at, _)| (c..c + n).contains(at))
                        .map(|(at, u)| ObjectUpdate {
                            at_sample: (at - c) as u32,
                            ..*u
                        });
                    (piece, u)
                })
                .collect();
            let t = Instant::now();
            enc.push_bed(&bed).unwrap();
            for (o, (piece, u)) in pieces.iter().enumerate() {
                enc.push_object(o as u8, piece, *u).unwrap();
            }
            let frame = enc.pull().map(|f| f.bytes.to_vec());
            let dt = t.elapsed().as_secs_f64();
            busy += dt;
            slowest = slowest.max(dt);
            if let Some(f) = frame {
                stream.extend(f);
            }
            core.extend_from_slice(enc.core_pcm());
            c += n;
        }
        std::fs::write(format!("{base}.ec3"), &stream).unwrap();
        let s16: Vec<u8> = core
            .iter()
            .flat_map(|v| ((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())
            .collect();
        std::fs::write(format!("{base}.core.s16"), s16).unwrap();

        // the same core through the plain 5.1 encoder, as is and less the container's bytes
        let p2 = omur::Config::new(
            omur::Codec::Eac3 {
                blocks: omur::Blocks::Six,
            },
            omur::Layout::Surround51,
            kbps,
        )
        .unwrap();
        let sizes: Vec<usize> = stream
            .chunks(p2.frame_bytes())
            .map(container_bytes)
            .collect();
        for (suffix, taken) in [("p2", false), ("p2x", true)] {
            let mut plain = omur::Encoder::new(p2).unwrap();
            let mut reference = Vec::new();
            let zeros = [0u8; 511];
            let mut k = 0;
            if taken {
                plain.set_extension(&zeros[..sizes[0]]).unwrap();
            }
            for piece in core.chunks(chunk * 6) {
                plain.push(piece).unwrap();
                if let Some(f) = plain.pull() {
                    reference.extend_from_slice(f.bytes);
                    k += 1;
                    if taken && let Some(n) = sizes.get(k) {
                        plain.set_extension(&zeros[..*n]).unwrap();
                    }
                }
            }
            std::fs::write(format!("{base}.{suffix}.ec3"), &reference).unwrap();
        }

        let frames = stream.len() / p2.frame_bytes();
        let emdf: usize = sizes.iter().sum();
        let secs = len as f64 / SR as f64;
        println!(
            "{:>7}: {} bed ch, {} objects, {frames} frames, {:.0}x realtime, slowest chunk \
             {:.0} us, EMDF {:.0} bytes a frame ({:.1} kbit/s)",
            s.name,
            nb,
            s.obj.len(),
            secs / busy,
            slowest * 1e6,
            emdf as f64 / frames.max(1) as f64,
            emdf as f64 * 8.0 / secs / 1000.0,
        );
    }
}
