//! Object audio through the public API (feature `joc`): every frame is a 5.1 E-AC-3
//! syncframe whose `addbsi` signals the extension and whose skip field carries an EMDF
//! container oxideav-ac3 parses; the core decodes in oxideav and in ffmpeg; a dynamic
//! program renders through oxideav's JOC renderer; the bytes do not depend on the chunking;
//! the documented errors hold.

#![cfg(feature = "joc")]

mod common;

use common::ffmpeg_decodes;
use omur::Error;
use omur::joc::{Bed, BedChannel, ObjectConfig, ObjectEncoder, ObjectUpdate};
use oxideav_ac3::audblk::Ac3State;
use oxideav_ac3::decoder::verify_packet_crc;
use oxideav_ac3::eac3::joc::{parse_ec3_extension_type_a, parse_joc_emdf};
use oxideav_ac3::eac3::joc_renderer::JocRenderer;
use oxideav_ac3::eac3::oamd::OamdDecoder;
use oxideav_ac3::eac3::{audfrm, bsi, decoder, dsp};
use oxideav_core::bits::BitReader;

const FRAME: usize = 1536;

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

/// A tone of `f` cycles per 48,000 samples by a recurrence: no libm call, the same bits on
/// every platform.
fn tone(f: f64, amplitude: f32, n: usize) -> Vec<f32> {
    let w = 2.0 * std::f64::consts::PI * f / 48_000.0;
    let k = 2.0 * w.cos() as f32;
    let (mut a, mut b) = (0f32, amplitude * w.sin() as f32);
    (0..n)
        .map(|_| {
            let v = a;
            (a, b) = (b, k * b - a);
            v
        })
        .collect()
}

/// `bed` with a tone per channel, `objects` tones moving on a square path, an update every
/// 480 samples 100 in: interleaved bed, then each object's audio and updates.
struct Scene {
    bed: Bed,
    bed_pcm: Vec<f32>,
    obj: Vec<Vec<f32>>,
    updates: Vec<Vec<(usize, ObjectUpdate)>>,
}

fn scene(bed: Bed, objects: usize, n: usize) -> Scene {
    let nb = bed.channels();
    let voices: Vec<Vec<f32>> = (0..nb)
        .map(|c| tone(375.0 * (1.5 + 2.0 * c as f64), 0.2, n))
        .collect();
    let bed_pcm = (0..n * nb).map(|i| voices[i % nb][i / nb]).collect();
    let obj = (0..objects)
        .map(|o| tone(375.0 * (30.5 + 3.0 * o as f64), 0.1, n))
        .collect();
    let updates = (0..objects)
        .map(|o| {
            (0..n)
                .step_by(480)
                .map(|i| {
                    let t = ((i / 480 + 7 * o) % 64) as f32 / 16.0 - 2.0;
                    let position = [t.clamp(-1.0, 1.0), (-t).clamp(-1.0, 1.0), 0.25 * o as f32];
                    let u = ObjectUpdate {
                        position,
                        gain: 1.0,
                        size: 0.1 * o as f32,
                        at_sample: 0,
                    };
                    (i + 100, u)
                })
                .collect()
        })
        .collect();
    Scene {
        bed,
        bed_pcm,
        obj,
        updates,
    }
}

/// A sine by the recurrence `s[n] = k·s[n−1] − s[n−2]`, `k = 2·cos ω`, from `s[1] = s1`.
fn sine(k: f32, s1: f32, n: usize) -> Vec<f32> {
    let (mut a, mut b) = (0f32, s1);
    (0..n)
        .map(|_| {
            let v = a;
            (a, b) = (b, k * b - a);
            v
        })
        .collect()
}

/// White noise in −1..1 from an LCG seeded with `seed`.
fn lcg(mut seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// A 7.1.4 bed of tones and noise with five objects, four on `scene`'s square path and one
/// jumping corner to corner with its gain: no libm call, the same bits on every platform.
fn pinned_mix(n: usize) -> Scene {
    let bed = bed_714();
    let tones = [
        (1.9991708, 0.0043190927),
        (1.9981344, 0.00647752),
        (1.995315, 0.0082088495),
        (1.9999572, 0.0019634813),
        (1.9916098, 0.010980194),
        (1.9867455, 0.013792458),
        (1.9732866, 0.016289547),
        (1.9471579, 0.022835087),
        (1.8938602, 0.025715157),
        (1.8162863, 0.033492777),
        (1.6773411, 0.032678343),
        (std::f32::consts::SQRT_2, 0.042426407),
    ];
    let voices: Vec<Vec<f32>> = tones.iter().map(|(k, s)| sine(*k, *s, n)).collect();
    let hiss = lcg(0x9e37_79b9, n);
    let bed_pcm = (0..n * 12)
        .map(|i| {
            voices[i % 12][i / 12]
                + if i % 12 == 4 {
                    0.05 * hiss[i / 12]
                } else {
                    0.0
                }
        })
        .collect();
    let noise = lcg(0x1234_5678, n);
    let obj = vec![
        sine(0.76536685, 0.09238795, n),
        sine(0.26105237, 0.07931559, n),
        noise.iter().map(|v| 0.1 * v).collect(),
        sine(1.9938346, 0.007845909, n)
            .iter()
            .zip(&noise)
            .map(|(t, v)| t - 0.03 * v)
            .collect(),
        sine(-0.76536685, 0.046193976, n),
    ];
    let updates = (0..5)
        .map(|o| {
            (0..n)
                .step_by(480)
                .map(|i| {
                    let k = i / 480;
                    let t = ((k + 7 * o) % 64) as f32 / 16.0 - 2.0;
                    let mut u = ObjectUpdate {
                        position: [t.clamp(-1.0, 1.0), (-t).clamp(-1.0, 1.0), 0.25 * o as f32],
                        gain: 1.0,
                        size: 0.1 * o as f32,
                        at_sample: 0,
                    };
                    if o == 4 {
                        let corner = (k / 10) % 4;
                        let x = if corner % 2 == 0 { -1.0 } else { 1.0 };
                        let y = if corner < 2 { 1.0 } else { -1.0 };
                        u.position = [x, y, (corner % 2) as f32];
                        u.gain = if corner == 3 { 0.5 } else { 1.0 };
                    }
                    (i + 100, u)
                })
                .collect()
        })
        .collect();
    Scene {
        bed,
        bed_pcm,
        obj,
        updates,
    }
}

/// A stereo bed and 32 noise objects on a grid, the odd ones 14 dB down and drifting: the
/// fold into 14 dynamic elements.
fn pinned_crowd(n: usize) -> Scene {
    let bed = Bed::new(&[BedChannel::FrontLeft, BedChannel::FrontRight]);
    let (l, r) = (
        sine(1.9991708, 0.0043190927, n),
        sine(1.995315, 0.0082088495, n),
    );
    let bed_pcm = (0..2 * n)
        .map(|i| if i % 2 == 0 { l[i / 2] } else { r[i / 2] })
        .collect();
    let obj = (0..32u32)
        .map(|o| {
            let level = if o % 2 == 0 { 0.1 } else { 0.02 };
            lcg(0x5555_0000 + o, n).iter().map(|v| level * v).collect()
        })
        .collect();
    let updates = (0..32usize)
        .map(|o| {
            (0..n)
                .step_by(2400)
                .map(|i| {
                    let drift = if o % 2 == 1 {
                        (i / 2400) as f32 * 0.125
                    } else {
                        0.0
                    };
                    let x = ((o % 8) as f32 / 3.5 - 1.0 + drift).clamp(-1.0, 1.0);
                    let y = (o / 8) as f32 / 1.5 - 1.0;
                    let u = ObjectUpdate {
                        position: [x, y, (o % 3) as f32 * 0.4],
                        gain: 1.0,
                        size: 0.0,
                        at_sample: 0,
                    };
                    (i, u)
                })
                .collect()
        })
        .collect();
    Scene {
        bed,
        bed_pcm,
        obj,
        updates,
    }
}

/// Two noise objects without a bed, one on `scene`'s square path, one jumping corner to
/// corner every 50 ms, at the floor and the ceiling in turn: the steep data points.
fn pinned_jumps(n: usize) -> Scene {
    let obj = vec![
        lcg(0x1111_2222, n).iter().map(|v| 0.3 * v).collect(),
        lcg(0x3333_4444, n).iter().map(|v| 0.3 * v).collect(),
    ];
    let update = |position| ObjectUpdate {
        position,
        gain: 1.0,
        size: 0.0,
        at_sample: 0,
    };
    let path = (0..n)
        .step_by(480)
        .map(|i| {
            let t = ((i / 480) % 64) as f32 / 16.0 - 2.0;
            (i, update([t.clamp(-1.0, 1.0), (-t).clamp(-1.0, 1.0), 0.0]))
        })
        .collect();
    let jumps = (0..n)
        .step_by(2400)
        .enumerate()
        .map(|(k, i)| {
            let x = if k % 2 == 0 { 1.0 } else { -1.0 };
            let y = if k % 4 < 2 { -1.0 } else { 1.0 };
            (i + 700, update([x, y, (k % 2) as f32]))
        })
        .collect();
    Scene {
        bed: Bed::new(&[]),
        bed_pcm: vec![],
        obj,
        updates: vec![path, jumps],
    }
}

/// A 7.1.4 bed of steady tones, each channel's in a QMF band of its own with an empty one
/// between, and five tones in high bands on `scene`'s square path: in most bands one core
/// channel carries the element, which sparse frames code.
fn pinned_bands(n: usize) -> Scene {
    let voices = [
        (1.9945809, 0.011034684),
        (1.9705553, 0.025644284),
        (1.9275521, 0.040006913),
        (1.9999726, 0.0015707891),
        (1.8659856, 0.053984255),
        (1.7864486, 0.0674417),
        (1.6897072, 0.080249645),
        (1.5766928, 0.09228474),
        (1.4484942, 0.10343108),
        (1.3063457, 0.11358133),
        (1.1516163, 0.12263772),
        (0.9857964, 0.13051304),
    ]
    .map(|(k, s)| sine(k, s, n));
    let bed_pcm = (0..n * 12).map(|i| voices[i % 12][i / 12]).collect();
    let obj = [
        (0.7197901, 0.09329928),
        (0.14712913, 0.099729046),
        (-0.43820247, 0.09757021),
        (-0.9857964, 0.0870087),
        (-1.4484942, 0.06895406),
    ]
    .iter()
    .map(|(k, s)| sine(*k, *s, n))
    .collect();
    let updates = (0..5)
        .map(|o| {
            (0..n)
                .step_by(480)
                .map(|i| {
                    let t = ((i / 480 + 7 * o) % 64) as f32 / 16.0 - 2.0;
                    let u = ObjectUpdate {
                        position: [t.clamp(-1.0, 1.0), (-t).clamp(-1.0, 1.0), 0.25 * o as f32],
                        gain: 1.0,
                        size: 0.0,
                        at_sample: 0,
                    };
                    (i + 100, u)
                })
                .collect()
        })
        .collect();
    Scene {
        bed: bed_714(),
        bed_pcm,
        obj,
        updates,
    }
}

/// FNV-1a, 64 bits.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Encodes `s` in `chunk`-sample chunks, then a frame and its lookahead of silence.
fn encode(s: &Scene, kbps: u32, chunk: usize) -> Vec<Vec<u8>> {
    let config = ObjectConfig {
        bed: s.bed,
        objects: s.obj.len() as u8,
        bitrate_kbps: kbps,
        blocks: omur::Blocks::Six,
    };
    let mut enc = ObjectEncoder::new(config).unwrap();
    let nb = s.bed.channels();
    let n = s
        .bed_pcm
        .len()
        .checked_div(nb)
        .unwrap_or_else(|| s.obj[0].len());
    let total = n + FRAME + 256;
    let mut frames = Vec::new();
    let mut c = 0;
    while c < total {
        let len = chunk.min(total - c);
        let bed: Vec<f32> = (c * nb..(c + len) * nb)
            .map(|i| s.bed_pcm.get(i).copied().unwrap_or(0.0))
            .collect();
        enc.push_bed(&bed).unwrap();
        for (o, pcm) in s.obj.iter().enumerate() {
            let piece: Vec<f32> = (c..c + len)
                .map(|i| pcm.get(i).copied().unwrap_or(0.0))
                .collect();
            let update = s.updates[o]
                .iter()
                .find(|(at, _)| (c..c + len).contains(at))
                .map(|(at, u)| ObjectUpdate {
                    at_sample: (at - c) as u32,
                    ..*u
                });
            enc.push_object(o as u8, &piece, update).unwrap();
        }
        if let Some(f) = enc.pull() {
            assert_eq!((f.syncframes.len(), f.bytes.len()), (1, 4 * kbps as usize));
            frames.push(f.bytes.to_vec());
        }
        c += len;
    }
    assert_eq!(frames.len(), (total - 256) / FRAME);
    frames
}

/// Checks every frame as oxideav reads it; returns per frame the printed JOC metadata and
/// OAMD, and the decoded core in bitstream order L C R Ls Rs LFE.
fn check(frames: &[Vec<u8>], objects: usize) -> (Vec<String>, Vec<String>, Vec<f32>) {
    let mut state = decoder::Eac3DecoderState::default();
    let mut oamd = OamdDecoder::default();
    let (mut joc, mut meta, mut core) = (vec![], vec![], vec![]);
    for (k, f) in frames.iter().enumerate() {
        assert!(verify_packet_crc(f).unwrap().all_ok(), "frame {k}");
        let b = bsi::parse(&f[2..]).unwrap();
        assert_eq!((b.bsid, b.acmod, b.lfeon, b.num_blocks), (16, 7, true, 6));
        let addbsi = b.addbsi.as_ref().unwrap().payload();
        assert_eq!(addbsi, [1, objects as u8]);
        let signal = parse_ec3_extension_type_a(addbsi).unwrap().unwrap();
        let mut br = BitReader::new(&f[2..]);
        br.skip(b.bits_consumed as u32).unwrap();
        let a = audfrm::parse_with(&mut br, &b).unwrap();
        let mut out = vec![0f32; FRAME * 6];
        let mut skips = vec![];
        dsp::decode_indep_audblks(&b, &a, &mut br, &mut Ac3State::new(), &mut out, &mut skips)
            .unwrap();
        let field = skips.iter().find(|s| s.starts_with(&[0x58, 0x38])).unwrap();
        let m = parse_joc_emdf(field, signal).unwrap_or_else(|e| panic!("frame {k}: {e}"));
        let dbg = format!("{m:?}");
        assert!(
            dbg.contains(&format!("num_objects: {objects},")),
            "frame {k}"
        );
        let at = dbg.find("oamd_payload: [").unwrap() + 15;
        let payload: Vec<u8> = dbg[at..dbg[at..].find(']').unwrap() + at]
            .split(", ")
            .map(|v| v.parse().unwrap())
            .collect();
        let o = oamd
            .decode(&payload)
            .unwrap_or_else(|e| panic!("frame {k}: {e}"));
        decoder::decode_eac3_packet(&mut state, f).unwrap();
        assert_eq!(state.frames_zero_filled, 0, "frame {k}");
        core.extend_from_slice(state.indep_pcm_f32());
        joc.push(dbg);
        meta.push(format!("{o:?}"));
    }
    (joc, meta, core)
}

#[test]
fn a_714_bed_and_objects_make_a_valid_stream() {
    for (objects, kbps) in [(0, 768), (5, 768), (5, 448), (2, 256)] {
        let s = scene(bed_714(), objects, 6 * FRAME);
        let frames = encode(&s, kbps, 240);
        let (joc, meta, core) = check(&frames, 11 + objects);
        // the sequence counter starts at 0 and counts frames
        for (k, j) in joc.iter().enumerate() {
            assert!(
                j.contains(&format!("sequence_count: {k},")),
                "{kbps}: frame {k}"
            );
        }
        assert!(meta[0].contains("kinds: [Bed, Bed, Bed, Bed, Bed, Bed, Bed, Bed, Lfe, Bed"));
        assert_eq!(meta[0].matches("Dynamic").count(), objects);
        // the core: the front left tone, out of L
        let peak = core.iter().step_by(6).fold(0f32, |m, v| m.max(v.abs()));
        assert!((0.15..0.6).contains(&peak), "{kbps}: {peak}");
        ffmpeg_decodes(
            &frames.concat(),
            &format!("objects-714-{objects}-{kbps}.ec3"),
        );
    }
}

#[test]
fn a_dynamic_program_renders_in_oxideavs_joc_renderer() {
    let lfe = Bed::new(&[BedChannel::Lfe]);
    for bed in [Bed::new(&[]), lfe] {
        let s = scene(bed, 3, 5 * FRAME);
        let frames = encode(&s, 640, 160);
        let mut state = decoder::Eac3DecoderState::default();
        let mut renderer = JocRenderer::default();
        for f in &frames {
            let b = bsi::parse(&f[2..]).unwrap();
            let signal = parse_ec3_extension_type_a(b.addbsi.as_ref().unwrap().payload())
                .unwrap()
                .unwrap();
            let mut br = BitReader::new(&f[2..]);
            br.skip(b.bits_consumed as u32).unwrap();
            let a = audfrm::parse_with(&mut br, &b).unwrap();
            let (mut out, mut skips) = (vec![0f32; FRAME * 6], vec![]);
            dsp::decode_indep_audblks(&b, &a, &mut br, &mut Ac3State::new(), &mut out, &mut skips)
                .unwrap();
            let field = skips.iter().find(|s| s.starts_with(&[0x58, 0x38])).unwrap();
            let meta = parse_joc_emdf(field, signal).unwrap();
            decoder::decode_eac3_packet(&mut state, f).unwrap();
            let stereo = renderer
                .render_stereo(&meta, state.indep_pcm_f32(), 6, 7, true)
                .unwrap();
            assert!(stereo.iter().all(|v| v.is_finite() && v.abs() < 4.0));
        }
        ffmpeg_decodes(
            &frames.concat(),
            &format!("objects-dynamic-{}.ec3", bed.channels()),
        );
    }
}

/// The object output, pinned like `tests/golden.rs`: a change to either hash is a change to
/// the bitstream, made on purpose and named in its commit.
#[test]
fn object_output_is_unchanged() {
    let cases = [
        ("mix", pinned_mix(12 * FRAME), 0x0f28_181f_1af8_7abfu64),
        ("crowd", pinned_crowd(8 * FRAME), 0x1548_9232_7be0_22c6),
        ("jumps", pinned_jumps(8 * FRAME), 0xf3c8_94b5_b2aa_fefd),
        ("bands", pinned_bands(12 * FRAME), 0x3f7d_f296_f481_c27d),
    ];
    for (name, s, want) in cases {
        let got = fnv1a(encode(&s, 768, 240).concat());
        assert_eq!(got, want, "{name}: {got:#018x}");
    }
}

#[test]
fn output_does_not_depend_on_the_chunking() {
    let s = scene(bed_714(), 5, 5 * FRAME);
    let want = encode(&s, 768, 240);
    for chunk in [480, 160, 96, 32, 1] {
        assert!(encode(&s, 768, chunk) == want, "chunk {chunk}");
    }
}

#[test]
fn the_documented_errors_hold() {
    let config = ObjectConfig {
        bed: bed_714(),
        objects: 2,
        bitrate_kbps: 768,
        blocks: omur::Blocks::Six,
    };
    let mut enc = ObjectEncoder::new(config).unwrap();
    assert_eq!(enc.push_object(0, &[0.0; 4], None), Err(Error::Length));
    assert_eq!(enc.push_bed(&[0.0; 13]), Err(Error::Length));
    assert_eq!(
        enc.push_bed(&vec![0.0; 12 * (FRAME + 1)]),
        Err(Error::Length)
    );
    enc.push_bed(&[0.0; 12 * 4]).unwrap();
    assert_eq!(
        enc.push_object(2, &[0.0; 4], None),
        Err(Error::Object { index: 2 })
    );
    assert_eq!(enc.push_object(1, &[0.0; 5], None), Err(Error::Length));
    enc.push_object(1, &[0.5; 4], None).unwrap();
    assert!(enc.pull().is_none());
    // chunks of a frame, never pulled: the third frame's chunk waits behind the second
    let mut enc = ObjectEncoder::new(config).unwrap();
    let bed = vec![0.1; 12 * FRAME];
    enc.push_bed(&bed).unwrap();
    enc.push_bed(&bed).unwrap();
    enc.push_bed(&bed).unwrap(); // encodes the second chunk: frame 0 completes
    assert_eq!(enc.push_bed(&bed), Err(Error::Backlog));
    assert_eq!(enc.pull().map(|f| f.pts), Some(0));
    assert_eq!(enc.pull().map(|f| f.pts), Some(1536));
    enc.push_bed(&bed).unwrap();
    // an ended object is silent until pushed again, and inactive in the metadata
    let mut enc = ObjectEncoder::new(config).unwrap();
    let mut frames = vec![];
    for k in 0..40 {
        enc.push_bed(&vec![0.0; 12 * 240]).unwrap();
        if k < 10 {
            let u = (k == 0).then_some(ObjectUpdate {
                position: [0.5, 0.5, 0.0],
                gain: 1.0,
                size: 0.0,
                at_sample: 0,
            });
            enc.push_object(1, &tone(1000.0, 0.3, 240), u).unwrap();
        } else if k == 10 {
            enc.end_object(1);
        }
        if let Some(f) = enc.pull() {
            frames.push(f.bytes.to_vec());
        }
    }
    let (_, meta, _) = check(&frames, 13);
    // (0.5, 0.5) in the cube is (47, 16) on the room's grid; the last object's last state
    let last = |m: &String| m.rsplit("OamdObjectState {").next().unwrap().to_string();
    assert!(last(&meta[0]).contains("active: true") && meta[0].contains("[47, 16, 0]"));
    assert!(last(meta.last().unwrap()).contains("active: false"));
}

#[test]
fn hostile_input_and_updates_never_panic() {
    let config = ObjectConfig {
        bed: Bed::new(&[BedChannel::FrontLeft, BedChannel::BackCenter]),
        objects: 32,
        bitrate_kbps: 256,
        blocks: omur::Blocks::Six,
    };
    let mut enc = ObjectEncoder::new(config).unwrap();
    let hostile = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e30,
        -1e30,
        f32::MAX,
        1e-40,
    ];
    let mut frames = vec![];
    for k in 0..60 {
        let v = hostile[k % hostile.len()];
        enc.push_bed(&[v; 2 * 100]).unwrap();
        for o in 0..32 {
            let u = ObjectUpdate {
                position: [v, -v, hostile[(k + o) % hostile.len()]],
                gain: hostile[(k + 2 * o) % hostile.len()],
                size: v,
                at_sample: u32::MAX,
            };
            enc.push_object(o as u8, &[hostile[(k + o) % 7]; 100], Some(u))
                .unwrap();
        }
        if let Some(f) = enc.pull() {
            frames.push(f.bytes.to_vec());
        }
    }
    check(&frames, 16);
}
