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
