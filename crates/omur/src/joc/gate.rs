//! The P3 gate (docs/design.md §8.5, §11) through oxideav-ac3: every frame's core decodes,
//! its `addbsi` signals the extension, its skip field holds an EMDF container whose OAMD and
//! JOC payloads oxideav parses. Its object reconstruction is crate-private, so the
//! elements are rebuilt here from what it parsed and dequantised: oxideav's decoded core
//! through the QMF analysis, the matrices interpolated by Pseudocode 6 and applied by
//! Pseudocode 7, the objects through the synthesis of clause 7.3 in the matrix form oxideav
//! verifies. A one-object scene checks that rebuild against oxideav's own stereo render.

use super::*;
use oxideav_ac3::audblk::Ac3State;
use oxideav_ac3::eac3::joc::{parse_ec3_extension_type_a, parse_joc_emdf};
use oxideav_ac3::eac3::joc_renderer::JocRenderer;
use oxideav_ac3::eac3::oamd::OamdDecoder;
use oxideav_ac3::eac3::{audfrm, bsi, decoder, dsp};
use oxideav_core::bits::BitReader;
use std::f64::consts::PI;

/// Samples from an element's sample to the rebuilt object's: QMF analysis and synthesis.
const DELAY: usize = 577;

/// White noise in −1..1.
pub(super) fn noise(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

/// Pinkish noise from white, filter state in `p`.
pub(super) fn pink(p: &mut [f32; 3], w: f32) -> f32 {
    p[0] = 0.997 * p[0] + 0.029 * w;
    p[1] = 0.985 * p[1] + 0.032 * w;
    p[2] = 0.950 * p[2] + 0.048 * w;
    (p[0] + p[1] + p[2] + 0.02 * w) * 2.0
}

/// A scene as a host would push it: the bed interleaved, each object's audio and its
/// updates at absolute samples.
pub(super) struct Scene {
    pub(super) bed: Bed,
    pub(super) objects: u8,
    pub(super) bed_pcm: Vec<f32>,
    pub(super) obj: Vec<Vec<f32>>,
    pub(super) updates: Vec<Vec<(usize, ObjectUpdate)>>,
}

impl Scene {
    pub(super) fn len(&self) -> usize {
        match self.bed.channels() {
            0 => self.obj.first().map_or(0, Vec::len),
            c => self.bed_pcm.len() / c,
        }
    }
}

/// The 7.1.4 bed: eleven formant-like voices, each a vibrato tone in a QMF subband of its
/// own with an empty one between, swelling at a syllable rate of its own, and 40 Hz on the
/// LFE. Every voice is narrow enough that a band holds one, so a matrix can split them.
pub(super) fn speech_bed(seconds: f32) -> Scene {
    use BedChannel::*;
    let bed = Bed::new(&[
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
    ]);
    let n = (seconds * 48_000.0) as usize;
    let mut pcm = Vec::with_capacity(n * 12);
    for i in 0..n {
        let t = i as f64 / 48_000.0;
        let mut k = 0;
        for c in 0..12 {
            if c == 3 {
                pcm.push((0.3 * (2.0 * PI * 40.0 * t).sin()) as f32);
                continue;
            }
            let f = (f64::from(1 + 2 * k as u32) + 0.5) * 375.0;
            let kf = f64::from(k as u32);
            let syllable = 0.5 - 0.5 * (2.0 * PI * (3.0 + 0.4 * kf) * t + kf).cos();
            let phase = 2.0 * PI * f * t - 3.0 * (2.0 * PI * 5.0 * t + kf).cos();
            pcm.push((0.25 * syllable * phase.sin()) as f32);
            k += 1;
        }
    }
    Scene {
        bed,
        objects: 0,
        bed_pcm: pcm,
        obj: vec![],
        updates: vec![],
    }
}

/// Two pink-noise objects without a bed: one circles the listener every four seconds, an
/// update every 10 ms; the other jumps every half second to the cube corner farthest from it,
/// at the floor and the ceiling in turn. Two broadband sources a matrix cannot split where
/// their pans coincide, so the jumps keep them apart.
pub(super) fn circling(seconds: f32) -> Scene {
    let n = (seconds * 48_000.0) as usize;
    let (mut s1, mut s2) = (0x1111_2222u32, 0x3333_4444u32);
    let (mut p1, mut p2) = ([0f32; 3], [0f32; 3]);
    let a: Vec<f32> = (0..n)
        .map(|_| 0.5 * pink(&mut p1, noise(&mut s1)))
        .collect();
    let b: Vec<f32> = (0..n)
        .map(|_| 0.5 * pink(&mut p2, noise(&mut s2)))
        .collect();
    let update = |position, at| ObjectUpdate {
        position,
        gain: 1.0,
        size: 0.0,
        at_sample: at,
    };
    let at = |i: usize| {
        let phi = 2.0 * PI * i as f64 / (4.0 * 48_000.0);
        [phi.sin() as f32, phi.cos() as f32]
    };
    let circle = (0..n)
        .step_by(480)
        .map(|i| {
            let [x, y] = at(i);
            (i, update([x, y, 0.0], 0))
        })
        .collect();
    let jumps = (0..n)
        .step_by(24_000)
        .enumerate()
        .map(|(k, i)| {
            let [x, y] = at(i + 12_000);
            let corner = [-x.signum(), -y.signum(), (k % 2) as f32];
            (i + 100, update(corner, 0))
        })
        .collect();
    Scene {
        bed: Bed::new(&[]),
        objects: 2,
        bed_pcm: vec![],
        obj: vec![a, b],
        updates: vec![circle, jumps],
    }
}

/// 32 objects without a bed, twice the 16 elements: object `o` a tone in QMF subband `o`,
/// the even ones 14 dB louder, scattered over a ring at three heights, the odd ones
/// drifting round it. A band holds four objects at most, at different places.
pub(super) fn crowd(seconds: f32) -> Scene {
    let n = (seconds * 48_000.0) as usize;
    let mut obj = Vec::new();
    let mut updates = Vec::new();
    for o in 0..32u32 {
        let f = (f64::from(o) + 0.5) * 375.0;
        let level = if o % 2 == 0 { 0.15 } else { 0.03 };
        let pcm = (0..n)
            .map(|i| (level * (2.0 * PI * f * i as f64 / 48_000.0).sin()) as f32)
            .collect();
        obj.push(pcm);
        let base = 2.0 * PI * f64::from(o * 7 % 32) / 32.0;
        let path: Vec<(usize, ObjectUpdate)> = (0..n)
            .step_by(4_800)
            .map(|i| {
                let drift = if o % 2 == 1 { i as f64 / 96_000.0 } else { 0.0 };
                let phi = base + drift;
                let u = ObjectUpdate {
                    position: [phi.sin() as f32, phi.cos() as f32, (o % 3) as f32 * 0.4],
                    gain: 1.0,
                    size: 0.0,
                    at_sample: 0,
                };
                (i, u)
            })
            .collect();
        updates.push(path);
    }
    Scene {
        bed: Bed::new(&[]),
        objects: 32,
        bed_pcm: vec![],
        obj,
        updates,
    }
}

/// Pushes `scene` in `chunk`-sample chunks, then silence until its last frame is out.
pub(super) fn encode(scene: &Scene, kbps: u32, chunk: usize) -> (Vec<Vec<u8>>, ObjectEncoder) {
    let config = ObjectConfig {
        bed: scene.bed,
        objects: scene.objects,
        bitrate_kbps: kbps,
        blocks: Blocks::Six,
    };
    let mut enc = ObjectEncoder::new(config).unwrap();
    let nb = scene.bed.channels();
    let total = scene.len() + FRAME + LOOKAHEAD;
    let mut frames = Vec::new();
    let mut c = 0;
    while c < total {
        let n = chunk.min(total - c);
        let bed: Vec<f32> = (c..c + n)
            .flat_map(|i| (0..nb).map(move |j| (i, j)))
            .map(|(i, j)| scene.bed_pcm.get(i * nb + j).copied().unwrap_or(0.0))
            .collect();
        enc.push_bed(&bed).unwrap();
        for (o, pcm) in scene.obj.iter().enumerate() {
            let piece: Vec<f32> = (c..c + n)
                .map(|i| pcm.get(i).copied().unwrap_or(0.0))
                .collect();
            let update = scene.updates[o]
                .iter()
                .find(|(at, _)| (c..c + n).contains(at))
                .map(|(at, u)| ObjectUpdate {
                    at_sample: (at - c) as u32,
                    ..*u
                });
            enc.push_object(o as u8, &piece, update).unwrap();
        }
        if let Some(f) = enc.pull() {
            frames.push(f.bytes.to_vec());
        }
        c += n;
    }
    (frames, enc)
}

/// One object's parsed JOC update: band count, steep, offsets, coefficients by data point,
/// channel and band; `None` reuses last frame's.
type Update = Option<(usize, bool, Vec<usize>, Vec<f32>)>;

/// A list of numbers after `key: [` in `s`.
pub(super) fn numbers(s: &str, key: &str) -> Vec<f64> {
    let Some(at) = s.find(&format!("{key}: [")) else {
        return vec![];
    };
    let rest = &s[at + key.len() + 3..];
    let end = rest.find(']').unwrap_or(0);
    rest[..end]
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect()
}

/// The object updates in oxideav's printed `JocMetadata`.
fn updates(dbg: &str) -> Vec<Update> {
    let at = dbg.find("object_updates: [").unwrap() + "object_updates: [".len();
    let mut rest = &dbg[at..];
    let mut out = Vec::new();
    loop {
        rest = rest.trim_start_matches([',', ' ']);
        if let Some(r) = rest.strip_prefix("Reuse") {
            out.push(None);
            rest = r;
        } else if rest.starts_with("Parameters") {
            let end = rest.find('}').unwrap();
            let p = &rest[..end];
            let bands = numbers(
                &format!(
                    "x: [{}]",
                    &p[p.find("num_bands: ").unwrap() + 11..]
                        .split(',')
                        .next()
                        .unwrap()
                ),
                "x",
            )[0];
            let slope = p.contains("slope: true");
            let offsets = numbers(p, "offsets").iter().map(|v| *v as usize).collect();
            let coef = numbers(p, "coefficients")
                .iter()
                .map(|v| *v as f32)
                .collect();
            out.push(Some((bands as usize, slope, offsets, coef)));
            rest = &rest[end + 1..];
        } else {
            break;
        }
    }
    out
}

/// What oxideav reads from a stream: the core per channel (L R C Ls Rs LFE), and per frame
/// the printed OAMD, the sequence count and the object updates.
pub(super) struct Decoded {
    pub(super) core: Vec<Vec<f32>>,
    pub(super) oamd: Vec<String>,
    seq: Vec<u16>,
    updates: Vec<Vec<Update>>,
    pub(super) metadata: Vec<oxideav_ac3::eac3::joc::JocMetadata>,
}

pub(super) fn decode(frames: &[Vec<u8>], elements: usize) -> Decoded {
    let mut state = decoder::Eac3DecoderState::default();
    let mut oamd = OamdDecoder::default();
    let mut d = Decoded {
        core: vec![vec![]; 6],
        oamd: vec![],
        seq: vec![],
        updates: vec![],
        metadata: vec![],
    };
    for (k, f) in frames.iter().enumerate() {
        decoder::decode_eac3_packet(&mut state, f).unwrap();
        assert_eq!(
            state.frames_zero_filled, 0,
            "frame {k}: {:?}",
            state.last_error
        );
        // bitstream order L C R Ls Rs LFE to JOC order
        for (i, s) in state.indep_pcm_f32().iter().enumerate() {
            d.core[[0, 2, 1, 3, 4, 5][i % 6]].push(*s);
        }
        let b = bsi::parse(&f[2..]).unwrap();
        let signal = parse_ec3_extension_type_a(b.addbsi.as_ref().unwrap().payload())
            .unwrap()
            .unwrap();
        let mut br = BitReader::new(&f[2..]);
        br.skip(b.bits_consumed as u32).unwrap();
        let a = audfrm::parse_with(&mut br, &b).unwrap();
        let mut out = vec![0f32; 1536 * 6];
        let mut skips = vec![];
        dsp::decode_indep_audblks(&b, &a, &mut br, &mut Ac3State::new(), &mut out, &mut skips)
            .unwrap();
        let field = skips.iter().find(|s| s.starts_with(&[0x58, 0x38])).unwrap();
        let meta = parse_joc_emdf(field, signal).unwrap_or_else(|e| panic!("frame {k}: {e}"));
        let dbg = format!("{meta:?}");
        assert!(
            dbg.contains(&format!("num_objects: {elements},")),
            "frame {k}"
        );
        let payload: Vec<u8> = numbers(&dbg, "oamd_payload")
            .iter()
            .map(|v| *v as u8)
            .collect();
        let o = oamd
            .decode(&payload)
            .unwrap_or_else(|e| panic!("frame {k}: {e}"));
        d.oamd.push(format!("{o:?}"));
        d.seq.push(
            numbers(
                &format!(
                    "s: [{}]",
                    dbg.split("sequence_count: ")
                        .nth(1)
                        .unwrap()
                        .split(',')
                        .next()
                        .unwrap()
                ),
                "s",
            )[0] as u16,
        );
        d.updates.push(updates(&dbg));
        d.metadata.push(meta);
    }
    d
}

/// The band of `sb` among `bands` parameter bands (Table 54).
fn band(bands: usize, sb: usize) -> usize {
    let idx = tables::NUM_BANDS
        .iter()
        .position(|b| usize::from(*b) == bands)
        .unwrap();
    let atom = tables::ATOM_START
        .iter()
        .rposition(|s| usize::from(*s) <= sb)
        .unwrap();
    usize::from(tables::ATOM_BAND[idx * 23 + atom])
}

/// Clause 7.3 for one object: Pseudocode 13 to 17, Pseudocode 14 in its matrix form
/// `(1/64)·Re Σ Q[k]·exp(iπ(k+½)(j−127.5)/64)`, as two DCT-IVs.
struct Synthesis {
    state: Vec<f32>,
}

impl Synthesis {
    fn process(&mut self, re: &[f32; 64], im: &[f32; 64], out: &mut [f32]) {
        self.state.copy_within(..1280 - 128, 128);
        let mut a = [0f32; 64];
        let mut b = [0f32; 64];
        qmf::dct4(re, &mut a);
        let rev: [f32; 64] = std::array::from_fn(|k| im[63 - k]);
        qmf::dct4(&rev, &mut b);
        for (r, v) in b.iter_mut().enumerate() {
            if r % 2 == 1 {
                *v = -*v;
            }
        }
        for j in 0..128 {
            let v = if j < 64 {
                -a[j] + b[j]
            } else {
                a[127 - j] + b[127 - j]
            };
            self.state[j] = v / 64.0;
        }
        for (ts, o) in out.iter_mut().enumerate().take(64) {
            let mut s = 0f32;
            for j in 0..5 {
                s += self.state[256 * j + ts] * tables::QWIN[128 * j + ts];
                s += self.state[256 * j + 192 + ts] * tables::QWIN[128 * j + 64 + ts];
            }
            *o = s;
        }
    }
}

/// Every element rebuilt from the decoded core and the parsed matrices.
#[allow(clippy::needless_range_loop)] // Pseudocode 6 and 7 index by object, timeslot, band
pub(super) fn rebuild(d: &Decoded, elements: usize) -> Vec<Vec<f32>> {
    let mut analysis: Vec<Analysis> = (0..CH).map(|_| Analysis::new()).collect();
    let mut synth: Vec<Synthesis> = (0..elements)
        .map(|_| Synthesis {
            state: vec![0.0; 1280],
        })
        .collect();
    let mut prev = vec![[[0f32; 64]; CH]; elements];
    let mut out = vec![vec![0f32; d.core[0].len()]; elements];
    for (k, ups) in d.updates.iter().enumerate() {
        if d.seq[k] == 0 {
            prev = vec![[[0f32; 64]; CH]; elements];
        }
        // per object, timeslot, channel, subband
        let mut m = vec![[[[0f32; 64]; CH]; 24]; elements];
        for (o, u) in ups.iter().enumerate() {
            for ch in 0..CH {
                for sb in 0..64 {
                    let p = prev[o][ch][sb];
                    let Some((nb, slope, offsets, coef)) = u else {
                        for t in 0..24 {
                            m[o][t][ch][sb] = p;
                        }
                        continue;
                    };
                    let per = CH * nb;
                    let first = coef[ch * nb + band(*nb, sb)];
                    let second =
                        (coef.len() == 2 * per).then(|| coef[per + ch * nb + band(*nb, sb)]);
                    for t in 0..24 {
                        m[o][t][ch][sb] = if *slope {
                            if t < offsets[0] {
                                p
                            } else {
                                match second {
                                    Some(s) if t >= offsets[1] => s,
                                    _ => first,
                                }
                            }
                        } else {
                            match second {
                                None => p + (first - p) * (t + 1) as f32 / 24.0,
                                Some(_) if t < 12 => p + (first - p) * (t + 1) as f32 / 12.0,
                                Some(s) => first + (s - first) * (t - 11) as f32 / 12.0,
                            }
                        };
                    }
                    prev[o][ch][sb] = second.unwrap_or(first);
                }
            }
        }
        for t in 0..24 {
            let at = 1536 * k + 64 * t;
            let mut x = [[[0f32; 64]; 2]; CH];
            for c in 0..CH {
                let pcm: [f32; 64] = std::array::from_fn(|i| d.core[c][at + i]);
                let [re, im] = &mut x[c];
                analysis[c].process(&pcm, re, im);
            }
            for o in 0..elements {
                let (mut re, mut im) = ([0f32; 64], [0f32; 64]);
                for c in 0..CH {
                    for sb in 0..64 {
                        re[sb] += x[c][0][sb] * m[o][t][c][sb];
                        im[sb] += x[c][1][sb] * m[o][t][c][sb];
                    }
                }
                synth[o].process(&re, &im, &mut out[o][at..at + 64]);
            }
        }
    }
    out
}

/// SNR in dB of `got` against `want` over `range`, `got` late by `DELAY`.
pub(super) fn snr(want: &[f32], got: &[f32], range: std::ops::Range<usize>) -> f64 {
    let (mut s, mut e) = (0f64, 0f64);
    for i in range {
        let (w, g) = (f64::from(want[i]), f64::from(got[i + DELAY]));
        s += w * w;
        e += (w - g) * (w - g);
    }
    10.0 * (s / e.max(1e-30)).log10()
}

#[test]
fn the_rebuild_is_a_delayed_identity_without_a_matrix() {
    // analysis then synthesis, no matrix: the input back 577 samples late
    let mut a = Analysis::new();
    let mut s = Synthesis {
        state: vec![0.0; 1280],
    };
    let mut seed = 7u32;
    let x: Vec<f32> = (0..64 * 60).map(|_| 0.3 * noise(&mut seed)).collect();
    let mut y = vec![0f32; x.len()];
    for t in 0..60 {
        let (mut re, mut im) = ([0f32; 64], [0f32; 64]);
        a.process(&std::array::from_fn(|i| x[64 * t + i]), &mut re, &mut im);
        s.process(&re, &im, &mut y[64 * t..64 * t + 64]);
    }
    assert!(snr(&x, &y, 700..64 * 60 - 700) > 60.0);
}

#[test]
fn the_rebuild_matches_oxideavs_own_render_of_one_object() {
    // one object hard left: oxideav's stereo left is the object it rebuilt, 577 late
    let mut scene = circling(1.0);
    scene.objects = 1;
    scene.obj.truncate(1);
    scene.updates = vec![vec![(
        0,
        ObjectUpdate {
            position: [-1.0, 0.0, 0.0],
            gain: 1.0,
            size: 0.0,
            at_sample: 0,
        },
    )]];
    let (frames, _) = encode(&scene, 768, 240);
    let d = decode(&frames, 1);
    let ours = rebuild(&d, 1);
    let mut r = JocRenderer::default();
    let mut left: Vec<f32> = Vec::new();
    for (k, meta) in d.metadata.iter().enumerate() {
        let pcm: Vec<f32> = (0..1536)
            .flat_map(|i| [0, 2, 1, 3, 4, 5].map(|c| d.core[c][1536 * k + i]))
            .collect();
        let st = r.render_stereo(meta, &pcm, 6, 7, true).unwrap();
        left.extend(st.iter().step_by(2));
    }
    let n = left.len().min(ours[0].len());
    let (mut s, mut e) = (0f64, 0f64);
    for i in 3000..n - 3000 {
        s += f64::from(left[i]).powi(2);
        e += f64::from(left[i] - ours[0][i]).powi(2);
    }
    assert!(
        10.0 * (s / e).log10() > 40.0,
        "{:.1} dB",
        10.0 * (s / e).log10()
    );
}

/// The probe's speed case: the 7.1.4 voices and five tones circling at their own heights.
pub(super) fn mix(seconds: f32) -> Scene {
    let mut s = speech_bed(seconds);
    for v in s.bed_pcm.iter_mut() {
        *v *= 0.6;
    }
    let n = s.len();
    for o in 0..5u32 {
        let f = (24.0 + 6.0 * f64::from(o) + 0.5) * 375.0;
        s.obj.push(
            (0..n)
                .map(|i| (0.1 * (2.0 * PI * f * i as f64 / 48_000.0).sin()) as f32)
                .collect(),
        );
        let speed = 1.0 + f64::from(o) * 0.5;
        s.updates.push(
            (0..n)
                .step_by(480)
                .map(|i| {
                    let phi = 2.0 * PI * speed * i as f64 / (4.0 * 48_000.0) + f64::from(o);
                    let u = ObjectUpdate {
                        position: [phi.sin() as f32, phi.cos() as f32, o as f32 * 0.25],
                        gain: 1.0,
                        size: 0.0,
                        at_sample: 0,
                    };
                    (i, u)
                })
                .collect(),
        );
    }
    s.objects = 5;
    s
}

/// Each element rebuilt against the signal the encoder gave it, frames 2 to the last but one.
fn rebuilt(scene: &Scene) -> (Vec<f64>, Decoded, ObjectEncoder) {
    let (frames, enc) = encode(scene, 768, 240);
    let elements = enc.elem.len();
    let d = decode(&frames, elements);
    let rec = rebuild(&d, elements);
    let n = enc.trace[0].len().min(rec[0].len() - DELAY);
    let snrs = (0..elements)
        .map(|k| snr(&enc.trace[k], &rec[k], 2 * FRAME..n - FRAME))
        .collect();
    (snrs, d, enc)
}

fn report(name: &str, snrs: &[f64]) {
    let s: Vec<String> = snrs.iter().map(|v| format!("{v:.1}")).collect();
    println!("{name}: {} dB", s.join(" "));
    assert!(snrs.iter().all(|v| *v >= 20.0), "{name}: {}", s.join(" "));
}

/// The positions oxideav read in each update block, by frame: start sample, and per object
/// its room position.
fn positions(oamd: &str) -> Vec<(usize, Vec<[f32; 3]>)> {
    let mut out = Vec::new();
    for block in oamd.split("OamdUpdateBlock").skip(1) {
        let start = numbers(
            &format!(
                "s: [{}]",
                block
                    .split("start_sample: ")
                    .nth(1)
                    .unwrap()
                    .split(',')
                    .next()
                    .unwrap()
            ),
            "s",
        )[0] as usize;
        let objects = block
            .split("OamdObjectState")
            .skip(1)
            .map(|o| {
                let p = numbers(o, "position");
                [p[0] as f32, p[1] as f32, p[2] as f32]
            })
            .collect();
        out.push((start, objects));
    }
    out
}

#[test]
fn a_714_bed_rebuilds_every_element() {
    let (snrs, d, _) = rebuilt(&speech_bed(4.0));
    report("7.1.4 bed", &snrs);
    // the 7.1.4 assignment: eleven bed objects and the LFE
    for o in &d.oamd {
        let kinds = o
            .split("kinds: [")
            .nth(1)
            .unwrap()
            .split(']')
            .next()
            .unwrap();
        assert_eq!(
            (kinds.matches("Bed").count(), kinds.matches("Lfe").count()),
            (11, 1)
        );
    }
}

#[test]
fn moving_objects_rebuild_and_read_back_within_a_step() {
    let scene = circling(4.0);
    let (snrs, d, _) = rebuilt(&scene);
    report("circling and jumping", &snrs);
    // every block's position: the scene's at its start or up to 31 samples on (the grid)
    let mut checked = 0;
    for (k, o) in d.oamd.iter().enumerate() {
        for (start, objects) in positions(o) {
            let at = 1536 * k + start;
            for (obj, got) in objects.iter().enumerate() {
                let Some((_, u)) = scene.updates[obj].iter().rev().find(|(s, _)| *s < at + 32)
                else {
                    continue;
                };
                let [x, y, z] = u.position;
                let want = [(x + 1.0) / 2.0, (1.0 - y) / 2.0, z];
                let step = [1.0 / 62.0, 1.0 / 62.0, 1.0 / 15.0];
                for i in 0..3 {
                    assert!(
                        (got[i] - want[i]).abs() <= step[i] + 1e-6,
                        "frame {k} {got:?} {want:?}"
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 2 * 125, "{checked}");
}

#[test]
fn thirty_two_objects_fold_into_sixteen_elements_without_touching_the_core() {
    let scene = crowd(2.0);
    let (snrs, _, enc) = rebuilt(&scene);
    report("32 objects folded", &snrs);
    // the quietest kept instead, a new set every frame: the core does not change by a bit
    let config = enc.config();
    let mut other = ObjectEncoder::new(config).unwrap();
    other.scene.scramble = true;
    let nb = 0;
    let total = scene.len() + FRAME + LOOKAHEAD;
    let mut c = 0;
    while c < total {
        let n = 240.min(total - c);
        other.push_bed(&vec![0.0; n * nb]).unwrap();
        for (o, pcm) in scene.obj.iter().enumerate() {
            let piece: Vec<f32> = (c..c + n)
                .map(|i| pcm.get(i).copied().unwrap_or(0.0))
                .collect();
            let update = scene.updates[o]
                .iter()
                .find(|(at, _)| (c..c + n).contains(at))
                .map(|(at, u)| ObjectUpdate {
                    at_sample: (at - c) as u32,
                    ..*u
                });
            other.push_object(o as u8, &piece, update).unwrap();
        }
        let _ = other.pull();
        c += n;
    }
    assert!(enc.core_trace == other.core_trace);
    assert!(enc.trace != other.trace);
}

#[test]
fn a_714_bed_and_five_objects_rebuild() {
    let (snrs, _, _) = rebuilt(&mix(2.0));
    report("7.1.4 bed and five objects", &snrs);
}
