//! `ObjectEncoder` under any bed, up to 32 objects, any chunking, hostile updates and hostile
//! f32: never a panic, and every frame is one 5.1 E-AC-3 syncframe oxideav-ac3 checks and
//! decodes, whose `addbsi` signals the extension and whose skip field holds an EMDF
//! container with the JOC and OAMD payloads oxideav parses; a dynamic program renders in
//! its JOC renderer.

#![no_main]

use arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use omur::Error;
use omur::joc::{Bed, BedChannel, ObjectConfig, ObjectEncoder, ObjectUpdate};
use oxideav_ac3::audblk::Ac3State;
use oxideav_ac3::decoder::verify_packet_crc;
use oxideav_ac3::eac3::joc::{parse_ec3_extension_type_a, parse_joc_emdf};
use oxideav_ac3::eac3::joc_renderer::JocRenderer;
use oxideav_ac3::eac3::oamd::OamdDecoder;
use oxideav_ac3::eac3::{audfrm, bsi, decoder, dsp};
use oxideav_core::bits::BitReader;

/// Values a game can hand the encoder that a tone never would.
const SPECIAL: [f32; 12] = [
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
    f32::MIN_POSITIVE,
    1e-40,
];

/// Frames one input encodes past the first.
const MAX_FRAMES: usize = 3;
const FRAME: usize = 1536;

fuzz_target!(|data: &[u8]| {
    let _ = run(&mut Unstructured::new(data));
});

/// A value: a special one, a uniform one at one of four scales, or raw bits.
fn value(u: &mut Unstructured) -> Result<f32> {
    Ok(match u.int_in_range(0..=2)? {
        0 => *u.choose(&SPECIAL)?,
        1 => {
            let scale = *u.choose(&[1.0f32, 0.1, 10.0, 1e6])?;
            scale * (u.int_in_range(-1000..=1000)? as f32 / 1000.0)
        }
        _ => f32::from_bits(u.arbitrary()?),
    })
}

/// `n` samples: one value held, a square wave of two, or noise of its own.
fn signal(u: &mut Unstructured, n: usize) -> Result<Vec<f32>> {
    let (a, b) = (value(u)?, value(u)?);
    let mut seed: u32 = u.arbitrary()?;
    Ok(match u.int_in_range(0..=2)? {
        0 => vec![a; n],
        1 => {
            let period = u.int_in_range(1..=300)?;
            (0..n)
                .map(|i| if (i / period) % 2 == 0 { a } else { b })
                .collect()
        }
        _ => (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed as i32 as f32 / i32::MAX as f32) * a.abs().min(2.0)
            })
            .collect(),
    })
}

fn run(u: &mut Unstructured) -> Result<()> {
    let mask: u16 = u.arbitrary()?;
    let channels: Vec<BedChannel> = BedChannel::ALL
        .iter()
        .enumerate()
        .filter(|(i, _)| mask >> i & 1 == 1)
        .map(|(_, c)| *c)
        .collect();
    let bed = Bed::new(&channels);
    let objects = u.int_in_range(0..=32)?;
    let kbps = *u.choose(&[256, 320, 384, 448, 512, 640, 768, 1024])?;
    let config = ObjectConfig {
        bed,
        objects,
        bitrate_kbps: kbps,
        blocks: omur::Blocks::Six,
    };
    let mut enc = match ObjectEncoder::new(config) {
        Ok(enc) => enc,
        Err(Error::Layout) => {
            assert_eq!(bed.elements() + usize::from(objects.min(16)), 0);
            return Ok(());
        }
        Err(e) => panic!("{config:?}: {e}"),
    };
    let elements = bed.elements() + usize::from(objects).min(16 - bed.elements());
    let nb = bed.channels();
    let total = FRAME + 256 + MAX_FRAMES * FRAME;
    let mut frames = Vec::new();
    let mut done = 0;
    while done < total {
        let n = u.int_in_range(1..=FRAME)?.min(total - done);
        let bed_pcm = signal(u, n * nb)?;
        if nb == 0 && objects == 0 {
            break;
        }
        enc.push_bed(if nb == 0 { &[] } else { &bed_pcm }).unwrap();
        for o in 0..objects {
            match u.int_in_range(0..=4)? {
                0 => {}
                1 => enc.end_object(o),
                k => {
                    let pcm = signal(u, n)?;
                    let update = (k == 4)
                        .then(|| -> Result<ObjectUpdate> {
                            Ok(ObjectUpdate {
                                position: [value(u)?, value(u)?, value(u)?],
                                gain: value(u)?,
                                size: value(u)?,
                                at_sample: u.arbitrary()?,
                            })
                        })
                        .transpose()?;
                    enc.push_object(o, &pcm, update).unwrap();
                }
            }
        }
        if nb == 0 && objects > 0 {
            // a bed without channels: some object must set the chunk's length
            enc.push_object(0, &signal(u, n)?, None).unwrap();
        }
        assert_eq!(
            enc.push_object(objects, &[0.0], None),
            Err(Error::Object { index: objects })
        );
        if let Some(f) = enc.pull() {
            assert_eq!(f.syncframes, [0..4 * kbps as usize]);
            frames.push(f.bytes.to_vec());
        }
        done += n;
    }
    check(&frames, elements, bed);
    Ok(())
}

fn check(frames: &[Vec<u8>], elements: usize, bed: Bed) {
    let named = bed.channels()
        - usize::from(bed.contains(BedChannel::Lfe))
        - usize::from(bed.contains(BedChannel::BackCenter));
    let mut state = decoder::Eac3DecoderState::default();
    let mut oamd = OamdDecoder::default();
    let mut renderer = JocRenderer::default();
    for (k, f) in frames.iter().enumerate() {
        assert!(verify_packet_crc(f).unwrap().all_ok(), "frame {k}");
        let b = bsi::parse(&f[2..]).unwrap();
        assert_eq!((b.bsid, b.acmod, b.lfeon), (16, 7, true));
        let addbsi = b.addbsi.as_ref().unwrap().payload();
        assert_eq!(addbsi, [1, elements as u8]);
        let signal = parse_ec3_extension_type_a(addbsi).unwrap().unwrap();
        let mut br = BitReader::new(&f[2..]);
        br.skip(b.bits_consumed as u32).unwrap();
        let a = audfrm::parse_with(&mut br, &b).unwrap();
        let (mut out, mut skips) = (vec![0f32; FRAME * 6], vec![]);
        dsp::decode_indep_audblks(&b, &a, &mut br, &mut Ac3State::new(), &mut out, &mut skips)
            .unwrap();
        let field = skips
            .iter()
            .find(|s| s.starts_with(&[0x58, 0x38]))
            .expect("an EMDF container");
        let meta = parse_joc_emdf(field, signal).unwrap_or_else(|e| panic!("frame {k}: {e}"));
        let dbg = format!("{meta:?}");
        assert!(
            dbg.contains(&format!("num_objects: {elements},")),
            "frame {k}"
        );
        let at = dbg.find("oamd_payload: [").unwrap() + 15;
        let payload: Vec<u8> = dbg[at..at + dbg[at..].find(']').unwrap()]
            .split(", ")
            .map(|v| v.parse().unwrap())
            .collect();
        oamd.decode(&payload)
            .unwrap_or_else(|e| panic!("frame {k}: {e}"));
        decoder::decode_eac3_packet(&mut state, f).unwrap();
        assert_eq!(state.frames_zero_filled, 0, "frame {k}");
        let pcm = state.indep_pcm_f32();
        assert!(
            pcm.iter().all(|v| v.is_finite() && v.abs() <= 64.0),
            "frame {k}"
        );
        if named == 0 {
            let stereo = renderer
                .render_stereo(&meta, pcm, 6, 7, true)
                .unwrap_or_else(|e| panic!("frame {k}: {e}"));
            assert!(stereo.iter().all(|v| v.is_finite()), "frame {k}");
        }
    }
}
