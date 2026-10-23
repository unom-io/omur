//! Every E-AC-3 frame through oxideav-ac3's Annex E decoder: the CRC, the header fields
//! the encoder fixes, and the decoded audio against the input; and through ffmpeg, where
//! it is on `PATH`, without a message.

mod common;

use common::{encode, signal};
use omur::{Blocks, Codec, Config, Layout};
use oxideav_ac3::decoder::verify_packet_crc;
use oxideav_ac3::eac3::{audfrm, bsi, decoder};
use oxideav_core::bits::BitReader;

const BLOCKS: [Blocks; 4] = [Blocks::One, Blocks::Two, Blocks::Three, Blocks::Six];

fn eac3(blocks: Blocks) -> Codec {
    Codec::Eac3 { blocks }
}

/// Decodes the frames with oxideav into one f32 stream per channel, bitstream order.
fn decode(frames: &[(u64, Vec<u8>)], nch: usize) -> Vec<Vec<f32>> {
    let mut state = decoder::Eac3DecoderState::default();
    let mut out = vec![Vec::new(); nch];
    for (k, (_, f)) in frames.iter().enumerate() {
        decoder::decode_eac3_packet(&mut state, f).unwrap();
        assert_eq!(
            state.frames_zero_filled, 0,
            "frame {k}: {:?}",
            state.last_error
        );
        assert_eq!(usize::from(state.indep_nchans()), nch);
        for (i, s) in state.indep_pcm_f32().iter().enumerate() {
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

/// ffmpeg, where it is on `PATH`, decodes `stream` without a message.
fn ffmpeg_decodes(stream: &[u8], what: &str) {
    let path = std::env::temp_dir().join(format!("omur-{}-{what}.ec3", std::process::id()));
    std::fs::write(&path, stream).unwrap();
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-f", "null", "-"])
        .output();
    let _ = std::fs::remove_file(&path);
    if let Ok(out) = out {
        assert!(out.status.success(), "{what}");
        assert_eq!(String::from_utf8_lossy(&out.stderr), "", "{what}");
    }
}

#[test]
fn every_length_and_rate_decodes_with_its_header() {
    for layout in [Layout::Stereo, Layout::Surround51] {
        // per channel in bitstream order, the floor at the lowest rate and one block; Rs is
        // the noise channel, whose top a coupled stream loses
        let (acmod, lfe, wav, floor): (u8, bool, &[usize], &[f64]) = match layout {
            Layout::Stereo => (2, false, &[0, 1], &[32.0, 24.0]),
            _ => (
                7,
                true,
                &[0, 2, 1, 4, 5, 3],
                &[27.0, 20.0, 19.0, 18.0, 1.0, 18.0],
            ),
        };
        for blocks in BLOCKS {
            for &kbps in Config::bitrates(eac3(blocks), layout) {
                let what = format!("{layout:?} {blocks:?} {kbps}");
                let config = Config::new(eac3(blocks), layout, kbps).unwrap();
                let n = blocks.count();
                let pcm = signal(layout, 9 * 1536);
                let frames = encode(config, &pcm, 240);
                assert_eq!(frames.len(), (9 * 1536 - 256) / (256 * n), "{what}");
                for (k, (pts, f)) in frames.iter().enumerate() {
                    assert_eq!(*pts, (256 * n * k) as u64, "{what}");
                    assert_eq!(f.len(), config.frame_bytes(), "{what}");
                    let crc = verify_packet_crc(f).unwrap();
                    assert_eq!((crc.crc1_ok, crc.crc2_ok), (None, Some(true)), "{what}");
                    let b = bsi::parse(&f[2..]).unwrap();
                    assert_eq!(
                        (b.bsid, b.substreamid, b.frame_bytes as usize, b.sample_rate),
                        (16, 0, f.len(), 48_000),
                        "{what}"
                    );
                    assert_eq!((b.num_blocks as usize, b.acmod, b.lfeon), (n, acmod, lfe));
                    assert_eq!(
                        (b.dialnorm, b.chanmap, b.addbsi.is_some()),
                        (31, None, false)
                    );
                    // a converter's group of six blocks starts with the frame at pts 0
                    let group = (256 * n * k).is_multiple_of(1536);
                    assert_eq!(b.convsync, (n < 6).then_some(group), "{what} frame {k}");
                    let mut br = BitReader::new(&f[2..]);
                    br.skip(b.bits_consumed as u32).unwrap();
                    let a = audfrm::parse_with(&mut br, &b).unwrap();
                    let flags = (a.expstre, a.ahte, a.snroffststr, a.transproce, a.blkswe);
                    assert_eq!(flags, (true, false, 0, false, true), "{what}");
                    let flags = (a.dithflage, a.bamode, a.frmfgaincode, a.dbaflde);
                    assert_eq!(flags, (true, true, false, false), "{what}");
                    assert_eq!((a.skipflde, a.spxattene), (true, false), "{what}");
                }
                let decoded = decode(&frames, wav.len());
                let ch = layout.channels();
                let end = frames.len() * 256 * n;
                for (c, slot) in wav.iter().enumerate() {
                    let want: Vec<f32> = pcm.iter().skip(*slot).step_by(ch).copied().collect();
                    // the first frame fades in from no history
                    let snr = snr_db(&want[1536..end], &decoded[c][1536..end]);
                    assert!(snr > floor[c], "{what}: channel {c} at {snr:.1} dB");
                }
                let stream: Vec<u8> = frames.iter().flat_map(|(_, f)| f.clone()).collect();
                ffmpeg_decodes(&stream, &what.replace(' ', "-"));
            }
        }
    }
}

#[test]
fn output_does_not_depend_on_the_chunking() {
    for (layout, blocks, kbps) in [
        (Layout::Surround51, Blocks::One, 448),
        (Layout::Surround51, Blocks::Three, 640),
        (Layout::Stereo, Blocks::Two, 192),
        (Layout::Surround51, Blocks::Six, 1024),
    ] {
        let config = Config::new(eac3(blocks), layout, kbps).unwrap();
        let pcm = signal(layout, 4 * 1536);
        let whole = encode(config, &pcm, 240);
        for chunk in [1, 77, 256, 1000, 1536] {
            assert!(
                encode(config, &pcm, chunk) == whole,
                "{layout:?} {blocks:?} chunk {chunk}"
            );
        }
    }
}

/// The bytes `set_extension` hands over come back, in order, from the skip fields of the
/// frame after it, which still decodes; AC-3 and an oversized payload are refused.
#[test]
fn extensions_ride_the_next_frames_skip_fields() {
    use omur::{Encoder, Error};
    use oxideav_ac3::audblk::Ac3State;
    use oxideav_ac3::eac3::dsp;

    let mut ac3 = Encoder::new(Config::new(Codec::Ac3, Layout::Surround51, 448).unwrap()).unwrap();
    assert_eq!(ac3.set_extension(&[1]), Err(Error::Length));
    for (blocks, kbps, sizes) in [
        (Blocks::Six, 640, &[5, 511, 512, 1200, 0][..]),
        (Blocks::One, 448, &[1, 37, 100][..]),
        (Blocks::Three, 1024, &[1, 1022, 1023, 3][..]),
    ] {
        let config = Config::new(eac3(blocks), Layout::Surround51, kbps).unwrap();
        let mut enc = Encoder::new(config).unwrap();
        assert_eq!(enc.set_extension(&[0; 5000]), Err(Error::Length));
        let pcm = signal(Layout::Surround51, 8 * 1536);
        let mut want = vec![vec![]];
        let mut frames = vec![];
        for chunk in pcm.chunks(6 * 240) {
            enc.push(chunk).unwrap();
            if let Some(f) = enc.pull() {
                frames.push(f.bytes.to_vec());
                let k = frames.len();
                let payload: Vec<u8> = (0..sizes[k % sizes.len()])
                    .map(|i| (i * 7 + k) as u8)
                    .collect();
                enc.set_extension(&payload).unwrap();
                want.push(payload);
            }
        }
        let mut state = decoder::Eac3DecoderState::default();
        for (k, f) in frames.iter().enumerate() {
            let what = format!("{blocks:?} {kbps} frame {k}");
            let b = bsi::parse(&f[2..]).unwrap();
            let mut br = BitReader::new(&f[2..]);
            br.skip(b.bits_consumed as u32).unwrap();
            let a = audfrm::parse_with(&mut br, &b).unwrap();
            let mut out = vec![0f32; blocks.count() * 256 * 6];
            let mut skips = vec![];
            dsp::decode_indep_audblks(&b, &a, &mut br, &mut Ac3State::new(), &mut out, &mut skips)
                .unwrap();
            assert_eq!(skips.concat(), want[k], "{what}");
            assert!(skips.iter().all(|s| s.len() <= 511), "{what}");
            decoder::decode_eac3_packet(&mut state, f).unwrap();
            assert_eq!(state.frames_zero_filled, 0, "{what}");
            assert!(verify_packet_crc(f).unwrap().all_ok(), "{what}");
        }
        ffmpeg_decodes(&frames.concat(), &format!("ext-{blocks:?}-{kbps}"));
    }
}
