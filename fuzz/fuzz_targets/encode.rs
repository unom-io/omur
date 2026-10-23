//! Any config, any push schedule, hostile f32: every frame is whole syncframes that
//! oxideav-ac3 checks, parses and decodes to finite audio near full scale.

#![no_main]

mod common;

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use omur::{Codec, Config, Layout};
use oxideav_ac3::eac3::bsi::StreamType;
use oxideav_ac3::{audblk, bsi, decoder, eac3, syncinfo};

fuzz_target!(|data: &[u8]| {
    let _ = run(&mut Unstructured::new(data));
});

/// The input is clamped to ±1.0 at push. Band-limiting alone lifts a full-scale square wave
/// to 1.18 (Gibbs); a one-block frame at 128 kbit/s quantises it to 1.6; coupling rebuilds a
/// channel's top band from the sum of all of them, which takes uncorrelated full-scale
/// squares past 2 on AC-3 and E-AC-3 alike. The bound is twice that.
fn assert_near_full_scale(pcm: &[f32], k: usize) {
    if let Some(s) = pcm.iter().find(|s| !s.is_finite() || s.abs() > 4.0) {
        panic!("frame {k} decodes to {s}");
    }
}

fn run(u: &mut Unstructured) -> arbitrary::Result<()> {
    let Some(config) = common::config(u)? else {
        return Ok(());
    };
    let schedule = common::schedule(u)?;
    let pcm = common::signal(u, config)?;
    let frames = common::encode(config, &pcm, &schedule);
    match config.codec() {
        Codec::Ac3 => ac3(config, &frames),
        Codec::Eac3 { .. } => eac3(config, &frames),
    }
    Ok(())
}

fn ac3(config: Config, frames: &[(u64, Vec<u8>)]) {
    let (acmod, lfeon) = match config.layout() {
        Layout::Stereo => (2, false),
        _ => (7, true),
    };
    let mut state = audblk::Ac3State::new();
    let mut out = vec![0f32; 1536 * config.layout().channels()];
    for (k, (_, f)) in frames.iter().enumerate() {
        assert_eq!(f.len(), config.frame_bytes(), "length of frame {k}");
        let crc = decoder::verify_packet_crc(f).expect("crc header");
        assert_eq!(
            (crc.crc1_ok, crc.crc2_ok),
            (Some(true), Some(true)),
            "crcs of frame {k}"
        );
        let si = syncinfo::parse(f).expect("syncinfo");
        let b = bsi::parse(&f[5..]).expect("bsi");
        let got = (b.bsid, b.acmod, b.lfeon, b.dialnorm);
        assert_eq!(got, (8, acmod, lfeon, 31), "bsi of frame {k}");
        audblk::decode_frame(&mut state, &si, &b, f, &mut out).expect("decode");
        assert_near_full_scale(&out, k);
    }
}

fn eac3(config: Config, frames: &[(u64, Vec<u8>)]) {
    let layout = config.layout();
    // (strmtyp, acmod, lfeon, chanmap) of each syncframe of an access unit
    let mut want = vec![(StreamType::Independent, 7, true, None)];
    match layout {
        Layout::Stereo => want[0] = (StreamType::Independent, 2, false, None),
        Layout::Surround51 => {}
        Layout::Surround71 => want.push((StreamType::Dependent, 6, false, Some(0x1a00))),
    }
    let mut state = eac3::decoder::Eac3DecoderState::default();
    for (k, (_, f)) in frames.iter().enumerate() {
        assert_eq!(f.len(), config.frame_bytes(), "length of frame {k}");
        let mut at = 0;
        for (strmtyp, acmod, lfeon, chanmap) in &want {
            let b = eac3::bsi::parse(&f[at + 2..]).expect("bsi");
            let got = (b.strmtyp, b.bsid, b.acmod, b.lfeon, b.chanmap, b.dialnorm);
            assert_eq!(
                got,
                (*strmtyp, 16, *acmod, *lfeon, *chanmap, 31),
                "bsi of frame {k} at {at}"
            );
            let end = at + b.frame_bytes as usize;
            let crc = decoder::verify_packet_crc(&f[at..end]).expect("crc header");
            assert_eq!((crc.crc1_ok, crc.crc2_ok), (None, Some(true)), "crc of frame {k}");
            at = end;
        }
        assert_eq!(at, f.len(), "syncframes of frame {k}");
        eac3::decoder::decode_eac3_packet(&mut state, f).expect("decode");
        assert_eq!(state.frames_zero_filled, 0, "frame {k}: {:?}", state.last_error);
        assert_eq!(usize::from(state.indep_nchans()), layout.channels());
        assert_near_full_scale(state.indep_pcm_f32(), k);
    }
}
