//! Any config, any push schedule, hostile f32: every frame is whole syncframes that
//! oxideav-ac3 checks, parses and decodes to finite audio no louder than dither allows.

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

/// Dither fills every masked bin at its exponent's level, and a block's noise spreads over
/// its whole 512-sample window, so a hostile input can decode louder than itself; garbage
/// from a misread frame is louder by orders of magnitude. Frame `k` holds input samples
/// `k · frame_samples` on (the stream adds no delay); its windows reach 256 either side.
fn assert_sane(config: Config, pcm: &[f32], out: &[f32], k: usize) {
    let ch = config.layout().channels();
    let n = config.frame_samples();
    let lo = (k * n).saturating_sub(256) * ch;
    let hi = (((k + 1) * n + 256) * ch).min(pcm.len());
    let frame_in = &pcm[lo..hi];
    let rms_in = (frame_in
        .iter()
        .map(|s| {
            let c = if s.is_finite() {
                s.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            f64::from(c) * f64::from(c)
        })
        .sum::<f64>()
        / frame_in.len() as f64)
        .sqrt();
    let rms_out =
        (out.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / out.len() as f64).sqrt();
    let peak = out.iter().fold(0f32, |m, s| m.max(s.abs()));
    assert!(
        out.iter().all(|s| s.is_finite()) && peak <= 64.0 && rms_out <= 4.0 * rms_in + 0.05,
        "frame {k} decodes to rms {rms_out:.3} (input {rms_in:.3}), peak {peak:.3}"
    );
}

fn run(u: &mut Unstructured) -> arbitrary::Result<()> {
    let Some(config) = common::config(u)? else {
        return Ok(());
    };
    let schedule = common::schedule(u)?;
    let pcm = common::signal(u, config)?;
    let frames = common::encode(config, &pcm, &schedule);
    match config.codec() {
        Codec::Ac3 => ac3(config, &pcm, &frames),
        Codec::Eac3 { .. } => eac3(config, &pcm, &frames),
    }
    Ok(())
}

fn ac3(config: Config, pcm: &[f32], frames: &[(u64, Vec<u8>)]) {
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
        assert_sane(config, pcm, &out, k);
    }
}

fn eac3(config: Config, pcm: &[f32], frames: &[(u64, Vec<u8>)]) {
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
            assert_eq!(
                (crc.crc1_ok, crc.crc2_ok),
                (None, Some(true)),
                "crc of frame {k}"
            );
            at = end;
        }
        assert_eq!(at, f.len(), "syncframes of frame {k}");
        eac3::decoder::decode_eac3_packet(&mut state, f).expect("decode");
        assert_eq!(
            state.frames_zero_filled, 0,
            "frame {k}: {:?}",
            state.last_error
        );
        assert_eq!(usize::from(state.indep_nchans()), layout.channels());
        assert_sane(config, pcm, state.indep_pcm_f32(), k);
    }
}
