//! Any AC-3 config, any push schedule, hostile f32: every frame is a whole syncframe that
//! oxideav-ac3 checks, parses and decodes to finite audio no louder than dither allows.

#![no_main]

mod common;

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use omur::Layout;
use oxideav_ac3::{audblk, bsi, decoder, syncinfo};

fuzz_target!(|data: &[u8]| {
    let _ = run(&mut Unstructured::new(data));
});

fn run(u: &mut Unstructured) -> arbitrary::Result<()> {
    let Some(config) = common::config(u)? else {
        return Ok(());
    };
    let schedule = common::schedule(u)?;
    let pcm = common::signal(u, config)?;
    let (acmod, lfeon) = match config.layout() {
        Layout::Stereo => (2, false),
        _ => (7, true),
    };
    let mut state = audblk::Ac3State::new();
    let mut out = vec![0f32; 1536 * config.layout().channels()];
    for (k, (_, f)) in common::encode(config, &pcm, &schedule).iter().enumerate() {
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
        // Dither fills every masked bin at its exponent's level, and a block's noise spreads
        // over its whole 512-sample window, so a hostile input can decode louder than itself;
        // garbage from a misread frame is louder by orders of magnitude. Frame k holds input
        // samples 1536k on (the stream adds no delay); its windows reach 256 either side.
        let ch = config.layout().channels();
        let lo = (k * 1536).saturating_sub(256) * ch;
        let hi = (((k + 1) * 1536 + 256) * ch).min(pcm.len());
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
    Ok(())
}
