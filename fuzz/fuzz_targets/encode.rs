//! Any AC-3 config, any push schedule, hostile f32: every frame is a whole syncframe that
//! oxideav-ac3 checks, parses and decodes to finite audio within full scale.

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
        if let Some(s) = out.iter().find(|s| !s.is_finite() || s.abs() > 1.0 + 1e-3) {
            panic!("frame {k} decodes to {s}");
        }
    }
    Ok(())
}
