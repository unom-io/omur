//! The AC-3 output, pinned: a hash of every frame the test signal encodes to. A change to
//! any of them is a change to the bitstream, made on purpose and named in its commit.

mod common;

use common::{encode, signal};
use omur::{Codec, Config, Layout};

/// FNV-1a, 64 bits.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[test]
fn ac3_output_is_unchanged() {
    let cases = [
        (Layout::Stereo, 192, 0x9827_06f6_cea6_ba49u64),
        (Layout::Surround51, 320, 0x2298_86da_369e_8921),
        (Layout::Surround51, 448, 0x7052_5db9_1ab1_10bf),
        (Layout::Surround51, 640, 0x15da_fe62_491f_c55a),
    ];
    for (layout, kbps, want) in cases {
        let config = Config::new(Codec::Ac3, layout, kbps).unwrap();
        let frames = encode(config, &signal(layout, 12 * 1536), 240);
        let got = fnv1a(frames.iter().flat_map(|(_, f)| f.iter().copied()));
        assert_eq!(got, want, "{layout:?} {kbps}: {got:#018x}");
    }
}
