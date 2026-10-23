//! The AC-3 output, pinned: a hash of every frame a platform-independent signal encodes to. A change to
//! any of them is a change to the bitstream, made on purpose and named in its commit.

mod common;

use common::{encode, pinned_signal};
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
        (Layout::Stereo, 192, 0xb657_9f4a_0ccc_becbu64),
        (Layout::Surround51, 320, 0xa0d7_4b00_893d_b4a7),
        (Layout::Surround51, 448, 0x80f0_ece0_874b_147b),
        (Layout::Surround51, 640, 0xe576_49b9_ad67_41c9),
    ];
    let got: Vec<u64> = cases
        .iter()
        .map(|(layout, kbps, _)| {
            let config = Config::new(Codec::Ac3, *layout, *kbps).unwrap();
            let frames = encode(config, &pinned_signal(*layout, 12 * 1536), 240);
            fnv1a(frames.iter().flat_map(|(_, f)| f.iter().copied()))
        })
        .collect();
    for ((layout, kbps, want), got) in cases.iter().zip(got) {
        assert_eq!(got, *want, "{layout:?} {kbps}: {got:#018x}");
    }
}
