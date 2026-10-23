//! Nothing allocates after `Encoder::new` (docs/design.md D3): a counting global allocator
//! around 100 push/pull cycles per stream. A binary of its own without the test harness, so
//! no other thread allocates meanwhile.

use std::alloc::System;
use std::hint::black_box;

use omur::{Blocks, Codec, Config, Encoder, Layout};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn main() {
    let one = Codec::Eac3 {
        blocks: Blocks::One,
    };
    let six = Codec::Eac3 {
        blocks: Blocks::Six,
    };
    let cases = [
        (Codec::Ac3, Layout::Stereo, 192),
        (Codec::Ac3, Layout::Surround51, 448),
        (one, Layout::Surround51, 448),
        (six, Layout::Surround71, 1024),
    ];
    let mut total = 0;
    for (codec, layout, kbps) in cases {
        let ch = layout.channels();
        let pcm: Vec<f32> = (0..100 * 240 * ch)
            .map(|i| ((i * 7919) % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let config = Config::new(codec, layout, kbps).unwrap();
        let mut enc = Encoder::new(config).unwrap();
        let region = Region::new(GLOBAL);
        let mut frames = 0;
        for chunk in pcm.chunks(240 * ch) {
            enc.push(chunk).unwrap();
            if let Some(f) = enc.pull() {
                frames += 1;
                black_box(f.bytes);
            }
            // the extension hook copies into the encoder's own buffer
            let _ = enc.set_extension(&[0x58, 0x38, 1, 2]);
        }
        let stats = region.change();
        assert_eq!(
            frames,
            1 + (100 * 240 - config.delay_samples()) / config.frame_samples()
        );
        assert_eq!(
            (stats.allocations, stats.reallocations),
            (0, 0),
            "{codec:?} {layout:?}: {stats:?}"
        );
        total += frames;
    }
    println!("realtime: 400 push/pull cycles, {total} frames, no allocation");
}
