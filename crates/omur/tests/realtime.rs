//! Nothing allocates after `Encoder::new` (docs/design.md D3): a counting global allocator
//! around 100 push/pull cycles. A binary of its own without the test harness, so no other
//! thread allocates meanwhile.

use std::alloc::System;
use std::hint::black_box;

use omur::{Codec, Config, Encoder, Layout};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn main() {
    for (layout, kbps) in [(Layout::Stereo, 192), (Layout::Surround51, 448)] {
        let ch = layout.channels();
        let pcm: Vec<f32> = (0..100 * 240 * ch)
            .map(|i| ((i * 7919) % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let mut enc = Encoder::new(Config::new(Codec::Ac3, layout, kbps).unwrap()).unwrap();
        let region = Region::new(GLOBAL);
        let mut frames = 0;
        for chunk in pcm.chunks(240 * ch) {
            enc.push(chunk).unwrap();
            if let Some(f) = enc.pull() {
                frames += 1;
                black_box(f.bytes);
            }
        }
        let stats = region.change();
        assert_eq!(frames, 15);
        assert_eq!(
            (stats.allocations, stats.reallocations),
            (0, 0),
            "{layout:?}: {stats:?}"
        );
    }
    println!("realtime: 200 push/pull cycles, 30 frames, no allocation");
}
