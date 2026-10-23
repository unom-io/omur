//! Nothing allocates after `Encoder::new` or `ObjectEncoder::new` (docs/design.md D3): a
//! counting global allocator around 100 push/pull cycles per stream. A binary of its own
//! without the test harness, so no other thread allocates meanwhile.

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
    #[cfg(feature = "joc")]
    {
        total += objects();
    }
    println!("realtime: push/pull cycles, {total} frames, no allocation");
}

/// A 7.1.4 bed and 32 objects, folding into the 16 elements, every object moving.
#[cfg(feature = "joc")]
fn objects() -> usize {
    use omur::joc::{Bed, BedChannel, ObjectConfig, ObjectEncoder, ObjectUpdate};
    let bed = Bed::new(&BedChannel::ALL[..12]);
    let config = ObjectConfig {
        bed,
        objects: 32,
        bitrate_kbps: 768,
        blocks: Blocks::Six,
    };
    let mut enc = ObjectEncoder::new(config).unwrap();
    let pcm: Vec<f32> = (0..12 * 240)
        .map(|i| ((i * 7919) % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    let mono: Vec<f32> = pcm.iter().step_by(12).copied().collect();
    let region = Region::new(GLOBAL);
    let mut frames = 0;
    for k in 0..100u32 {
        enc.push_bed(&pcm).unwrap();
        for o in 0..32u8 {
            let t = (k + u32::from(o)) as f32 / 50.0 - 1.0;
            let update = ObjectUpdate {
                position: [t, -t, 0.5],
                gain: 1.0,
                size: 0.2,
                at_sample: k % 240,
            };
            enc.push_object(o, &mono, Some(update)).unwrap();
        }
        if k == 60 {
            enc.end_object(3);
        }
        if let Some(f) = enc.pull() {
            frames += 1;
            black_box(f.bytes);
        }
    }
    let stats = region.change();
    assert_eq!(frames, (100 * 240 - 1792) / 1536 + 1);
    assert_eq!(
        (stats.allocations, stats.reallocations),
        (0, 0),
        "ObjectEncoder: {stats:?}"
    );
    frames
}
