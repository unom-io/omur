# omur

An AC-3 and E-AC-3 encoder (ATSC A/52:2018) in safe Rust, built for a real-time audio
thread: interleaved f32 at 48 kHz in, whole syncframes out, nothing allocated after
construction. Object audio (ETSI TS 103 420) — a height bed plus dynamic objects — sits
behind the `joc` feature, off by default.

*ómur* is Icelandic for an echo, a faint or distant sound.

**Status: P2.** AC-3 2.0 and 5.1, and E-AC-3 2.0, 5.1 and 7.1 (an independent 5.1 with
the surrounds folded plus a dependent substream) at every frame length, encode with
coupling and block switching; object audio does not yet. The design — scope, modules, the
bit-allocation plan, the test gates, the phases — is [`docs/design.md`](docs/design.md).

```rust
use omur::{Codec, Config, Encoder, Layout};

let config = Config::new(Codec::Ac3, Layout::Surround51, 448)?;
println!("{} bytes a frame, {} samples behind", config.frame_bytes(), config.delay_samples());
let mut enc = Encoder::new(config)?;
enc.push(&chunk)?;                 // any chunk, WAV order: FL FR FC LFE BL BR (SL SR)
while let Some(frame) = enc.pull() {
    send(frame.bytes);             // one syncframe per datagram
}
```

## What it will do

| | AC-3 | E-AC-3 |
|---|---|---|
| Layouts | 2.0, 5.1 | 2.0, 5.1, 7.1 (two substreams) |
| Frame | 1536 samples, 32 ms | 256 × 1, 2, 3 or 6 samples; short frames for latency and small datagrams |
| Carries over | optical S/PDIF, HDMI | HDMI ARC / eARC |
| Quality bar | within 1 dB of ffmpeg's encoder per channel, plus a listening pass | the same core |

Plays on any receiver that decodes AC-3 or E-AC-3.

## E-AC-3 frame length

Shorter frames cut the delay and the datagram; they cost header and fresh exponents. SNR in
dB per channel of the probe's 5.1 signal after an ffmpeg decode, bytes per syncframe
(2026-10-23, ffmpeg 9.0.1):

| kbps | frame | bytes | FL | FR | FC | LFE | SL | SR |
|---|---|---|---|---|---|---|---|---|
| 448 | AC-3 | 1792 | 41.1 | 31.1 | 6.3 | 36.4 | 23.0 | 6.9 |
| 448 | 6 blocks | 1792 | 40.2 | 30.4 | 6.3 | 35.6 | 23.0 | 6.9 |
| 448 | 3 blocks | 896 | 43.2 | 28.7 | 6.2 | 35.1 | 22.9 | 6.8 |
| 448 | 2 blocks | 598 | 42.1 | 27.3 | 6.0 | 34.4 | 22.8 | 6.6 |
| 448 | 1 block | 298 | 41.3 | 24.5 | 5.6 | 32.8 | 22.2 | 6.0 |
| 448 | ffmpeg `eac3` | 1792 | 41.1 | 27.7 | 6.2 | 33.5 | 22.9 | 6.7 |
| 640 | AC-3 | 2560 | 49.4 | 38.0 | 8.0 | 45.6 | 25.8 | 8.1 |
| 640 | 6 blocks | 2560 | 48.6 | 37.1 | 8.0 | 44.8 | 25.8 | 8.1 |
| 640 | 3 blocks | 1280 | 51.5 | 35.9 | 8.0 | 43.7 | 25.7 | 8.0 |
| 640 | 2 blocks | 854 | 50.6 | 35.1 | 8.0 | 42.6 | 25.6 | 8.0 |
| 640 | 1 block | 426 | 50.1 | 33.4 | 7.9 | 42.7 | 25.4 | 7.9 |
| 640 | ffmpeg `eac3` | 2560 | 49.6 | 35.1 | 8.0 | 40.9 | 25.6 | 8.0 |

Recommended: three blocks. At 640 kbit/s a syncframe is 1,280 bytes, one datagram on a
1,500-byte path, 21.3 ms of delay, and costs FR and the LFE about 1 dB against six blocks
while FL gains. E-AC-3 sends one fine SNR offset for every channel: ffmpeg 9.0.1 decodes
the per-channel strategies wrongly, so the per-channel step AC-3 takes (FL, FR and the LFE,
0.4–0.9 dB) is not available.

## Why

[punktfunk](https://github.com/unom-io/punktfunk) streams a game host's audio as Opus. A
TV cannot put Opus or multichannel PCM out over optical, so a soundbar behind it gets stereo.
The host encodes AC-3 or E-AC-3 itself; clients pass the bitstream through unchanged.

## Verifying

`tools/probe` is the quality gate: it generates a 10-second 5.1 test signal, encodes it,
and scores an ffmpeg decode per channel against the input. The recipe is in
[`docs/design.md` §9](docs/design.md).

## Credits

Written from the specifications. Where a table or a packing routine is adapted from
[oxideav-ac3](https://github.com/OxideAV/oxideav-ac3) (Karpelès Lab Inc., MIT) or its fork
[vuio-codec-ac3](https://github.com/vuiodev/vuio) (vyrti, MIT OR Apache-2.0), the file says
so and `NOTICE` carries their notice. ffmpeg is used as a black-box oracle in tests and
nothing else.

## Licence

MIT or Apache-2.0, at your option. AC-3's patents have expired; E-AC-3's last US patent
reportedly expired in January 2026, other countries vary; object coding is behind a release
gate until its patent question is answered. None of that is legal advice.
