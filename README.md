# omur

An AC-3 and E-AC-3 encoder (ATSC A/52:2018) in safe Rust, built for a real-time audio
thread: interleaved f32 at 48 kHz in, whole syncframes out, nothing allocated after
construction. Object audio (ETSI TS 103 420) — a height bed plus dynamic objects — sits
behind the `joc` feature, off by default.

*ómur* is Icelandic for an echo, a faint or distant sound.

**Status: design.** Nothing encodes yet. The design — scope, modules, the bit-allocation
plan, the test gates, the phases — is [`docs/design.md`](docs/design.md). The public API is
in place and the size and delay queries are live; `push`/`pull` land in P1.

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
