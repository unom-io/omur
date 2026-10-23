# omur

An AC-3 and E-AC-3 encoder (ATSC A/52:2018) in safe Rust, built for a real-time audio
thread: interleaved f32 at 48 kHz in, whole syncframes out, nothing allocated after
construction. Object audio (ETSI TS 103 420) — a height bed plus dynamic objects — sits
behind the `joc` feature, off by default.

*ómur* is Icelandic for an echo, a faint or distant sound.

**Status.** AC-3 2.0 and 5.1; E-AC-3 2.0, 5.1 and 7.1 (an independent 5.1 with the surrounds
folded, plus a dependent substream) at every frame length; object audio over a 5.1 E-AC-3
core behind `joc`. Coupling, block switching, per-channel rate control, phase flags. No
`unsafe`, no dependencies, byte-identical output on every platform. Not yet on crates.io;
the listening pass and the receiver checks of [`docs/design.md`](docs/design.md) §14 are
owed. The design — scope, modules, the bit-allocation plan, the test gates, the phases — is
[`docs/design.md`](docs/design.md).

```rust
use omur::{Codec, Config, Encoder, Layout};

let config = Config::new(Codec::Ac3, Layout::Surround51, 448)?;
println!("{} bytes a frame, {} samples behind", config.frame_bytes(), config.delay_samples());
let mut enc = Encoder::new(config)?;
enc.push(&chunk)?;                 // up to a frame per push, WAV order: FL FR FC LFE BL BR (SL SR)
while let Some(frame) = enc.pull() {
    send(frame.bytes);             // one syncframe per datagram; 7.1 is two, `frame.syncframes`
}
```

| | AC-3 | E-AC-3 |
|---|---|---|
| Layouts | 2.0, 5.1 | 2.0, 5.1, 7.1 (two substreams) |
| Frame | 1536 samples, 32 ms | 256 × 1, 2, 3 or 6 samples; short frames for latency and small datagrams |
| Carries over | optical S/PDIF, HDMI | HDMI ARC / eARC |
| Delay | 1,792 samples (37.3 ms) | 256 × blocks + 256 |

Plays on any receiver that decodes AC-3 or E-AC-3.

## Measured

All numbers 2026-10-23 on an Apple M3 Ultra, one thread, release build, ffmpeg 9.0.1 as the
decoder and as the reference encoder. The signal is `probe gen`'s 10-second 5.1: FL a
harmonic tone with tremolo, FR a log sweep, FC speech-like noise bursts, LFE 40 + 80 Hz, SL
pink noise, SR gunshot-like bursts (`cmp` scores a waveform SNR, which cannot judge bursts).

### AC-3, SNR in dB per channel

| kbps | encoder | FL | FR | FC | LFE | SL | SR |
|---|---|---|---|---|---|---|---|
| 320 | omur | 37.0 | 27.4 | 3.6 | 33.2 | 20.3 | 4.0 |
| 320 | ffmpeg `ac3` | 36.4 | 26.6 | 3.7 | 32.6 | 20.5 | 4.0 |
| 448 | omur | 41.1 | 31.1 | 6.3 | 36.4 | 23.0 | 6.9 |
| 448 | ffmpeg `ac3` | 40.5 | 30.5 | 6.3 | 35.4 | 23.2 | 6.9 |
| 640 | omur | 49.4 | 38.0 | 8.0 | 45.6 | 25.8 | 8.1 |
| 640 | ffmpeg `ac3` | 48.8 | 37.4 | 8.0 | 44.5 | 25.8 | 8.1 |

2.0 at 192 kbit/s (FL, FR of the same signal): omur 56.7 / 41.4, ffmpeg 56.7 / 41.1. 7.1
E-AC-3 at 1,024 kbit/s, decoded by ffmpeg to eight channels: FL 43.7, FR 33.0, FC 7.8,
LFE 38.3, BL 24.0, BR 7.4, SL 28.3, SR 24.0.

### Speed, one core, a 300-second input

| | omur | ffmpeg (`-threads 1`) |
|---|---|---|
| AC-3 5.1 at 448 | 526× real time, slowest frame 100 µs | 220× |
| E-AC-3 5.1 at 448, six blocks | 575×, 87 µs | 219× |
| E-AC-3 7.1 at 1,024 | 319×, 152 µs | — (ffmpeg's encoder stops at 5.1) |

### E-AC-3 frame length

Shorter frames cut the delay and the datagram; they cost header and fresh exponents. Bytes
per syncframe and SNR per channel:

| kbps | frame | bytes | FL | FR | FC | LFE | SL | SR |
|---|---|---|---|---|---|---|---|---|
| 448 | AC-3 | 1792 | 41.1 | 31.1 | 6.3 | 36.4 | 23.0 | 6.9 |
| 448 | 6 blocks | 1792 | 40.6 | 30.6 | 6.3 | 36.3 | 23.0 | 6.9 |
| 448 | 3 blocks | 896 | 43.4 | 28.8 | 6.2 | 35.6 | 22.9 | 6.8 |
| 448 | 2 blocks | 598 | 42.2 | 27.3 | 6.0 | 34.6 | 22.8 | 6.6 |
| 448 | 1 block | 298 | 41.3 | 24.5 | 5.6 | 32.8 | 22.2 | 6.0 |
| 448 | ffmpeg `eac3` | 1792 | 41.1 | 27.7 | 6.2 | 33.5 | 22.9 | 6.7 |
| 640 | AC-3 | 2560 | 49.4 | 38.0 | 8.0 | 45.6 | 25.8 | 8.1 |
| 640 | 6 blocks | 2560 | 49.3 | 37.3 | 8.0 | 45.9 | 25.8 | 8.1 |
| 640 | 3 blocks | 1280 | 51.9 | 36.0 | 8.0 | 44.6 | 25.7 | 8.0 |
| 640 | 2 blocks | 854 | 50.8 | 35.3 | 8.0 | 42.9 | 25.6 | 8.0 |
| 640 | 1 block | 426 | 50.1 | 33.4 | 7.9 | 42.7 | 25.4 | 7.9 |
| 640 | ffmpeg `eac3` | 2560 | 49.6 | 35.1 | 8.0 | 40.9 | 25.6 | 8.0 |

Recommended: three blocks. At 640 kbit/s a syncframe is 1,280 bytes, one datagram on a
1,500-byte path, 21.3 ms of delay, and costs FR and the LFE about 1 dB against six blocks
while FL gains. E-AC-3 sends one fine SNR offset for every channel (ffmpeg 9.0.1 misreads
the per-block strategies) and spends what that leaves on per-channel fast gain, which puts it
within 0.8 dB of AC-3 on every channel.

### Object audio

A 7.1.4 bed and the synthetic scenes of `probe scene`, encoded at 768 kbit/s on a 5.1 core
and rebuilt from the core by the reference decoder's parsers: bed channels at 61 dB SNR,
height channels at 41 dB, two moving pink-noise objects at 25 dB (the core's own ceiling on
pink noise), 32 objects folded into 16 elements at 24–36 dB; every position reads back
within one quantisation step. The core costs 0.1 dB (bed) to 0.6 dB (five moving objects)
against plain 5.1 at the same rate. Speed: 311× for two objects, 175× for a 7.1.4 bed,
122× for the bed with five objects, 96× for 32 objects; slowest chunk 189 µs. A receiver
has not played it yet.

## Why

[punktfunk](https://github.com/unom-io/punktfunk) streams a game host's audio as Opus. A
TV cannot put Opus or multichannel PCM out over optical, so a soundbar behind it gets stereo.
The host encodes AC-3 or E-AC-3 itself; clients pass the bitstream through unchanged.

## Verifying

`tools/probe` is the quality gate: `gen` writes the test signal, `enc` encodes it with omur
(`--features omur`; `BLOCKS=1|2|3|6` selects E-AC-3, `CH=2|8` the layout), `cmp` scores an
ffmpeg decode per channel, `dump` prints what any AC-3 or E-AC-3 stream's encoder decided,
and `scene` (`--features omur,joc`) writes the object-audio scenes for a receiver. The recipe
is in [`docs/design.md` §9](docs/design.md). `cargo test --workspace` round-trips every
rate through oxideav-ac3 and, where ffmpeg is on `PATH`, through ffmpeg; `fuzz/` holds four
`cargo fuzz` targets.

## Credits

Written from the specifications. Where a table or a packing routine is adapted from
[oxideav-ac3](https://github.com/OxideAV/oxideav-ac3) (Karpelès Lab Inc., MIT) or its fork
[vuio-codec-ac3](https://github.com/vuiodev/vuio) (vyrti, MIT OR Apache-2.0), the file says
so and `NOTICE` carries their notice; the object-audio QMF window and Huffman trees are
such tables. ffmpeg is used as a black-box oracle in tests and nothing else.

## Licence

MIT or Apache-2.0, at your option. AC-3's patents have expired; E-AC-3's last US patent
reportedly expired in January 2026, other countries vary; object coding is behind a release
gate until its patent question is answered. None of that is legal advice.
