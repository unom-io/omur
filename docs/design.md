# omur — design

| | |
|---|---|
| **Status** | Built 2026-10-23: P1–P3 on `main`, measured in the README; the listening pass, the receiver checks and P4 are owed (§14) |
| **Date** | 2026-10-23 |
| **Name** | Icelandic *ómur*: an echo, a faint or distant sound; from *óma*, "to resound". Alternatives: *hljómur* (sound, tone, chord) and *dynur* (din, rumble). None is a trademark; `omur`, `hljomur` and `dynur` are free on crates.io. |
| **One line** | An AC-3 and E-AC-3 encoder (ATSC A/52:2018) in safe Rust for a real-time audio thread: interleaved f32 in, whole syncframes out, nothing allocated after construction. Object audio (ETSI TS 103 420) inside the crate, behind an off-by-default feature and a release gate. |

[punktfunk](https://github.com/unom-io/punktfunk) streams a game host's audio as Opus. A
Samsung TV cannot put Opus or multichannel PCM out over optical, so a soundbar behind it gets
stereo. The host will encode AC-3 (optical S/PDIF, 5.1 at most) or E-AC-3 (HDMI ARC/eARC,
7.1, later object audio) itself, and every client passes the bitstream through to the soundbar
or AVR unchanged. omur is that encoder. It never touches punktfunk; integration — a new
`Welcome` audio codec id, client passthrough sinks — is a later monorepo task, and §1 is what
makes it easy.

## 0. Decisions

| # | Decision | Recommendation |
|---|---|---|
| D1 | Scope | **Encoder only, 48 kHz only, three layouts: 2.0, 5.1, 7.1.** No decoder (tests borrow one), no file I/O, no metadata surface: `dialnorm` is fixed at −31 dB (unity on every decoder), no `compr`/`dynrng`, no downmix hints. A game host has no use for any of it. |
| D2 | Where the code comes from | **The specification.** A/52:2018 and TS 103 420 are the source; the two MIT references (`NOTICE`) may be read and, for tables, bit allocation and packing, adapted with a header line. ffmpeg, Aften and liba52 source is never opened; ffmpeg's binary is the acceptance oracle. |
| D3 | Real-time contract | **`new` allocates everything; `push`/`pull` allocate nothing, lock nothing, spawn nothing, touch no I/O, never panic.** Enforced by a counting allocator in tests, by `unsafe_code = "forbid"`, and by the absence of `std::sync`, `std::thread` and `std::io` from the library. |
| D4 | Determinism | **Byte-identical output on every platform** for the same input and config. No runtime transcendental in the hot path: the window, the MDCT twiddles and every allocation table are committed constants; f32 arithmetic runs in one fixed order; no fast-math. CI hashes the probe's output on three OSes. |
| D5 | Speed without `unsafe` | **One thread, no intrinsics, fixed-size arrays the compiler vectorises.** Target ≥ 100× real time for 5.1 on one core (a host needs 1×; the vuio fork reaches 215× the same way). rayon is not an option on an audio thread. |
| D6 | Where the bits go | **Masking curve once per channel-block per frame, a bit tally per candidate offset, a coarse frame-wide search followed by per-channel fine offsets.** The per-channel step targets the measured imbalance of the references (FL over-served, FR and LFE starved). §5. |
| D7 | Coupling | **On for every layout, start band by bitrate**, chosen against the ffmpeg stream's own decisions as read by `probe dump` (its bitstream is data, not source). §5.6. |
| D8 | Block switching | **Implemented from the start**, driven by the §8.2.2 transient detector, on by default; the listening pass may still turn it off. Measured: the burst channels' SNR does not move either way. ffmpeg never uses it, and a waveform SNR on bursts cannot judge it. §6. |
| D9 | 7.1 on E-AC-3 | **Independent substream = 5.1 with the surrounds folded; dependent substream = four channels (Ls, Rs replaced; Lrs, Rrs added).** A 5.1-only decoder plays the independent substream alone and hears everything. Verified in P2 by ffmpeg's replace-or-extend decode. §7.3. |
| D10 | E-AC-3 tools | **None of SPX, AHT, enhanced coupling or TPNP.** The quality comes from the AC-3 core; the E-AC-3 features we need are `bsid` 16, the 1/2/3/6-block frame lengths and the substream pair. The tools can come later behind the same tests. |
| D11 | Object audio | **Inside the crate behind the `joc` feature, off by default; dynamic objects designed and built in full, not only beds.** The feature keeps the module out of every default build and out of punktfunk until the gate clears; because a published crate ships every source file, no tag that contains the module is published before the answer (§12). Wine delivers beds and no objects today; the object path is built against synthetic scenes so the host changes one call when Proton catches up. §8. |
| D12 | Licence and names | **MIT OR Apache-2.0**, adapted portions keep their MIT notice. "Dolby", "Dolby Digital" and "Atmos" appear nowhere in the name, the API or a headline; the formats are AC-3 / E-AC-3 (ATSC A/52) and object audio (ETSI TS 103 420). §12. |
| D13 | API | **`Config` (codec, layout, bitrate) validated once; `Encoder::push`/`pull`; `Frame` borrowed.** Frame size and delay are queries on `Config`, so a host sizes datagrams and A/V sync before it builds an encoder. The only E-AC-3 parameter is the frame length, because only the host knows its path MTU and latency need. §3. |
| D14 | Toolchain | **MSRV 1.96, edition 2024, no dependencies in the library.** Dev-dependencies: `oxideav-ac3` and `oxideav-core`, pinned (§9). |

## 1. What the consumer wants

The punktfunk host, in call order:

1. **Can I offer this?** — `Config::bitrates(codec, layout)` and `Config::frame_bytes()`: the
   ladder, and whether a frame fits one datagram.
2. **How far behind video will audio be?** — `Config::delay_samples()`.
3. **Here is 5 ms of audio.** — `push(&[f32])`, interleaved, WAV order, any chunk.
4. **Give me the frame.** — `pull()`, whole syncframes, one access unit at a time.

The contract, from the host's side:

| | |
|---|---|
| Input | Interleaved f32, 48 kHz, WAV/SMPTE order FL FR FC LFE BL BR (SL SR), in whatever chunks capture delivers (5 ms = 240 samples today). Non-finite samples read as silence. |
| Output | Whole syncframes. 7.1 E-AC-3 yields an access unit of two syncframes with their byte ranges, so the host packs one or both per datagram. |
| Datagram | One syncframe per QUIC datagram, never fragmented. A 1,500-byte path leaves ~1,400 bytes: AC-3 5.1 fits at 320 kbps (1,280 bytes); 448 and 640 (1,792 and 2,560) need a jumbo path. E-AC-3's short frames fit far higher rates (§7.2). |
| Delay | `frame_samples + 256`: 1,792 samples (37.3 ms) on AC-3; 512 (10.7 ms) on a one-block E-AC-3 frame. The receiver's overlap-add adds 256 of its own. |
| Thread | The host's audio thread. D3. |
| Errors | `Config::new` and `Encoder::new` validate everything; `push` fails only on a wrong slice length or an unread frame; nothing panics. |

Channel order, once, in `push`:

| WAV slot | 2.0 | 5.1 → AC-3 `acmod` 2/0 or 3/2 | 7.1 → E-AC-3 I0 (3/2 + LFE) and D0 |
|---|---|---|---|
| 0 FL, 1 FR | L, R | L, R | L, R |
| 2 FC, 3 LFE | — | C, LFE | C, LFE |
| 4 BL, 5 BR | — | Ls, Rs | D0: Lrs, Rrs; folded into I0's Ls, Rs |
| 6 SL, 7 SR | — | — | D0: Ls, Rs (replace I0's) |

Client passthrough is the monorepo's work, listed here only so the output shape is right for
it: Android/Fire TV `AudioTrack` with `ENCODING_AC3`/`ENCODING_E_AC3`, tvOS
`AVSampleBufferAudioRenderer`, Windows WASAPI exclusive `WAVE_FORMAT_DOLBY_AC3_SPDIF`, Linux
IEC 958 passthrough, webOS/Tizen native players. All take an elementary stream of whole
syncframes; IEC 61937 framing, where needed, is the sink's.

## 2. Structure

### 2.1 Pipeline

```
push(f32, WAV order)
  │ reorder → per-channel ring: 256 samples of history + one frame + one chunk
  ▼
transient detector (§8.2.2), per block per full-bandwidth channel ──► blksw[ch][blk]
  ▼
window (Table 7.33) + MDCT 512 or 256-pair (§8.2.3)                     mdct
  ▼
coupling: shared channel above the start band + coordinates (§7.4)     ac3
  ▼
exponents: raw → strategy per block → slope limit → groups (§7.1)      exponents
  ▼
bit allocation core: psd → bndpsd → excitation → mask (§7.2.2)         bitalloc
  ▼
tuner: csnroffst, fsnroffst[ch], cplfsnroffst, lfefsnroffst            tuner
  ▼
mantissas: quantise, group, dither flags (§7.3)                        mantissas
  ▼
pack: syncinfo + BSI + audblk × n + auxdata + CRC (§5, §7.10)          pack, bits, tables
  ▼
E-AC-3 only: bsid 16 BSI, audfrm, numblkscod, substream pair (Annex E) eac3
  ▼
pull() → Frame { bytes, syncframes, pts }
```

Everything between `push` and `pull` runs inside `push`, once per completed frame, on the
caller's thread. Nothing is retained across frames except the 256-sample history per channel,
the transient detector's filter state, the sample counter, and the last frame's bytes.

### 2.2 Modules

| Module | Owns | Spec | Adapted from a reference? |
|---|---|---|---|
| `tables` | window, bit-allocation tables, quantizer tables, frame sizes, band/mask tables, MDCT twiddles — all `const` | Tables 5.18, 7.x, §7.2.2 | normative data; may be transcribed from either reference, header line if so |
| `bits` | bit writer over the frame buffer, fixed capacity, overflow is an error | — | no |
| `mdct` | TDAC fold + DCT-IV by half-length FFT, long and short; the O(N²) sum as test oracle | §8.2.3 | the decomposition is textbook; vuio's `mdct.rs` is the exemplar for the twiddle convention |
| `exponents` | raw exponents, per-block strategy, ±2 slope limit, grouped coding | §7.1, §8.3 | no |
| `bitalloc` | PSD, banded PSD, leak filters, threshold, mask — once per channel-block | §7.2.2 | the `logadd`/`calc_lowcomp` arithmetic is the spec's pseudo-code; expect a header line |
| `tuner` | the search over SNR offsets with a bit tally per candidate | §7.2.2.5, §8.3 | the tally idea is vuio's; written fresh |
| `mantissas` | quantizers per `bap`, the three grouped classes, dither | §7.3 | no |
| `pack` | syncinfo, BSI, audblk syntax order, auxdata, CRC-16 (poly 0x8005, crc1 over the first 5/8, crc2 over the rest) | §5.3–5.4, §7.10 | possibly the CRC tables |
| `ac3` | the six-block frame, coupling decisions, the channel map, the per-frame driver | §7.4, §8 | no |
| `eac3` | `bsid` 16 syntax, `numblkscod`, audfrm, I0 + D0 pair with `chanmap` | Annex E | no |
| `lib` | `Config`, `Encoder`, `Frame`, `Error` | — | no |

Dependencies point downward: `lib → ac3/eac3 → {exponents, bitalloc, tuner, mantissas, pack}
→ {mdct, bits, tables}`. `tables` imports nothing.

### 2.3 State and memory

Per channel: a 256-sample history, a frame plus one chunk of input (1,536 + 240 at most for
AC-3), 6 blocks × 256 coefficients, 6 × 256 exponents and `bap` values, 6 masking curves of 50
bands. For 7.1 (8 channels + a coupling channel) that is on the order of 200 KiB, allocated in
`new`, plus the 4 KiB output frame (8 KiB for the substream pair). The encoder is `Send`, not
`Sync`; the host owns one per session.

## 3. Public API

`crates/omur/src/lib.rs` carries this today, with `push`/`pull` unimplemented; everything
else is live and tested.

```rust
pub const BLOCK_SAMPLES: usize = 256;
pub const SAMPLE_RATE_HZ: u32 = 48_000;

pub enum Codec { Ac3, Eac3 { blocks: Blocks } }
pub enum Blocks { One, Two, Three, Six /* default */ }
pub enum Layout { Stereo, Surround51, Surround71 }

pub struct Config { /* codec, layout, bitrate_kbps — private, validated */ }
impl Config {
    pub fn new(codec: Codec, layout: Layout, bitrate_kbps: u32) -> Result<Config, Error>;
    pub fn bitrates(codec: Codec, layout: Layout) -> &'static [u32]; // the ladder, empty = cannot
    pub fn frame_samples(&self) -> usize;   // 1536, or 256 × blocks
    pub fn frame_bytes(&self) -> usize;     // one access unit; E-AC-3 rounds to whole words
    pub fn delay_samples(&self) -> usize;   // frame_samples + 256
}

pub enum Error { Bitrate { kbps: u32 }, Layout, Backlog, Length }

pub struct Frame<'a> { pub bytes: &'a [u8], pub syncframes: &'a [Range<usize>], pub pts: u64 }

pub struct Encoder { /* everything, allocated once */ }
impl Encoder {
    pub fn new(config: Config) -> Result<Encoder, Error>;
    pub fn push(&mut self, pcm: &[f32]) -> Result<(), Error>; // any chunk, WAV order
    pub fn pull(&mut self) -> Option<Frame<'_>>;              // the frame push completed, once
}
```

The shape is deliberate: `push` encodes, `pull` hands over. A frame completes inside the push
that delivers its 256-sample lookahead; the host pulls after every push, and a push that would
complete a second frame before the first was pulled returns `Error::Backlog` rather than
dropping or growing — the host's bug is surfaced, not buffered. There is no `flush`: a stream
has no end, and a file tool pads its own tail with silence (the probe does).

Object input is the `joc` module (§8.4), compiled only with the feature.

## 4. Real-time thread rules

| Rule | How it holds |
|---|---|
| No allocation after `new` | every buffer is sized from `Config` in `new`; a test wraps `push`/`pull` in a counting global allocator and asserts zero |
| No locks, threads, I/O | the library imports none of `std::sync`, `std::thread`, `std::io`; `cargo deny`-style grep in CI |
| No panic | `unsafe_code = "forbid"`; no `unwrap`/`expect`/indexing on data-dependent indices without a bound the type carries; the fuzzer feeds arbitrary configs and arbitrary f32 (NaN, ±inf, 1e30) for hours |
| Deterministic | D4; CI compares a sha256 of the probe's 10-second encode across Linux, Windows and macOS |
| Bounded time | every loop is over a fixed array; the tuner's search is a bounded binary search plus a bounded refinement (§5.2); worst case is measured by the probe and stated in the README |
| Non-finite input | clamped to 0 at `push`; a game that emits NaN gets silence, not a broken frame |

## 5. Bits: allocation and the tuner

This is where quality is decided, and where the references fall short. The measured facts
(2026-10-23, Apple M3 Ultra, ffmpeg 9.0.1 as decoder; the 10-second 5.1 signal of §9.2,
SNR in dB per channel after an ffmpeg decode):

| Encoder @ kbps | FL | FR | LFE | SL | FC | SR | Speed, one thread |
|---|---|---|---|---|---|---|---|
| ffmpeg `ac3` @ 448 | 40.5 | 30.5 | 35.4 | 23.2 | 6.3 | 6.9 | ~170× incl. process start |
| ffmpeg `ac3` @ 320 | 36.4 | 26.6 | 32.6 | 20.5 | 3.7 | 4.0 | |
| vuio fork @ 448 | 45.8 | 28.1 | 34.3 | 19.9 | 2.6 | 4.2 | 215× |
| vuio fork @ 320 | 37.0 | 20.2 | 25.7 | 17.5 | 1.8 | 2.0 | |
| oxideav-ac3 @ 448 (640 identical) | 19.0 | 16.6 | 6.2 | 9.2 | −0.2 | −1.2 | 1× |

FL is a harmonic tone with tremolo, FR a log sweep, LFE 40 + 80 Hz, SL pink noise, FC
speech-like noise bursts, SR gunshot-like bursts. The gain-corrected SNR matched within 0.3 dB
everywhere except the burst channels. Read it as: the fork's bit *count* is fine (640 kbps
scales to FL 55.8 / FR 37.3 / LFE 45.9) and its bit *placement* is not — FL takes what FR and
LFE needed. oxideav spends 43 % of its time in an O(N²) MDCT with a `cos()` per term, tunes
by brute force, and pads everything above 448 kbps.

### 5.1 The §7.2.2 chain

For each channel-block, from the exponents alone: `psd = 3072 − (exp << 7)` per bin; banded
PSD by log-add over the 50 bands; the excitation function with its fast and slow leak
(`fdcycod`, `sdcycod`, `sgaincod`, `dbpbcod`, `floorcod`, fixed for the stream) and the
low-band compensation; the masking curve as the maximum of excitation and the hearing
threshold for `fscod`. Then, and only then, the SNR offset: `address = (psd − mask +
snroffset) >> 5`, clamped to 0..63, through `baptab` to a `bap` of 0..15. The offset is
`(((csnroffst − 15) << 4) + fsnroffst) << 2`, with `csnroffst` shared by the frame and
`fsnroffst` per channel, per coupling channel and per LFE.

The split matters: everything up to the mask is computed **once** per channel-block per
frame; the search touches only the last step.

### 5.2 The search

A frame has a fixed bit budget (`frame_bytes × 8` minus syncinfo, BSI, block side
information, exponents, CRC). The tuner's job is the largest offset set whose mantissa bits
fit.

1. **Coarse**: binary search `csnroffst` over 0..63 with every `fsnroffst` at a fixed
   midpoint — six evaluations, each a tally over every channel-block.
2. **Fine, per channel**: with `csnroffst` fixed, walk each channel's `fsnroffst` over
   0..15 and keep the set that fits; order channels by demand (§5.3) so the starved ones are
   served first.
3. **Fill**: the remaining bits, if any, go to the channel whose next step costs least.

Each evaluation is a *tally*, not a `bap` array: the three grouped quantizer classes (three
`bap` 1 values to 5 bits, three `bap` 2 to 7, two `bap` 4 to 7) are a function of counts, so a
block's mantissa cost is four counters per channel. That idea is the vuio fork's, and it is
the reason the fork is 200× faster than its parent; ours is written from the spec's grouping
rule with the same conclusion.

### 5.3 Per-channel demand

A single frame-wide offset gives every channel the same margin against its own masking
curve, which is the right thing *perceptually* and the wrong thing when one channel's mask
is cheap to satisfy. The fork's FL/FR/LFE spread is exactly that. Two instruments:

- `fsnroffst[ch]` (§5.4.3.40) — per-channel fine offsets, which the references leave flat.
- `fgaincod[ch]` — the fast-gain code, which shapes how much a channel's own peaks mask
  their neighbourhood; a tonal channel tolerates a lower gain.

Demand is estimated per frame from the exponent spectrum: tonal, slowly varying channels
(small block-to-block exponent change, peaky band PSD) are served last; noise-like and
low-band channels first. The LFE's seven bins always get their own offset. The P1 target
(§11) is the measure of whether this is enough; if it is not, the next instrument is delta
bit allocation (§7.2.2.6, `deltba`), per band.

### 5.4 Exponents

Raw exponents are `-log2` of the coefficient magnitude, clamped to 0..24. Per block, one of
D15 (one exponent per bin), D25 (per pair), D45 (per quad) or reuse of the previous block's;
block 0 of every frame must send fresh exponents. The choice trades exponent bits (D15 on
253 bins ≈ 600 bits per channel-block) against the quantisation loss of a coarser envelope.
The decision is per channel from the block-to-block exponent difference against two
thresholds, with a fixed menu of strategy sequences over the frame's six blocks; ffmpeg's
sequences, read from its streams with `probe dump`, are the first menu to try. The slope
limit (an exponent may rise by at most 2 per group step) is applied before coding, downwards
only, so the envelope never under-states a coefficient.

### 5.5 Mantissas

Quantizer per `bap`: `bap` 1, 2, 3 are 3-, 5-, 7-level symmetric; 4 and 5 are 11- and
15-level symmetric; 6..15 are 5- to 16-bit asymmetric two's complement. `bap` 0 sends nothing
and the decoder dithers unless `dithflag` is clear — we clear it for tonal channels. Grouping
state runs across channels within a block and resets per block.

### 5.6 Coupling

Above the coupling start band, the full-bandwidth channels share one spectrum plus per-band
coordinates (`cplco`), which is the single largest saving in 5.1. Decisions: coupling on for
every layout; the start band (`cplbegf`, bins `37 + 12·cplbegf` at 93.75 Hz per bin, so
`cplbegf = 8` is 12.5 kHz, not the 6 kHz a reference comment claims — read code, not
comments) chosen per bitrate from a small table; coordinates re-sent when any band's
coordinate moves by more than 3 dB (one step resent in nearly every block and cost FL
0.5 dB); phase flags per band in 2/0 when the cross-spectrum says L + R cancels (an
anti-phase 15 kHz tone: 3 dB without them, 63 dB with); the centre included. The table is
what ffmpeg's streams do at each rate, read by `probe dump`.

## 6. Transients and block switching

The §8.2.2 detector: a high-pass at 8 kHz, then peak ratios over 256 / 2 × 128 / 4 × 64
segments against the spec's thresholds (0.1, 0.075, 0.05) and a silence floor. A hit sets
`blksw[ch]` and the block's 512 windowed samples go through two 256-point transforms,
halving the time spread of quantisation noise from 10.7 ms to 5.3 ms. The decoder applies the
same window either way (§7.9.4), so there are no transition windows to get wrong.

A waveform SNR on a gunshot cannot see pre-echo. The P1 listening pass (§9.4) is the only
judge: ours with and without switching, against ffmpeg (which never switches), on real game
captures. The default follows the result.

## 7. E-AC-3

### 7.1 What changes

`bsid` 16. The syncframe loses `crc1` and `frmsizecod`, gains `strmtyp`, `substreamid`, an
11-bit `frmsiz` in 16-bit words (so a syncframe is at most 4,096 bytes), `numblkscod`, and
an `audfrm` strategy block; the audio block syntax gains the tool flags (all off, D10). Base
AC-3 decoders must mute on `bsid` > 10, which is why optical stays AC-3.

### 7.2 Frame length

`numblkscod` picks 1, 2, 3 or 6 blocks per syncframe. For 5.1, bytes per syncframe and the
encoder delay:

| blocks | samples | delay | 448 kbps | 640 | 768 | 1024 |
|---|---|---|---|---|---|---|
| 6 | 1536 | 1792 (37.3 ms) | 1792 | 2560 | 3072 | 4096 |
| 3 | 768 | 1024 (21.3 ms) | 896 | 1280 | 1536 | 2048 |
| 2 | 512 | 768 (16.0 ms) | 598 | 854 | 1024 | 1366 |
| 1 | 256 | 512 (10.7 ms) | 298 | 426 | 512 | 682 |

Two things fall out. Short frames are a **latency** option for HDMI clients, and they are
also the **MTU** option: 640 kbps at three blocks and 1,024 kbps at two blocks both fit a
1,400-byte datagram that AC-3 at 384 kbps does not. The cost is header and, above all,
exponents: block 0 of every syncframe sends them fresh, so a one-block stream has no reuse at
all and may spend half its budget on envelopes. P2 measures the SNR cost of each length at
448 and 640 kbps and reports the table; measured (README), three blocks is the
recommendation: 1,280 bytes at 640 kbps, about 1 dB on FR and the LFE.

SNR offsets travel at frame level (`snroffststr` 0): ffmpeg 9.0.1 reads the per-block
`snroffste` bit in block 0 and in no later block, the reverse of E.1.2.4, so the per-slot
strategies decode wrongly in every ffmpeg-based player. The per-channel step AC-3 takes is
made up with per-channel fast gain codes (`frmfgaincode`), which both decoders read; a block
after a raised one resends its codes, because ffmpeg keeps them where the spec resets to 4.

Rates: an E-AC-3 frame is whole words, so a rate rounds to the nearest word (at most
1.5 kbit/s); `Config::frame_bytes` reports the rounded size. The ladder in `Config::bitrates`
is a proposal until P2.

### 7.3 7.1: two substreams

The independent substream I0 carries 3/2 + LFE; the dependent substream D0 follows it in the
same access unit with `chanmap` naming four channels: Ls and Rs (which *replace* I0's) and
Lrs and Rrs (which *extend* it). I0's Ls and Rs are the fold of side and back (each at −3 dB,
headroom reserved), so a 5.1 decoder that ignores D0 hears a correct downmix, and a 7.1
decoder restores the four discrete surrounds. Two coded channels would be cheaper; the fold
is what makes the stream safe on every receiver, and it is what ffmpeg's decoder expects.
The budget splits across I0 and D0 by coded channel count, rounded to words, each under the
4,096-byte ceiling; `Frame::syncframes` gives the host both ranges.

### 7.4 Compatibility

Decoders are required to accept every `numblkscod`. Receivers that advertise E-AC-3 over
ARC/eARC accept `bsid` 16 with dependent substreams; ffmpeg decodes the pair to eight
channels with identity mapping and the independent substream alone to the fold. The
on-glass check is the maintainer's AVR (a Denon AVC-X3800H sits on the LAN) reporting the
input as 7.1 — owed.

## 8. Object audio — feature `joc`

### 8.1 What the host captures today

The brief settles that Wine/Proton delivers spatial audio on a Linux host and asks only for
the shape of that data. From `dlls/mmdevapi/spatialaudio.c` (upstream Wine `master`, Proton
10.0 and Valve's `bleeding-edge` are the same code here, 2026-10-23):

| | |
|---|---|
| Static objects (the bed) | `ActivateSpatialAudioObject(type)` for any of FL FR FC LFE SL SR BL BR TFL TFR TBL TBR BC. Mixed into one WASAPI shared-mode float render stream whose channel mask is the set the game asked for — 7.1.4 is 12 channels. The four `Bottom*` types are accepted and silently dropped. |
| Dynamic objects | `GetMaxDynamicObjectCount` returns 0; `ActivateSpatialAudioObject(Dynamic)` returns `SPTLAUDCLNT_E_NO_MORE_OBJECTS`; `SetPosition` and `SetVolume` are `E_NOTIMPL`. None yet. |
| Position metadata | None. Bed positions are implied by the channel. |
| Update rate | The device period: 10 ms (480 frames at 48 kHz) per `BeginUpdatingAudioObjects`. |
| Format | 32-bit float, 48 kHz, one channel per object. |

So what reaches the host today is a **height bed, up to 7.1.4 (+BC), as PCM channels**. A
7.1.4 bed carried as bed objects over a 5.1 core is already what a receiver calls object
audio, so beds are the first thing the module encodes — and dynamic objects are designed in
full below and built in the same phase, so the day Proton delivers them the host changes one
call.

Host prerequisite (Q2): the capture sink must offer a 12-channel map, or the audio server folds
Wine's 7.1.4 stream to 7.1 before omur sees it. A Windows host sees none of this: a game's
objects go to Windows' own spatial engine and only its rendered mix is capturable; objects
there would need a spatial-capable virtual endpoint, outside this crate (Q3).

### 8.2 What arrives once Proton adds dynamic objects

The interface is fixed by `spatialaudioclient.idl`, so the shape is known before the
implementation exists:

| | |
|---|---|
| Count | `GetMaxDynamicObjectCount`; Windows' own spatial formats allow 16–32 per stream. The module accepts up to 32 and reduces to the stream's budget (§8.3). |
| Signal | one mono f32 buffer per object per period (`GetBuffer`), 48 kHz, 10 ms at the default device period. |
| Position | `SetPosition(x, y, z)` at most once per period: metres relative to the listener, right-handed, +x right, +y up, +z behind the listener. |
| Gain | `SetVolume(v)`, linear, at most once per period. |
| Lifetime | `ActivateSpatialAudioObject(Dynamic)` … `SetEndOfStream(frames)`; `IsActive` says whether it still produces audio; an index is reused after its end. |

The host forwards, per object and chunk, the PCM and the (position, gain) pair with the
sample at which it took effect. The crate's `ObjectUpdate` takes a listener-centred unit
cube — +x right, +y front, +z up, each in −1..1 with 1 at the speaker wall — so the host's
conversion from Windows is `[x, −z, y] / r` for its chosen room radius `r`: one line,
documented beside the call.

### 8.3 What the crate encodes

TS 103 420 carries object audio in an E-AC-3 stream as a 5.1 core (the thing every receiver
plays) plus, in the audio blocks' skip fields, an EMDF container (sync `0x5838`) with two
payloads: **OAMD** (object metadata: bed assignment, positions, gains, sizes, per-update
timing) and **JOC** (per element, per parameter band, the matrix that reconstructs it from
the core, with one or two data points per frame and Huffman-coded coarse or fine
coefficients). `addbsi` signals the extension (type A) so a receiver knows to look. A
receiver reconstructs up to 16 elements in a 64-band QMF domain and renders them to its
speakers; a 5.1 receiver plays the core and never notices.

The encoder, per frame:

1. **Element budget.** A stream carries 16 elements. The bed's non-LFE channels take theirs
   (11 for 7.1.4; the LFE goes to the core directly); dynamic objects fill the rest — five
   with that bed, fifteen with a 2.0 bed. When the host offers more than fit, the frame's
   loudest objects (energy × priority) keep their element and each remaining object is
   folded into the nearest kept element by position, energy preserved, the keeper's metadata
   unchanged — so a 32-object game still renders, with its quietest sources merged.
2. **Render the core.** A 5.1 panner (amplitude panning over the floor ring, heights folded
   at −3 dB) places every element; gains are interpolated sample by sample between updates so
   a moving object leaves no clicks in the core. The core goes through an `omur` E-AC-3 5.1
   encoder.
3. **QMF analysis** of every element and of the rendered core (the spec's 64-band bank; the
   prototype filter is a committed table).
4. **Matrices** per data point and parameter band: the least-squares solution of element ≈
   matrix × core over the band's samples (a 6 × 6 solve per element and band), regularised;
   sparse mode when one core channel dominates. One data point when nothing moved; two with
   a steep switch at the update's timeslot when an object moved by more than a quantisation
   step or a transient landed mid-frame; `Reuse` when the matrix is unchanged.
5. **OAMD.** Program assignment from the bed set; per update block the timing (sample offset
   in the frame, ramp duration to the next update) and per element the active flag, gain,
   priority, quantised position and size, with zone constraints and screen snap off. A host
   update every 10 ms is three or four blocks per 32 ms frame, within the syntax's limit; bed
   elements carry their fixed position in every frame so a receiver that joins mid-stream
   places them.
6. **EMDF** → bytes → the core frame's skip fields, with `addbsi` signalling the extension.
   The core budget is reduced by the payload: a few hundred bits per frame for a bed, a few
   thousand with sixteen moving elements at two data points.

### 8.4 API

Compiled only with `--features joc`; in `lib.rs` today as signatures.

```rust
pub mod joc {
    pub enum BedChannel { FrontLeft, FrontRight, FrontCenter, Lfe, SideLeft, SideRight, BackLeft, BackRight,
                          TopFrontLeft, TopFrontRight, TopBackLeft, TopBackRight, BackCenter }
    pub struct Bed { /* the set; push_bed takes samples interleaved in BedChannel order */ }
    pub const MAX_ELEMENTS: usize = 16;  // bed channels and objects together
    pub const MAX_OBJECTS: u8 = 32;      // what a host may push; the rest fold (§8.3)

    pub struct ObjectConfig { pub bed: Bed, pub objects: u8, pub bitrate_kbps: u32, pub blocks: Blocks }
    pub struct ObjectUpdate { pub position: [f32; 3], pub gain: f32, pub size: f32, pub at_sample: u32 }

    pub struct ObjectEncoder { /* an omur 5.1 E-AC-3 encoder plus QMF, matrices and metadata state */ }
    impl ObjectEncoder {
        pub fn new(config: ObjectConfig) -> Result<ObjectEncoder, Error>;
        pub fn push_bed(&mut self, pcm: &[f32]) -> Result<(), Error>;
        pub fn push_object(&mut self, index: u8, pcm: &[f32], update: Option<ObjectUpdate>) -> Result<(), Error>;
        pub fn end_object(&mut self, index: u8);      // silent until pushed again
        pub fn pull(&mut self) -> Option<Frame<'_>>;
    }
}
```

One chunk is one `push_bed` plus a `push_object` per live object with the same sample count,
then `pull`; an object not pushed in a chunk is silent for it; `at_sample` is the offset
within the chunk at which the update took effect. §4 holds unchanged: everything is allocated
in `new`, sized by the bed and `objects`.

What the spec settled when P3 was built, against the plan above:

- Six-block frames only: clause 5.3.2 defines the frame offset against 1,536 samples, so
  `ObjectConfig::blocks` must be `Six`.
- A bed element sends no position: clause 5.5.9 places it by its label; BackCenter has no
  label and goes out as an object at its fixed place.
- One container per frame, in one skip field of at most 511 bytes; a scene that needs more
  (sixteen moving elements at two data points) steps its elements down.
- The element count excludes the LFE; a 7.1.4 bed is 11 elements, five objects fit beside it.
- Up to eight timed updates per frame on a 32-sample grid, each ramping within the frame.
- Sparse coding only for a one-band element; everything else dense, fine or coarse.
- `codecdatae` follows TS 103 420 Table 56, where A/52 Annex H says 0: the receiver check
  decides (§14).

### 8.5 Verification and gates

- oxideav-ac3's opt-in JOC decoder reconstructs the elements and parses the OAMD: per-element
  SNR against the input, positions read back within one quantisation step; the core judged by
  ffmpeg like any 5.1 stream.
- Synthetic scenes, because no host delivers objects yet: a pink-noise object circling the
  listener once every four seconds, a second one jumping between corners (steep), a speech
  bed, and a 32-object scene that exercises the fold. `probe scene` writes them and the
  stream.
- On glass: the Denon reports the stream as object audio, the height speakers carry the height
  bed, and the circling object pans across the room.
- Measured (README): bed channels 61 dB, heights 41 dB (the fine quantiser's error on the
  −3 dB fold), moving pink-noise objects 25 dB — the core's own ceiling on pink noise; two
  broadband objects panned to the same spot cannot be separated from a 5-channel core by
  any matrix, so the scenes keep them apart.
- Release gate: no tag that contains `src/joc` is published, and no punktfunk build enables
  `joc`, until the maintainer has an answer from counsel or from the rights holder on the
  active patent family (§12).

## 9. Testing

### 9.1 Layers

| Layer | What | How |
|---|---|---|
| Unit | MDCT vs the O(N²) spec sum; exponent slope limit; grouping tally vs a written `bap` array; CRC vectors; frame sizes | `cargo test`, property tests on random blocks |
| Round trip | every frame parses, both CRCs verify, decodes to the input within an SNR floor | `oxideav-ac3` 0.0.11 as dev-dependency (`decoder::make_decoder`, `verify_packet_crc`, `eac3::decode_eac3_packet`, the typed `bsi::parse` to assert our headers), pinned |
| Acceptance | the SNR gate of §11 per channel, at each rate, against ffmpeg's encoder on the same signal | `tools/probe` + `ffmpeg` on `PATH`; CI on Linux, skipped where absent |
| Determinism | byte-identical output across OS and chunk size | sha256 of the probe's encode, compared across the CI matrix and across `CHUNK` sizes |
| Real time | zero allocations in `push`/`pull`; worst-case frame time | counting allocator test; the probe reports `× realtime` and the slowest frame |
| Fuzz | arbitrary config + arbitrary f32 (NaN, inf, huge, denormal) never panics and every frame decodes | `cargo fuzz`, three targets (`encode`, `bits`, `objects`), hours before a release |
| Listening | pre-echo, coupling artefacts | §9.4 |

Where the two decoders disagree, that is a finding, not a tie: ffmpeg is the authority and
oxideav is the convenience.

### 9.2 The probe

`tools/probe` is the harness that produced the §5 table, moved here unchanged except for the
workspace manifest; `omur` becomes its third encoder in P1.

```sh
cargo run --release -p omur-probe -- gen in.s16                  # 10 s of 5.1, s16le, WAV order
cargo run --release -p omur-probe -- enc in.s16 ox448.ac3 448    # the oxideav reference
cargo run --release -p omur-probe --no-default-features --features vuio --ignore-rust-version -- enc in.s16 vu320.ac3 320
ffmpeg -v error -y -f s16le -ar 48000 -ch_layout 5.1 -i in.s16 -c:a ac3 -b:a 320k ff320.ac3
ffmpeg -v error -y -i vu320.ac3 -f f32le -ac 6 vu320.f32         # decode any of them
cargo run --release -p omur-probe -- cmp in.s16 vu320.f32        # lag, channel map, gain, SNR per channel
```

P1 adds `dump <stream>`: per frame, `blksw`, the exponent strategies, `cplbegf`/`cplendf`,
`csnroffst`/`fsnroffst`, bits per channel — what any AC-3 stream's encoder decided, ffmpeg's
included. Reading a stream is not reading source.

### 9.3 Fuzzing

`fuzz/` with `cargo fuzz`: `encode` builds a `Config` from the first bytes, pushes the rest
as f32 in random chunk sizes and decodes every frame; `bits` drives the writer at its
capacity edge; `objects` does the same through `ObjectEncoder` with random beds, object
counts and updates. Crashes are fixed with a regression test; a release waits for a clean run.

### 9.4 Listening

Material: three 10-second captures from the maintainer's hosts — a shooter (gunshots, FC
dialogue), a racing game (broadband engines, LFE), and music with sustained tones. Conditions:
omur with and without block switching, ffmpeg at the same rate, the original. Method: the
probe writes randomised ABX pairs and a key; two listeners on the HDMI chain and on
headphones. Pass: no condition where omur is distinguishable from ffmpeg more often than
chance, and no audible pre-echo on the shooter clip at 448 kbps.

## 10. Repository

```
omur/
  Cargo.toml                 workspace: omur, omur-probe; unsafe_code forbidden; MSRV 1.96
  rust-toolchain.toml        1.96.0 + rustfmt + clippy
  README.md  CONTRIBUTING.md  CHANGELOG.md  NOTICE  LICENSE-MIT  LICENSE-APACHE
  docs/design.md
  crates/omur/src/
    lib.rs                   Config, Encoder, Frame, Error
    tables.rs bits.rs mdct.rs exponents.rs bitalloc.rs tuner.rs mantissas.rs pack.rs
    ac3.rs eac3.rs
    joc/{mod,render,qmf,matrix,oamd,emdf}.rs   feature joc, off by default (P3)
  tools/probe/src/main.rs    gen | enc | cmp | dump
  fuzz/                      P1
  .github/workflows/{ci,release}.yml
```

Dependencies: none in `omur`. Dev: `oxideav-ac3 = "=0.0.11"`, `oxideav-core = "=0.1.41"`.
The probe adds the vuio fork behind a feature (it declares rustc 1.98; `--ignore-rust-version`
builds it on 1.96).

## 11. Phases

| # | Delivers | Done when |
|---|---|---|
| **P1 — AC-3 5.1** | tables, bits, MDCT, exponents, allocation, tuner, mantissas, pack; 2.0 and 5.1; the probe's third encoder and `dump`; round trip, determinism, allocation and fuzz tests; `delay_samples` verified by the probe's lag | per-channel SNR on FL/FR/LFE/SL **within 1 dB of ffmpeg** at 320 and 448 kbps: ≥ 35.4 / 25.6 / 31.6 / 19.5 at 320, ≥ 39.5 / 29.5 / 34.4 / 22.2 at 448; 2.0 at 192 kbps within 1 dB of ffmpeg on FL/FR; ≥ 100× real time on one core; sha256 equal on three OSes and across chunk sizes; zero allocations after `new`; the §9.4 listening pass recorded with its verdict on block switching |
| **P2 — E-AC-3** | `bsid` 16 syntax, `numblkscod` 0..3, the I0 + D0 pair, 7.1 channel map, the rate split; the frame-length cost table | 5.1 at six blocks matches P1's numbers (same core); 7.1 at 1,024 kbps decoded by ffmpeg to eight channels with every channel within 1 dB of the 5.1 numbers for its role, and the folded I0 alone decoding to a correct 5.1; the 1/2/3-block cost table at 448 and 640 kbps published in the README with a recommended default (expected: a length that costs ≤ 1 dB and fits 1,400 bytes at 640 kbps); the Denon shows 7.1 |
| **P3 — object audio** | the `joc` module: element budget and fold, core renderer, QMF, matrix estimation, coarse/fine coding, OAMD with timed updates, EMDF, skip-field injection; beds and dynamic objects; `probe scene`; the 12-channel bed path end to end from a Wine game | a 7.1.4 bed and the synthetic scenes of §8.5 reconstructed by oxideav's JOC decoder with every element ≥ 20 dB SNR at 768 kbps, positions read back within one step, and the core within 1 dB of P2's 5.1; the 32-object scene folds without a click; a receiver on glass lights up object audio and drives height speakers; the counsel/rights-holder answer on file — **no tag with `src/joc` published before it** |
| **P4 — crates.io** | README stating what works, measured numbers and the recommended rates; docs.rs clean; `cargo publish --dry-run`; MSRV job; a clean fuzz run; `v0.1.0` | `omur` published from a tag that either predates `src/joc` or follows the P3 answer; the probe not; punktfunk's integration task opened with the codec id and the ladder |

Status 2026-10-23: P1, P2 and P3 are built on `main` and meet their numbers (README),
except the listening pass and the receiver checks, which need the maintainer; P4 waits on
those and on the patent answer, since `main` holds `src/joc`. P1 came to ~3.5k lines of
library plus tests, P2 ~1.5k, P3 ~5k.

## 12. Credits, licences, patents

- Code: MIT OR Apache-2.0. Adapted portions carry a header line and the `NOTICE` names both
  references and their authors: oxideav-ac3 (Karpelès Lab Inc., Mark Karpelès, MIT) and
  vuio-codec-ac3 (vyrti, MIT OR Apache-2.0). The README credits both.
- Specifications: ATSC A/52:2018 (= ETSI TS 102 366; Annex E is E-AC-3) and ETSI TS 103 420.
  Both are free downloads.
- Patents, not a legal opinion: AC-3's last patents expired in 2017. E-AC-3's last US patent
  (US 7,516,064) reportedly expired on 2026-01-30; other countries vary. Object coding —
  "Efficient coding of audio scenes comprising audio objects", US 9,756,448 and
  US 11,270,709, priority May 2013 — shows as active, hence D11 and P3's gate. ffmpeg proves
  nothing here: it is LGPL, its legal page leaves patents to distributors, and Fedora's build
  left E-AC-3 out over this question.
- Trademarks: never "Dolby", "Dolby Digital" or "Atmos" in the crate name, the API or a
  headline. "Plays on Dolby Digital receivers" in a sentence is fine.
- Release gates: P1's SNR table and listening verdict before any tag; no tag that contains
  `src/joc` is published, and no punktfunk build enables `joc`, before P3's answer.

## 13. Prior art

- **oxideav-ac3 0.0.11** (MIT) — pure-Rust AC-3/E-AC-3 decoder and encoder, 7.1 via a
  dependent substream, SPX/AHT/enhanced-coupling encoders, JOC/OAMD decode with a stereo
  renderer; its decoder passes its own corpus at 86–92 dB PSNR. Its encoder is what the §5
  table shows. Our decoder for tests.
- **vuio-codec-ac3 0.0.1** (MIT OR Apache-2.0) — a fork of the above: DCT-IV MDCT through an
  FFT, a cached masking curve and a packed bit tally that make its tuner fast, rayon fan-out,
  every lint off. The exemplar for §5.2's cost model; not a dependency of the library.
- Both are single-maintainer, six months and four days old, ~100k lines, largely
  machine-written; comments in them are not evidence (§5.6).
- **ffmpeg's `ac3`/`eac3` encoders** — the quality bar, used as a black box. The `eac3`
  encoder stops at 5.1, so 7.1 is judged by its decoder, not its encoder. It never uses short
  blocks.
- **Aften, liba52** — GPL/LGPL; never opened.

## 14. Open questions for the maintainer

1. **Name**: `omur` (recommended), `hljomur` or `dynur`? The skeleton says `omur`; a rename
   is a find-and-replace before the first push.
2. **The 12-channel capture sink**: Wine renders a 7.1.4 bed as a 12-channel float stream;
   unless punktfunk's capture sink (pf-audio) offers that map, the audio server folds it to
   7.1 and P3 has nothing to encode beyond 7.1. Is that sink change in scope alongside P3?
3. **Windows hosts**: a game's objects there go to Windows' spatial engine and only the
   rendered mix is capturable; objects would need a spatial-capable virtual endpoint (an
   APO), outside this crate. Confirm Linux/Proton is the object source for now.
4. **The bitrate ladder** (`Config::bitrates`): the proposal is 256–640 for AC-3 5.1,
   256–1024 for E-AC-3 5.1, 640–1536 for 7.1, 128–320 for 2.0. Trim or extend before the
   host's settings bake it in?
5. **7.1 dependent substream**: four channels with a folded I0 (D9) costs more than two but
   is safe on every receiver. Confirm, or accept 5.1 receivers losing the backs.
6. **`dialnorm` −31 and no DRC** (D1): confirm that the host never needs loudness metadata.
7. **Listening material and listeners** for P1's pass (§9.4): which hosts and games, and who
   besides you. It also decides two things the SNR cannot: whether block switching stays on
   (D8), and whether per-channel fast gain (`fgaincod` 5–7 on tonal channels, worth
   5–16 dB of waveform SNR on FL/FR/LFE) is adopted for AC-3 as it is for E-AC-3.
8. **Who asks counsel**, and about what exactly: the object-coding family only, or E-AC-3
   outside the US as well, given where the host is distributed (Flathub, the App Stores).
9. **The reference encoders in the probe**: keep oxideav and the vuio fork as permanent
   baselines, or drop them now that omur beats both?
10. **Receiver checks** on the Denon: 7.1 E-AC-3 shows as 7.1; a `probe scene` stream lights
    up object audio and drives the height speakers; and whether it accepts `codecdatae` as
    TS 103 420 writes it (§8.4).
