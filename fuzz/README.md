# Fuzzing

Four [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) targets (`docs/design.md` §9.3).
The crate stands outside the workspace and builds on nightly.

| Target | What it holds to |
|---|---|
| `encode` | Any AC-3 or E-AC-3 config, push schedule and f32 (NaN, ±inf, ±1e30, denormals, ±1.0, silence): every frame has `frame_bytes`, its CRCs, our header on each syncframe, and decodes in oxideav-ac3 to finite samples within ±4 (full-scale squares come back past 2 through coupling). |
| `chunking` | The same input under two push schedules gives byte-identical frames at the same pts. |
| `api` | Random `push` lengths, `pull`, `set_extension` and `Config::new` triples keep the documented errors (`Length`, `Backlog` without consuming, `Layout`, `Bitrate`); a shadow encoder fed only the accepted calls makes the same frames. |
| `objects` | `ObjectEncoder` (feature `joc`) under any bed, up to 32 objects, any chunk length, hostile updates (NaN, ±inf, 1e30 positions, gains and sizes; any `at_sample`) and f32: every frame is one 5.1 syncframe with its CRC, `addbsi` naming the objects, an EMDF container whose JOC and OAMD payloads oxideav-ac3 parses, a core it decodes to finite samples, and a dynamic program its JOC renderer renders. |

## Run

From the repository root, with `cargo install cargo-fuzz --version 0.13.2 --locked`:

```sh
cargo +nightly fuzz run encode -- -max_total_time=600
cargo +nightly fuzz run encode fuzz/corpus/encode fuzz/seeds/encode -- -max_total_time=600
```

The second line also reads the checked-in seeds; new inputs land in the first directory, so
it must exist (`mkdir -p fuzz/corpus/encode`). Swap in `chunking`, `api` or `objects`.

## Reproduce a crash

```sh
RUST_BACKTRACE=1 cargo +nightly fuzz run encode fuzz/artifacts/encode/crash-<hash>
cargo +nightly fuzz tmin encode fuzz/artifacts/encode/crash-<hash>
```

`tmin` writes the smallest input that still crashes next to the original.
`-fork=1 -ignore_crashes=1` after `--` keeps fuzzing past a known crash and saves each one.

## Corpus

`seeds/<target>/` is checked in and stays small: sixteen inputs per target, the largest of a
minimised corpus that pass. `corpus/` grows locally and is never committed; nor is
`artifacts/`: a crash becomes a regression test in `crates/omur/tests`, and the artifact is
deleted.
