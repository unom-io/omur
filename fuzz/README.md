# Fuzzing

Three [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) targets (`docs/design.md` §9.3).
The crate stands outside the workspace and builds on nightly.

| Target | What it holds to |
|---|---|
| `encode` | Any AC-3 config, push schedule and f32 (NaN, ±inf, ±1e30, denormals, ±1.0, silence): every frame has `frame_bytes`, both CRCs, our header, and decodes in oxideav-ac3 to finite samples within ±1.001. |
| `chunking` | The same input under two push schedules gives byte-identical frames at the same pts. |
| `api` | Random `push` lengths, `pull` and `Config::new` triples keep the documented errors (`Length`, `Backlog` without consuming, `Layout`, `Bitrate`); a shadow encoder fed only the accepted pushes makes the same frames. |

## Run

From the repository root, with `cargo install cargo-fuzz --version 0.13.2 --locked`:

```sh
cargo +nightly fuzz run encode -- -max_total_time=600
cargo +nightly fuzz run encode fuzz/corpus/encode fuzz/seeds/encode -- -max_total_time=600
```

The second line also reads the checked-in seeds; new inputs land in the first directory, so
it must exist (`mkdir -p fuzz/corpus/encode`). Swap in `chunking` or `api`.

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
