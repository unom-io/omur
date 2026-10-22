# Contributing

## Clean room

omur is written from the public specifications: ATSC A/52:2018 (= ETSI TS 102 366) and ETSI
TS 103 420. Never open the source of ffmpeg, Aften, liba52 or any other LGPL/GPL codec while
working here; ffmpeg as a black-box binary in tests is fine. The two MIT references
(`NOTICE`) may be read; a piece adapted from one of them carries a header line naming it.

## Commits

`type(scope): summary` — imperative, 72 characters at most, no trailing period. The body says
why, in 200 words at most. Scope is a module (`mdct`, `tuner`, `eac3`, `probe`).

## Comments

Present tense, the rule as it stands; no history. `// SAFETY:` never appears: `unsafe_code`
is forbidden.

## Checks

```
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The quality gate is `tools/probe` plus ffmpeg on `PATH` (`docs/design.md` §9).
