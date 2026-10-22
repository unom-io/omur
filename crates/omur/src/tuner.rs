//! Rate control: picks `csnroffst` and the per-channel `fsnroffst` values that fill the
//! frame. A bit tally per candidate, never a bap array; the per-channel step is what spreads
//! bits across channels (docs/design.md §5).
