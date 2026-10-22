//! The core renderer: every element panned onto the 5.1 floor ring with heights folded at
//! −3 dB, gains interpolated sample by sample between updates; the element budget and the
//! fold of objects beyond it (docs/design.md §8.3).
