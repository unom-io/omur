//! The bit allocation core (§7.2.2): PSD, banded PSD, the excitation leak filters, the
//! hearing threshold and the masking curve — computed once per channel-block per frame.
//! The SNR offset enters only at the final address step, which is what lets the tuner search
//! it cheaply.
