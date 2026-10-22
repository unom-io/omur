//! Forward MDCT (§8.2.3): the 512-point long transform and the 256-point short pair, as a
//! TDAC fold into a DCT-IV evaluated by a half-length complex FFT with table twiddles. The
//! O(N²) spec sum stays here as the test oracle.
