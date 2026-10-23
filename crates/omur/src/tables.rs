//! The normative tables of A/52, transcribed from ETSI TS 102 366 (= A/52:2018): frame
//! sizes (Table 5.18), the bit allocation tables (Tables 7.6 to 7.16) and the quantizer
//! sizes (Table 7.17). The window and the MDCT twiddles are computed, not transcribed:
//! `tables_gen.rs`. Nothing here is computed at run time (docs/design.md D4).
//!
//! Sections and tables are numbered as in A/52:2018 throughout the crate; ETSI's numbers
//! are one chapter lower (its §6.2.2 is A/52's §7.2.2).

/// Nominal bitrates in kbit/s by `frmsizecod >> 1` (Table 5.18). At 48 kHz a syncframe is
/// `2 × kbps` 16-bit words and both codes of a pair mean the same size.
pub(crate) const BITRATES: [u32; 19] = [
    32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640,
];

/// `slowdec[sdcycod]` (Table 7.6).
pub(crate) const SLOWDEC: [i32; 4] = [0x0f, 0x11, 0x13, 0x15];
/// `fastdec[fdcycod]` (Table 7.7).
pub(crate) const FASTDEC: [i32; 4] = [0x3f, 0x53, 0x67, 0x7b];
/// `slowgain[sgaincod]` (Table 7.8).
pub(crate) const SLOWGAIN: [i32; 4] = [0x540, 0x4d8, 0x478, 0x410];
/// `dbpbtab[dbpbcod]` (Table 7.9).
pub(crate) const DBPBTAB: [i32; 4] = [0x000, 0x700, 0x900, 0xb00];
/// `floortab[floorcod]` (Table 7.10); `0xf800` is -2048 as a 16-bit two's complement.
pub(crate) const FLOORTAB: [i32; 8] = [0x2f0, 0x2b0, 0x270, 0x230, 0x1f0, 0x170, 0x0f0, -0x800];
/// `fastgain[fgaincod]` (Table 7.11).
pub(crate) const FASTGAIN: [i32; 8] = [0x080, 0x100, 0x180, 0x200, 0x280, 0x300, 0x380, 0x400];

/// First bin of each of the 50 bit allocation bands, `bndtab[]` (Table 7.12).
pub(crate) const BNDTAB: [u8; 50] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 31, 34, 37, 40, 43, 46, 49, 55, 61, 67, 73, 79, 85, 97, 109, 121, 133, 157, 181,
    205, 229,
];

/// Width of each band in bins, `bndsz[]` (Table 7.12).
pub(crate) const BNDSZ: [u8; 50] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3,
    3, 3, 3, 6, 6, 6, 6, 6, 6, 12, 12, 12, 12, 24, 24, 24, 24, 24,
];

/// Band of each bin, `masktab[]` (Table 7.13). Bins 253 to 255 carry no band.
pub(crate) const MASKTAB: [u8; 256] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35,
    35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 39,
    39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
    45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
    46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
    47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
    48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49,
    49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 0, 0, 0,
];

/// Log-addition increments by half the difference, `latab[]` (Table 7.14).
pub(crate) const LATAB: [i32; 256] = [
    0x0040, 0x003f, 0x003e, 0x003d, 0x003c, 0x003b, 0x003a, 0x0039, 0x0038, 0x0037, 0x0036, 0x0035,
    0x0034, 0x0034, 0x0033, 0x0032, 0x0031, 0x0030, 0x002f, 0x002f, 0x002e, 0x002d, 0x002c, 0x002c,
    0x002b, 0x002a, 0x0029, 0x0029, 0x0028, 0x0027, 0x0026, 0x0026, 0x0025, 0x0024, 0x0024, 0x0023,
    0x0023, 0x0022, 0x0021, 0x0021, 0x0020, 0x0020, 0x001f, 0x001e, 0x001e, 0x001d, 0x001d, 0x001c,
    0x001c, 0x001b, 0x001b, 0x001a, 0x001a, 0x0019, 0x0019, 0x0018, 0x0018, 0x0017, 0x0017, 0x0016,
    0x0016, 0x0015, 0x0015, 0x0015, 0x0014, 0x0014, 0x0013, 0x0013, 0x0013, 0x0012, 0x0012, 0x0012,
    0x0011, 0x0011, 0x0011, 0x0010, 0x0010, 0x0010, 0x000f, 0x000f, 0x000f, 0x000e, 0x000e, 0x000e,
    0x000d, 0x000d, 0x000d, 0x000d, 0x000c, 0x000c, 0x000c, 0x000c, 0x000b, 0x000b, 0x000b, 0x000b,
    0x000a, 0x000a, 0x000a, 0x000a, 0x000a, 0x0009, 0x0009, 0x0009, 0x0009, 0x0009, 0x0008, 0x0008,
    0x0008, 0x0008, 0x0008, 0x0008, 0x0007, 0x0007, 0x0007, 0x0007, 0x0007, 0x0007, 0x0006, 0x0006,
    0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005, 0x0005,
    0x0005, 0x0005, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004, 0x0004,
    0x0004, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003,
    0x0003, 0x0003, 0x0003, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002,
    0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0001, 0x0001,
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001,
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001,
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000,
];

/// Hearing threshold at 48 kHz, `hth[0][band]` (Table 7.15).
pub(crate) const HTH: [i32; 50] = [
    0x04d0, 0x04d0, 0x0440, 0x0400, 0x03e0, 0x03c0, 0x03b0, 0x03b0, 0x03a0, 0x03a0, 0x03a0, 0x03a0,
    0x03a0, 0x0390, 0x0390, 0x0390, 0x0380, 0x0380, 0x0370, 0x0370, 0x0360, 0x0360, 0x0350, 0x0350,
    0x0340, 0x0340, 0x0330, 0x0320, 0x0310, 0x0300, 0x02f0, 0x02f0, 0x02f0, 0x02f0, 0x0300, 0x0310,
    0x0340, 0x0390, 0x03e0, 0x0420, 0x0460, 0x0490, 0x04a0, 0x0460, 0x0440, 0x0440, 0x0520, 0x0800,
    0x0840, 0x0840,
];

/// Bit allocation pointer by address, `baptab[]` (Table 7.16).
pub(crate) const BAPTAB: [u8; 64] = [
    0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9,
    10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14,
    15, 15, 15, 15, 15, 15, 15, 15, 15,
];

/// Bits one mantissa takes at each `bap` that is not grouped (Table 7.17). Baps 1, 2 and 4
/// are grouped: three in 5 bits, three in 7, two in 7.
pub(crate) const MANT_BITS: [u8; 16] = [0, 0, 0, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16];

/// Quantizer levels at the symmetric baps 1 to 5 (Table 7.17).
pub(crate) const SYM_LEVELS: [u8; 6] = [0, 3, 5, 7, 11, 15];

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_ac3::tables as ox;

    /// A second transcription, by other hands, of the same tables.
    #[test]
    fn transcription_matches_oxideav() {
        assert_eq!(SLOWDEC, ox::SLOWDEC);
        assert_eq!(FASTDEC, ox::FASTDEC);
        assert_eq!(SLOWGAIN, ox::SLOWGAIN);
        assert_eq!(DBPBTAB, ox::DBPBTAB);
        assert_eq!(FLOORTAB, ox::FLOORTAB);
        assert_eq!(FASTGAIN, ox::FASTGAIN);
        assert!(
            BNDTAB
                .iter()
                .zip(ox::BNDTAB)
                .all(|(a, b)| u32::from(*a) == b)
        );
        assert!(BNDSZ.iter().zip(ox::BNDSZ).all(|(a, b)| u32::from(*a) == b));
        assert_eq!(MASKTAB[..253], ox::MASKTAB[..253]);
        assert!(LATAB.iter().zip(ox::LATAB).all(|(a, b)| *a == i32::from(b)));
        assert!(HTH.iter().zip(ox::HTH[0]).all(|(a, b)| *a == i32::from(b)));
        assert_eq!(BAPTAB, ox::BAPTAB);
        for bap in [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
            assert_eq!(MANT_BITS[bap], ox::QUANTIZATION_BITS[bap], "bap {bap}");
        }
    }

    #[test]
    fn bands_tile_the_bins() {
        for band in 0..49 {
            assert_eq!(BNDTAB[band] + BNDSZ[band], BNDTAB[band + 1]);
        }
        for (bin, band) in MASKTAB.iter().enumerate().take(253) {
            let band = *band as usize;
            assert!((BNDTAB[band] as usize..(BNDTAB[band] + BNDSZ[band]) as usize).contains(&bin));
        }
    }
}
