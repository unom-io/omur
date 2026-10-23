//! Mantissa quantization (§7.3, §8.2.11, §8.2.13): symmetric and asymmetric quantizers per
//! `bap`, the three grouped low-precision classes, and the block's mantissa bit cost from
//! its `bap` counts alone.

use crate::tables::{MANT_BITS, SYM_LEVELS};

/// Channels a block packs: five full-bandwidth, the LFE and the coupling channel.
pub(crate) const MAX_CH: usize = 7;

/// `2^e` for every exponent, exact.
const POW2: [f32; 25] = {
    let mut t = [0f32; 25];
    let mut e = 0;
    while e < 25 {
        t[e] = f32::from_bits((127 + e as u32) << 23);
        e += 1;
    }
    t
};

/// Mantissa bits of one block from how many of its mantissas, over all its channels, sit at
/// each `bap`. Grouping runs across channels and pads at the block's end (§7.3.5): three
/// bap-1 values in 5 bits, three bap-2 in 7, two bap-4 in 7.
pub(crate) fn block_bits(count: &[u32; 16]) -> u32 {
    let grouped = 5 * count[1].div_ceil(3) + 7 * count[2].div_ceil(3) + 7 * count[4].div_ceil(2);
    grouped
        + count
            .iter()
            .zip(MANT_BITS)
            .map(|(n, bits)| n * u32::from(bits))
            .sum::<u32>()
}

/// The code of `m` on the `levels`-step symmetric quantizer (Tables 7.19 to 7.23).
fn symmetric(m: f32, levels: u8) -> u16 {
    let t = (m.clamp(-1.0, 1.0) + 1.0) * f32::from(levels) * 0.5;
    (t as u16).min(u16::from(levels) - 1)
}

/// `m` rounded to `bits` bits of two's complement fraction, as the field's bit pattern.
fn asymmetric(m: f32, bits: u8) -> u16 {
    let half = 1i32 << (bits - 1);
    let s = half as f32;
    let q = ((m.clamp(-1.0, 1.0) * s + s + 0.5) as i32 - half).min(half - 1);
    (q as u16) & (((1u32 << bits) - 1) as u16)
}

/// One block's quantized mantissas in packing order: `code[ch][bin]` written in `bits[ch][bin]`
/// bits. A grouped value sits at its group's first member; the others carry no bits.
pub(crate) struct Mantissas {
    pub(crate) code: [[u16; 256]; MAX_CH],
    pub(crate) bits: [[u8; 256]; MAX_CH],
    /// The open group of bap 1, 2 and 4: its first member's channel and bin, members in.
    open: [(usize, usize, u8); 3],
}

impl Mantissas {
    pub(crate) fn new() -> Mantissas {
        Mantissas {
            code: [[0; 256]; MAX_CH],
            bits: [[0; 256]; MAX_CH],
            open: [(0, 0, 0); 3],
        }
    }

    /// Closes the open groups; their missing members read as code 0.
    pub(crate) fn begin_block(&mut self) {
        self.open = [(0, 0, 0); 3];
    }

    /// Quantizes `coef[start..end]` of slot `ch` against the exponents the decoder rebuilds
    /// and the `bap` of each bin. Slots are quantized in packing order: groups run across
    /// them.
    pub(crate) fn channel(
        &mut self,
        ch: usize,
        coef: &[f32; 256],
        exp: &[u8; 256],
        bap: &[u8; 256],
        start: usize,
        end: usize,
    ) {
        let ch = ch.min(MAX_CH - 1);
        for bin in start..end.min(256) {
            let b = bap[bin] & 15;
            let m = coef[bin] * POW2[usize::from(exp[bin]).min(24)];
            let (code, bits) = match b {
                0 => (0, 0),
                1 | 2 | 4 => {
                    let (class, weights, size) = match b {
                        1 => (0, [9, 3, 1], 3),
                        2 => (1, [25, 5, 1], 3),
                        _ => (2, [11, 1, 0], 2),
                    };
                    let q = symmetric(m, SYM_LEVELS[usize::from(b)]);
                    let (gch, gbin, n) = self.open[class];
                    if n == 0 {
                        self.open[class] = (ch, bin, 1 % size);
                        (q * weights[0], if b == 1 { 5 } else { 7 })
                    } else {
                        self.code[gch][gbin] += q * weights[usize::from(n)];
                        self.open[class].2 = (n + 1) % size;
                        (0, 0)
                    }
                }
                3 | 5 => (
                    symmetric(m, SYM_LEVELS[usize::from(b)]),
                    MANT_BITS[usize::from(b)],
                ),
                _ => (
                    asymmetric(m, MANT_BITS[usize::from(b)]),
                    MANT_BITS[usize::from(b)],
                ),
            };
            self.code[ch][bin] = code;
            self.bits[ch][bin] = bits;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    fn rnd(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *seed >> 8
    }

    /// The decoder's values per code (Tables 7.19 to 7.23, §7.3.2).
    fn dequantize(bap: u8, code: u16) -> f32 {
        if bap <= 5 {
            let l = SYM_LEVELS[bap as usize] as f32;
            (2.0 * code as f32 - l + 1.0) / l
        } else {
            let bits = MANT_BITS[bap as usize];
            let v = ((code as i32) << (32 - bits)) >> (32 - bits);
            v as f32 / (1i32 << (bits - 1)) as f32
        }
    }

    #[test]
    fn tally_matches_the_bits_a_writer_emits() {
        let mut seed = 5;
        for trial in 0..40 {
            let mut q = Mantissas::new();
            q.begin_block();
            let mut count = [0u32; 16];
            let channels = 1 + trial % MAX_CH;
            for ch in 0..channels {
                let end = [7, 73, 253][(rnd(&mut seed) % 3) as usize];
                let bap: [u8; 256] = core::array::from_fn(|_| {
                    // weight the grouped classes so partial groups are common
                    let r = rnd(&mut seed) % 25;
                    if r < 12 {
                        [0, 1, 2, 4][r as usize % 4]
                    } else {
                        (r - 9) as u8
                    }
                });
                let coef: [f32; 256] =
                    core::array::from_fn(|_| (rnd(&mut seed) % 2000) as f32 / 1000.0 - 1.0);
                for b in &bap[..end] {
                    count[*b as usize] += 1;
                }
                q.channel(ch, &coef, &[0; 256], &bap, 0, end);
            }
            let mut buf = [0u8; 4096];
            let mut w = BitWriter::new(&mut buf);
            for ch in 0..channels {
                for bin in 0..256 {
                    let bits = q.bits[ch][bin];
                    if bits > 0 {
                        w.put(u32::from(q.code[ch][bin]), u32::from(bits)).unwrap();
                    }
                }
            }
            assert_eq!(w.position() as u32, block_bits(&count), "trial {trial}");
        }
    }

    #[test]
    fn groups_unpack_to_their_members_across_channels() {
        let mut q = Mantissas::new();
        q.begin_block();
        let coef: [f32; 256] = core::array::from_fn(|i| ((i * 7) % 11) as f32 / 5.5 - 1.0);
        let mut bap = [0u8; 256];
        bap[..5].copy_from_slice(&[1, 4, 1, 2, 4]);
        q.channel(0, &coef, &[0; 256], &bap, 0, 5);
        q.channel(1, &coef, &[0; 256], &bap, 0, 5);
        // bap 1: (0,0) (0,2) (1,0); then (1,2) pads; bap 4: (0,1) (0,4); (1,1) (1,4).
        let one = |b: usize| symmetric(coef[b], 3);
        let eleven = |b: usize| symmetric(coef[b], 11);
        assert_eq!(q.code[0][0], 9 * one(0) + 3 * one(2) + one(0));
        assert_eq!(q.code[1][2], 9 * one(2));
        assert_eq!(q.code[0][1], 11 * eleven(1) + eleven(4));
        assert_eq!(q.code[1][1], 11 * eleven(1) + eleven(4));
        assert_eq!((q.bits[0][2], q.bits[1][0], q.bits[1][4]), (0, 0, 0));
        assert_eq!(q.bits[0][3], 7);
    }

    #[test]
    fn quantizers_round_to_the_nearest_level() {
        for bap in 1..16u8 {
            let step = if bap <= 5 {
                2.0 / SYM_LEVELS[bap as usize] as f32
            } else {
                1.0 / (1u32 << (MANT_BITS[bap as usize] - 1)) as f32
            };
            for i in 0..=400 {
                let m = i as f32 / 200.0 - 1.0;
                let code = if bap <= 5 {
                    symmetric(m, SYM_LEVELS[bap as usize])
                } else {
                    asymmetric(m, MANT_BITS[bap as usize])
                };
                let err = (dequantize(bap, code) - m).abs();
                // the top asymmetric level is one step short of +1
                let limit = if m > 1.0 - step {
                    step
                } else {
                    step / 2.0 + 1e-6
                };
                assert!(err <= limit, "bap {bap} m {m} err {err}");
            }
        }
    }
}
