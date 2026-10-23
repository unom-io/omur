//! The bit allocation core (§7.2.2): PSD, banded PSD, the excitation leak filters, the
//! hearing threshold and the masking curve — computed once per channel and exponent set.
//! The SNR offset enters only at the final address step, which is what lets the tuner search
//! it cheaply.

use crate::tables::{
    BAPTAB, BNDSZ, BNDTAB, DBPBTAB, FASTDEC, FASTGAIN, FLOORTAB, HTH, LATAB, MASKTAB, SLOWDEC,
    SLOWGAIN,
};

/// The parametric model's codes, fixed for the stream: the basic encoder's (§8.2.12).
pub(crate) const SDCYCOD: u8 = 2;
pub(crate) const FDCYCOD: u8 = 1;
pub(crate) const SGAINCOD: u8 = 1;
pub(crate) const DBPBCOD: u8 = 2;
pub(crate) const FLOORCOD: u8 = 4;
pub(crate) const FGAINCOD: u8 = 4;

/// Bins a band-indexed loop may touch: the highest `endmant` (chbwcod 60).
pub(crate) const MAX_END: usize = 253;

/// The frame's SNR offsets: coarse, fine for every full-bandwidth channel, fine for the LFE.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Offsets {
    pub(crate) csnr: u8,
    pub(crate) fsnr: u8,
    pub(crate) lfefsnr: u8,
}

/// One channel's allocation over a frame. Each exponent set lives at the block that sends
/// it; `set[b]` names the block whose set block `b` uses.
pub(crate) struct Alloc {
    /// Exponents as the decoder rebuilds them.
    pub(crate) exp: [[u8; 256]; 6],
    pub(crate) psd: [[i16; 256]; 6],
    /// The masking curve before the SNR offset.
    pub(crate) mask: [[i16; 50]; 6],
    pub(crate) bap: [[u8; 256]; 6],
    pub(crate) set: [usize; 6],
    pub(crate) end: usize,
}

impl Alloc {
    pub(crate) fn new(end: usize) -> Alloc {
        Alloc {
            exp: [[24; 256]; 6],
            psd: [[0; 256]; 6],
            mask: [[0; 50]; 6],
            bap: [[0; 256]; 6],
            set: [0; 6],
            end,
        }
    }
}

/// The SNR offset of §7.2.2.1, `((csnroffst - 15) << 4 + fsnroffst) << 2`.
pub(crate) fn snr_offset(csnroffst: u8, fsnroffst: u8) -> i32 {
    (((i32::from(csnroffst) - 15) << 4) + i32::from(fsnroffst)) << 2
}

fn logadd(a: i32, b: i32) -> i32 {
    let c = a - b;
    let add = LATAB[((c.abs() >> 1) as usize).min(255)];
    if c >= 0 { a + add } else { b + add }
}

fn lowcomp(a: i32, b0: i32, b1: i32, band: usize) -> i32 {
    if band < 20 {
        if b0 + 256 == b1 {
            if band < 7 { 384 } else { 320 }
        } else if b0 > b1 {
            (a - 64).max(0)
        } else {
            a
        }
    } else {
        (a - 128).max(0)
    }
}

/// `psd` per bin and the masking curve per band, before the SNR offset, for the exponents
/// `exp[..end]` of a full-bandwidth or LFE channel (`strtmant` 0).
pub(crate) fn curve(exp: &[u8; 256], end: usize, psd: &mut [i16; 256], mask: &mut [i16; 50]) {
    let end = end.clamp(1, MAX_END);
    for (p, e) in psd.iter_mut().zip(exp).take(end) {
        *p = (3072 - (i32::from(*e) << 7)) as i16;
    }
    let mut bndpsd = [0i32; 50];
    let mut bin = 0;
    let mut band = 0;
    while bin < end {
        let last = (BNDTAB[band] as usize + BNDSZ[band] as usize).min(end);
        let mut acc = i32::from(psd[bin]);
        for p in &psd[bin + 1..last] {
            acc = logadd(acc, i32::from(*p));
        }
        bndpsd[band] = acc;
        bin = last;
        band += 1;
    }

    let bndend = MASKTAB[end - 1] as usize + 1;
    let lfe = bndend == 7;
    let fgain = FASTGAIN[FGAINCOD as usize];
    let sgain = SLOWGAIN[SGAINCOD as usize];
    let fdecay = FASTDEC[FDCYCOD as usize];
    let sdecay = SLOWDEC[SDCYCOD as usize];
    let mut excite = [0i32; 50];
    let mut low = lowcomp(0, bndpsd[0], bndpsd[1], 0);
    excite[0] = bndpsd[0] - fgain - low;
    low = lowcomp(low, bndpsd[1], bndpsd[2], 1);
    excite[1] = bndpsd[1] - fgain - low;
    let (mut fast, mut slow) = (0, 0);
    let mut begin = 7;
    for b in 2..7 {
        let last = lfe && b == 6;
        if !last {
            low = lowcomp(low, bndpsd[b], bndpsd[b + 1], b);
        }
        fast = bndpsd[b] - fgain;
        slow = bndpsd[b] - sgain;
        excite[b] = fast - low;
        if !last && bndpsd[b] <= bndpsd[b + 1] {
            begin = b + 1;
            break;
        }
    }
    for b in begin..bndend.min(22) {
        if !(lfe && b == 6) {
            low = lowcomp(low, bndpsd[b], bndpsd[b + 1], b);
        }
        fast = (fast - fdecay).max(bndpsd[b] - fgain);
        slow = (slow - sdecay).max(bndpsd[b] - sgain);
        excite[b] = (fast - low).max(slow);
    }
    for b in 22..bndend {
        fast = (fast - fdecay).max(bndpsd[b] - fgain);
        slow = (slow - sdecay).max(bndpsd[b] - sgain);
        excite[b] = fast.max(slow);
    }

    let dbknee = DBPBTAB[DBPBCOD as usize];
    for b in 0..bndend {
        if bndpsd[b] < dbknee {
            excite[b] += (dbknee - bndpsd[b]) >> 2;
        }
        mask[b] = excite[b].max(HTH[b]) as i16;
    }
}

/// The final address step (§7.2.2.7): calls `f(bin, bap)` for every bin below `end` at
/// SNR offset `offset`.
#[inline]
pub(crate) fn for_each_bap(
    psd: &[i16; 256],
    mask: &[i16; 50],
    end: usize,
    offset: i32,
    mut f: impl FnMut(usize, u8),
) {
    let end = end.min(MAX_END);
    let floor = FLOORTAB[FLOORCOD as usize];
    let mut bin = 0;
    let mut band = 0;
    while bin < end {
        let last = (BNDTAB[band] as usize + BNDSZ[band] as usize).min(end);
        let m = ((i32::from(mask[band]) - offset - floor).max(0) & 0x1fe0) + floor;
        for (i, p) in psd.iter().enumerate().take(last).skip(bin) {
            let address = ((i32::from(*p) - m) >> 5).clamp(0, 63);
            f(i, BAPTAB[address as usize]);
        }
        bin = last;
        band += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §7.2.2 as printed, step by step, for one fbw or LFE channel: returns (psd, mask, bap).
    fn spec(exp: &[u8], end: usize, snroffset: i32) -> (Vec<i32>, [i32; 50], Vec<u8>) {
        let (start, mut lowcomp) = (0usize, 0i32);
        let sdecay = SLOWDEC[SDCYCOD as usize];
        let fdecay = FASTDEC[FDCYCOD as usize];
        let sgain = SLOWGAIN[SGAINCOD as usize];
        let dbknee = DBPBTAB[DBPBCOD as usize];
        let floor = FLOORTAB[FLOORCOD as usize];
        let fgain = FASTGAIN[FGAINCOD as usize];
        let psd: Vec<i32> = exp.iter().map(|e| 3072 - ((*e as i32) << 7)).collect();
        let logadd = |a: i32, b: i32| {
            let c = a - b;
            let address = ((c.abs() >> 1) as usize).min(255);
            if c >= 0 {
                a + LATAB[address]
            } else {
                b + LATAB[address]
            }
        };
        let calc_lowcomp = |mut a: i32, b0: i32, b1: i32, bin: usize| {
            if bin < 7 {
                if b0 + 256 == b1 {
                    a = 384;
                } else if b0 > b1 {
                    a = 0.max(a - 64);
                }
            } else if bin < 20 {
                if b0 + 256 == b1 {
                    a = 320;
                } else if b0 > b1 {
                    a = 0.max(a - 64);
                }
            } else {
                a = 0.max(a - 128);
            }
            a
        };
        let mut bndpsd = [0i32; 51];
        let (mut j, mut k) = (start, MASKTAB[start] as usize);
        loop {
            let lastbin = (BNDTAB[k] as usize + BNDSZ[k] as usize).min(end);
            bndpsd[k] = psd[j];
            j += 1;
            while j < lastbin {
                bndpsd[k] = logadd(bndpsd[k], psd[j]);
                j += 1;
            }
            k += 1;
            if end <= lastbin {
                break;
            }
        }
        let bndstrt = MASKTAB[start] as usize;
        let bndend = MASKTAB[end - 1] as usize + 1;
        let mut excite = [0i32; 50];
        let (mut fastleak, mut slowleak) = (0, 0);
        lowcomp = calc_lowcomp(lowcomp, bndpsd[0], bndpsd[1], 0);
        excite[0] = bndpsd[0] - fgain - lowcomp;
        lowcomp = calc_lowcomp(lowcomp, bndpsd[1], bndpsd[2], 1);
        excite[1] = bndpsd[1] - fgain - lowcomp;
        let mut begin = 7;
        for bin in 2..7 {
            if bndend != 7 || bin != 6 {
                lowcomp = calc_lowcomp(lowcomp, bndpsd[bin], bndpsd[bin + 1], bin);
            }
            fastleak = bndpsd[bin] - fgain;
            slowleak = bndpsd[bin] - sgain;
            excite[bin] = fastleak - lowcomp;
            if (bndend != 7 || bin != 6) && bndpsd[bin] <= bndpsd[bin + 1] {
                begin = bin + 1;
                break;
            }
        }
        for bin in begin..bndend.min(22) {
            if bndend != 7 || bin != 6 {
                lowcomp = calc_lowcomp(lowcomp, bndpsd[bin], bndpsd[bin + 1], bin);
            }
            fastleak -= fdecay;
            fastleak = fastleak.max(bndpsd[bin] - fgain);
            slowleak -= sdecay;
            slowleak = slowleak.max(bndpsd[bin] - sgain);
            excite[bin] = (fastleak - lowcomp).max(slowleak);
        }
        for bin in 22..bndend {
            fastleak -= fdecay;
            fastleak = fastleak.max(bndpsd[bin] - fgain);
            slowleak -= sdecay;
            slowleak = slowleak.max(bndpsd[bin] - sgain);
            excite[bin] = fastleak.max(slowleak);
        }
        let mut mask = [0i32; 50];
        for bin in bndstrt..bndend {
            if bndpsd[bin] < dbknee {
                excite[bin] += (dbknee - bndpsd[bin]) >> 2;
            }
            mask[bin] = excite[bin].max(HTH[bin]);
        }
        let clean = mask;
        let mut bap = vec![0u8; end];
        let (mut i, mut j) = (start, MASKTAB[start] as usize);
        loop {
            let lastbin = (BNDTAB[j] as usize + BNDSZ[j] as usize).min(end);
            mask[j] -= snroffset;
            mask[j] -= floor;
            if mask[j] < 0 {
                mask[j] = 0;
            }
            mask[j] &= 0x1fe0;
            mask[j] += floor;
            while i < lastbin {
                let address = ((psd[i] - mask[j]) >> 5).clamp(0, 63);
                bap[i] = BAPTAB[address as usize];
                i += 1;
            }
            j += 1;
            if end <= lastbin {
                break;
            }
        }
        (psd, clean, bap)
    }

    fn exponent_sets() -> Vec<(Vec<u8>, usize)> {
        let mut seed = 99u32;
        let mut rnd = move |m: u32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (seed >> 16) % m
        };
        let tonal: Vec<u8> = (0..253)
            .map(|i| if i % 17 == 3 { 1 } else { 9 + (i / 20) as u8 })
            .collect();
        let mut walk = vec![6u8];
        for _ in 1..253 {
            let last = *walk.last().unwrap() as i32 + rnd(5) as i32 - 2;
            walk.push(last.clamp(0, 24) as u8);
        }
        let rising: Vec<u8> = (0..253).map(|i| 24 - (i / 11) as u8).collect();
        vec![
            (tonal.clone(), 253),
            (walk.clone(), 172),
            (walk[..7].to_vec(), 7),
            (vec![3, 1, 3, 5, 3, 4, 2], 7),
            (rising, 253),
            (vec![24; 253], 253),
            (tonal[..73].to_vec(), 73),
        ]
    }

    #[test]
    fn mask_and_bap_match_the_printed_pseudo_code() {
        for (set, end) in exponent_sets() {
            let mut exp = [24u8; 256];
            exp[..end].copy_from_slice(&set[..end]);
            let mut psd = [0i16; 256];
            let mut mask = [0i16; 50];
            curve(&exp, end, &mut psd, &mut mask);
            for (csnr, fsnr) in [(0, 1), (8, 0), (15, 0), (22, 7), (40, 15), (63, 15)] {
                let offset = snr_offset(csnr, fsnr);
                let (want_psd, want_mask, want_bap) = spec(&exp[..end], end, offset);
                let bndend = MASKTAB[end - 1] as usize + 1;
                for b in 0..bndend {
                    assert_eq!(mask[b] as i32, want_mask[b], "end {end} band {b}");
                }
                let mut bap = [0u8; 256];
                for_each_bap(&psd, &mask, end, offset, |i, b| bap[i] = b);
                for i in 0..end {
                    assert_eq!(psd[i] as i32, want_psd[i]);
                    assert_eq!(bap[i], want_bap[i], "end {end} bin {i} offset {offset}");
                }
            }
        }
    }
}
