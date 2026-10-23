//! The bit allocation core (§7.2.2): PSD, banded PSD, the excitation leak filters, the
//! hearing threshold and the masking curve — computed once per channel and exponent set.
//! The SNR offset enters only at the final address step, which is what lets the tuner search
//! it cheaply.

use crate::tables::{
    BAPTAB, BNDSZ, BNDTAB, DBPBTAB, FASTDEC, FASTGAIN, FLOORTAB, HTH, LATAB, MASKTAB, SLOWDEC,
    SLOWGAIN,
};

/// The parametric model's codes, fixed for the stream: the basic encoder's (§8.2.12) with
/// the knee at `dbpbcod` 3 and no masking floor (`floorcod` 7), the values ffmpeg's streams
/// carry at every rate. A higher knee lifts the mask of all but the loudest bands, which
/// moves bits towards the peaks.
pub(crate) const SDCYCOD: u8 = 2;
pub(crate) const FDCYCOD: u8 = 1;
pub(crate) const SGAINCOD: u8 = 1;
pub(crate) const DBPBCOD: u8 = 3;
pub(crate) const FLOORCOD: u8 = 7;
pub(crate) const FGAINCOD: u8 = 4;

/// The coupling channel's leak initialisation, `cplfleak` and `cplsleak`: ffmpeg's 0.
pub(crate) const CPLFLEAK: u8 = 0;
pub(crate) const CPLSLEAK: u8 = 0;

/// Bins a band-indexed loop may touch: the highest `endmant` (chbwcod 60).
pub(crate) const MAX_END: usize = 253;

/// Slots an allocation covers: five full-bandwidth channels, the LFE, the coupling channel.
pub(crate) const MAX_SLOTS: usize = 7;

/// The frame's SNR offsets: coarse, and fine per slot (`fsnroffst`, `lfefsnroffst`,
/// `cplfsnroffst`, in the encoder's slot order); on E-AC-3, the fast gain codes when the
/// frame sends any (`frmfgaincode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Offsets {
    pub(crate) csnr: u8,
    pub(crate) fsnr: [u8; MAX_SLOTS],
    pub(crate) gains: Option<Gains>,
}

/// E-AC-3's fast gain codes of each block per slot, and the blocks that send them
/// (`fgaincode`). A block that sends none decodes at `FGAINCOD`, as the syntax has it;
/// ffmpeg 9.0.1 keeps the last codes it read instead, so a block after one with other
/// codes sends its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Gains {
    pub(crate) codes: [[u8; MAX_SLOTS]; 6],
    pub(crate) sent: [bool; 6],
}

impl Gains {
    /// The codes `gain[slot][block]` of a frame of `blocks` blocks, sent where they or
    /// the block's before differ from `FGAINCOD`; `after` when the last block before the
    /// frame did. `None` when no block sends.
    pub(crate) fn of(gain: &[[u8; 6]], blocks: usize, after: bool) -> Option<Gains> {
        let mut g = Gains {
            codes: [[FGAINCOD; MAX_SLOTS]; 6],
            sent: [false; 6],
        };
        let mut before = after;
        for b in 0..blocks.min(6) {
            for (code, slot) in g.codes[b].iter_mut().zip(gain) {
                *code = slot[b];
            }
            let own = g.codes[b].iter().any(|c| *c != FGAINCOD);
            g.sent[b] = own || before;
            before = own;
        }
        g.sent.iter().any(|s| *s).then_some(g)
    }

    /// Whether the frame's last block leaves a decoder at codes other than `FGAINCOD`.
    pub(crate) fn last_differs(gains: Option<&Gains>, blocks: usize) -> bool {
        gains.is_some_and(|g| {
            g.codes[blocks.clamp(1, 6) - 1]
                .iter()
                .any(|c| *c != FGAINCOD)
        })
    }
}

/// One channel's allocation over a frame of `blocks` blocks, bins `start..end`. Each
/// exponent set lives at the block that sends it; `set[b]` names the block whose set block
/// `b` uses. A set whose blocks take different fast gain codes splits where they change,
/// each part a set of its own with the exponents copied.
pub(crate) struct Alloc {
    /// Exponents as the decoder rebuilds them.
    pub(crate) exp: [[u8; 256]; 6],
    pub(crate) psd: [[i16; 256]; 6],
    pub(crate) bndpsd: [[i32; 50]; 6],
    /// The masking curve before the SNR offset, at the set's fast gain code.
    pub(crate) mask: [[i16; 50]; 6],
    pub(crate) bap: [[u8; 256]; 6],
    pub(crate) set: [usize; 6],
    /// Each block's fast gain code: `FGAINCOD` but where E-AC-3's climb moves it.
    pub(crate) gain: [u8; 6],
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) blocks: usize,
}

impl Alloc {
    pub(crate) fn new(start: usize, end: usize, blocks: usize) -> Alloc {
        Alloc {
            exp: [[24; 256]; 6],
            psd: [[0; 256]; 6],
            bndpsd: [[0; 50]; 6],
            mask: [[0; 50]; 6],
            bap: [[0; 256]; 6],
            set: [0; 6],
            gain: [FGAINCOD; 6],
            start,
            end,
            blocks,
        }
    }

    /// Set `s`'s psd, banded psd and masking curve at `FGAINCOD`, from its exponents.
    pub(crate) fn analyse(&mut self, s: usize) {
        if let (Some(exp), Some(psd), Some(bnd), Some(mask)) = (
            self.exp.get(s),
            self.psd.get_mut(s),
            self.bndpsd.get_mut(s),
            self.mask.get_mut(s),
        ) {
            curve(exp, self.start, self.end, FGAINCOD, psd, bnd, mask);
        }
        if let Some(g) = self.gain.get_mut(s) {
            *g = FGAINCOD;
        }
    }

    /// Gives block `b` its fast gain code `gain[b]`, splitting the sets where the code
    /// changes inside one; a part copies its set's exponents and takes its own mask.
    pub(crate) fn regroup(&mut self, gain: &[u8; 6]) {
        let n = self.blocks.min(6);
        let sets = self.set;
        for b in 0..n {
            let s = sets[b];
            if b == s || gain[b] != gain[b - 1] {
                if b != s {
                    self.exp[b] = self.exp[s];
                    self.psd[b] = self.psd[s];
                    self.bndpsd[b] = self.bndpsd[s];
                }
                if b != s || gain[b] != self.gain[b] {
                    let (start, end) = (self.start, self.end);
                    masking(&self.bndpsd[b], start, end, gain[b], &mut self.mask[b]);
                }
                self.set[b] = b;
            } else {
                self.set[b] = self.set[b - 1];
            }
        }
        self.gain = *gain;
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

/// `psd` per bin, the banded `bndpsd` and the masking curve per band, before the SNR
/// offset, for the exponents `exp[start..end]` at fast gain code `fgaincod`: a
/// full-bandwidth or LFE channel from bin 0 with low-band compensation, the coupling channel
/// from its start with the leak initialisation.
pub(crate) fn curve(
    exp: &[u8; 256],
    start: usize,
    end: usize,
    fgaincod: u8,
    psd: &mut [i16; 256],
    bndpsd: &mut [i32; 50],
    mask: &mut [i16; 50],
) {
    let end = end.min(MAX_END);
    if start >= end {
        return;
    }
    for (p, e) in psd[start..end].iter_mut().zip(&exp[start..end]) {
        *p = (3072 - (i32::from(*e) << 7)) as i16;
    }
    let mut bin = start;
    let mut band = MASKTAB[start] as usize;
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
    masking(bndpsd, start, end, fgaincod, mask);
}

/// The masking curve over the bands of `start..end` from their `bndpsd` at fast gain code
/// `fgaincod`: the excitation's fast and slow leaks, the knee and the hearing threshold.
pub(crate) fn masking(
    bndpsd: &[i32; 50],
    start: usize,
    end: usize,
    fgaincod: u8,
    mask: &mut [i16; 50],
) {
    let end = end.min(MAX_END);
    if start >= end {
        return;
    }
    let bndstrt = MASKTAB[start] as usize;
    let bndend = MASKTAB[end - 1] as usize + 1;
    let fgain = FASTGAIN[usize::from(fgaincod & 7)];
    let sgain = SLOWGAIN[SGAINCOD as usize];
    let fdecay = FASTDEC[FDCYCOD as usize];
    let sdecay = SLOWDEC[SDCYCOD as usize];
    let mut excite = [0i32; 50];
    let (mut fast, mut slow, tail) = if start == 0 {
        let lfe = bndend == 7;
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
        (fast, slow, 22)
    } else {
        let leak = |code: u8| (i32::from(code) << 8) + 768;
        (leak(CPLFLEAK), leak(CPLSLEAK), bndstrt)
    };
    for b in tail..bndend {
        fast = (fast - fdecay).max(bndpsd[b] - fgain);
        slow = (slow - sdecay).max(bndpsd[b] - sgain);
        excite[b] = fast.max(slow);
    }

    let dbknee = DBPBTAB[DBPBCOD as usize];
    for b in bndstrt..bndend {
        if bndpsd[b] < dbknee {
            excite[b] += (dbknee - bndpsd[b]) >> 2;
        }
        mask[b] = excite[b].max(HTH[b]) as i16;
    }
}

/// The final address step (§7.2.2.7): calls `f(bin, address)` for every bin of
/// `start..end` at SNR offset `offset`; `baptab[address]` is the bin's `bap`.
#[inline]
pub(crate) fn for_each_address(
    psd: &[i16; 256],
    mask: &[i16; 50],
    start: usize,
    end: usize,
    offset: i32,
    mut f: impl FnMut(usize, usize),
) {
    let end = end.min(MAX_END);
    let floor = FLOORTAB[FLOORCOD as usize];
    let mut bin = start;
    let mut band = MASKTAB[start.min(255)] as usize;
    while bin < end {
        let last = (BNDTAB[band] as usize + BNDSZ[band] as usize).min(end);
        let m = ((i32::from(mask[band]) - offset - floor).max(0) & 0x1fe0) + floor;
        for (i, p) in psd.get(bin..last).unwrap_or_default().iter().enumerate() {
            f(bin + i, ((i32::from(*p) - m) >> 5).clamp(0, 63) as usize);
        }
        bin = last;
        band += 1;
    }
}

/// `for_each_address` through `baptab`: calls `f(bin, bap)`.
#[inline]
pub(crate) fn for_each_bap(
    psd: &[i16; 256],
    mask: &[i16; 50],
    start: usize,
    end: usize,
    offset: i32,
    mut f: impl FnMut(usize, u8),
) {
    for_each_address(psd, mask, start, end, offset, |i, a| f(i, BAPTAB[a & 63]));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §7.2.2 as printed, step by step, for an fbw or LFE channel (`start` 0) or the
    /// coupling channel: returns (psd, mask, bap) over `exp[start..end]`.
    fn spec(
        exp: &[u8],
        start: usize,
        end: usize,
        snroffset: i32,
        fgaincod: u8,
    ) -> (Vec<i32>, [i32; 50], Vec<u8>) {
        let mut lowcomp = 0i32;
        let sdecay = SLOWDEC[SDCYCOD as usize];
        let fdecay = FASTDEC[FDCYCOD as usize];
        let sgain = SLOWGAIN[SGAINCOD as usize];
        let dbknee = DBPBTAB[DBPBCOD as usize];
        let floor = FLOORTAB[FLOORCOD as usize];
        let fgain = FASTGAIN[fgaincod as usize];
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
        let mut begin;
        if bndstrt == 0 {
            lowcomp = calc_lowcomp(lowcomp, bndpsd[0], bndpsd[1], 0);
            excite[0] = bndpsd[0] - fgain - lowcomp;
            lowcomp = calc_lowcomp(lowcomp, bndpsd[1], bndpsd[2], 1);
            excite[1] = bndpsd[1] - fgain - lowcomp;
            begin = 7;
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
            begin = 22;
        } else {
            begin = bndstrt;
            fastleak = ((CPLFLEAK as i32) << 8) + 768;
            slowleak = ((CPLSLEAK as i32) << 8) + 768;
        }
        for bin in begin..bndend {
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

    /// (exponents, start, end): fbw and LFE sets from bin 0, coupling sets from a sub-band.
    fn exponent_sets() -> Vec<(Vec<u8>, usize, usize)> {
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
            (tonal.clone(), 0, 253),
            (walk.clone(), 0, 172),
            (walk[..7].to_vec(), 0, 7),
            (vec![3, 1, 3, 5, 3, 4, 2], 0, 7),
            (rising.clone(), 0, 253),
            (vec![24; 253], 0, 253),
            (tonal[..73].to_vec(), 0, 73),
            (walk.clone(), 133, 169),
            (tonal.clone(), 145, 217),
            (rising, 181, 253),
            (walk, 37, 253),
        ]
    }

    /// A block sends codes where they or its predecessor's differ from the default; the
    /// frame sends none when no block does.
    #[test]
    fn gains_reset_after_a_raised_block() {
        let mut gain = [[FGAINCOD; 6]; 3];
        assert_eq!(Gains::of(&gain, 6, false), None);
        let g = Gains::of(&gain, 6, true).unwrap();
        assert_eq!(g.sent, [true, false, false, false, false, false]);
        gain[1][1] = 5;
        gain[2][4] = 5;
        let g = Gains::of(&gain, 6, false).unwrap();
        assert_eq!(g.sent, [false, true, true, false, true, true]);
        assert_eq!(
            (g.codes[1][1], g.codes[4][2], g.codes[4][1]),
            (5, 5, FGAINCOD)
        );
        assert!(!Gains::last_differs(Some(&g), 6));
        assert!(Gains::last_differs(Some(&g), 5));
    }

    /// A set whose blocks take two codes splits into two sets with the same exponents,
    /// each masked at its own code.
    #[test]
    fn regroup_splits_sets_where_the_code_changes() {
        let (start, end) = (0, 253);
        let mut a = Alloc::new(start, end, 6);
        a.set = [0, 0, 0, 3, 3, 3];
        for s in [0, 3] {
            for (i, e) in a.exp[s].iter_mut().enumerate() {
                *e = (3 + (i % 7) as u8 + s as u8).min(24);
            }
            a.analyse(s);
        }
        a.regroup(&[5, 5, 4, 4, 4, 5]);
        assert_eq!(a.set, [0, 0, 2, 3, 3, 5]);
        for (s, code, from) in [(0, 5, 0), (2, 4, 0), (3, 4, 3), (5, 5, 3)] {
            assert_eq!(a.exp[s], a.exp[from], "set {s}");
            let (mut psd, mut mask) = ([0i16; 256], [0i16; 50]);
            curve(
                &a.exp[s],
                start,
                end,
                code,
                &mut psd,
                &mut [0; 50],
                &mut mask,
            );
            assert_eq!((a.psd[s], a.mask[s]), (psd, mask), "set {s}");
        }
    }

    /// Each set at one of the fast gain codes 4, 7, 0 and 5 in turn.
    #[test]
    fn mask_and_bap_match_the_printed_pseudo_code() {
        for ((set, start, end), fgaincod) in
            exponent_sets().into_iter().zip([4, 7, 0, 5].iter().cycle())
        {
            let mut exp = [24u8; 256];
            exp[..end].copy_from_slice(&set[..end]);
            let mut psd = [0i16; 256];
            let mut mask = [0i16; 50];
            curve(
                &exp,
                start,
                end,
                *fgaincod,
                &mut psd,
                &mut [0; 50],
                &mut mask,
            );
            for (csnr, fsnr) in [(0, 1), (8, 0), (15, 0), (22, 7), (40, 15), (63, 15)] {
                let offset = snr_offset(csnr, fsnr);
                let (want_psd, want_mask, want_bap) =
                    spec(&exp[..end], start, end, offset, *fgaincod);
                let bndend = MASKTAB[end - 1] as usize + 1;
                for b in MASKTAB[start] as usize..bndend {
                    assert_eq!(mask[b] as i32, want_mask[b], "{start}..{end} band {b}");
                }
                let mut bap = [0u8; 256];
                for_each_bap(&psd, &mask, start, end, offset, |i, b| bap[i] = b);
                for i in start..end {
                    assert_eq!(psd[i] as i32, want_psd[i]);
                    assert_eq!(
                        bap[i], want_bap[i],
                        "{start}..{end} bin {i} offset {offset}"
                    );
                }
            }
        }
    }
}
