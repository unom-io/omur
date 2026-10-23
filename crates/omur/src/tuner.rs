//! Rate control: picks the SNR offsets that fill the frame (§8.2.12). Each candidate is a bit
//! tally — `bap` counts per block, priced by the grouping rule — never a `bap` array; only
//! the winner's arrays are written (docs/design.md §5).

use crate::bitalloc::{Alloc, Offsets, for_each_bap, snr_offset};
use crate::mantissas::block_bits;

/// Where the mantissas must fit: every other bit of the frame, counted by a dry run.
pub(crate) struct Budget {
    /// syncinfo and BSI.
    pub(crate) header: u32,
    /// Each block's side information.
    pub(crate) side: [u32; 6],
    /// auxdatae, crcrsv and crc2.
    pub(crate) trailer: u32,
    pub(crate) frame: u32,
    /// The first 5/8 of the frame, which must hold syncinfo, BSI and blocks 0 and 1 (§5.5).
    pub(crate) first: u32,
}

/// `bap` counts per block.
type Tally = [[u32; 16]; 6];

/// Adds channel `a`'s `bap` counts at `offset` to every block that uses each of its sets.
fn add(tally: &mut Tally, a: &Alloc, offset: i32) {
    for s in 0..6 {
        if a.set[s] != s {
            continue;
        }
        let mut count = [0u32; 16];
        for_each_bap(&a.psd[s], &a.mask[s], a.end, offset, |_, b| {
            count[usize::from(b & 15)] += 1;
        });
        for (blk, t) in tally.iter_mut().enumerate() {
            if a.set[blk] == s {
                for (t, c) in t.iter_mut().zip(count) {
                    *t += c;
                }
            }
        }
    }
}

fn fits(tally: &Tally, budget: &Budget) -> bool {
    let bits: [u32; 6] = core::array::from_fn(|b| budget.side[b] + block_bits(&tally[b]));
    let total = budget.header + bits.iter().sum::<u32>() + budget.trailer;
    total <= budget.frame && budget.header + bits[0] + bits[1] <= budget.first
}

/// Offsets all zero mean every `bap` is zero (§7.2.2.1).
fn silent(off: Offsets, lfe: bool) -> bool {
    off.csnr == 0 && off.fsnr == 0 && (!lfe || off.lfefsnr == 0)
}

fn tally(alloc: &[Alloc], lfe: bool, off: Offsets) -> Tally {
    let mut t = [[0u32; 16]; 6];
    if silent(off, lfe) {
        return t;
    }
    let nf = alloc.len() - usize::from(lfe);
    for (c, a) in alloc.iter().enumerate() {
        let fine = if c < nf { off.fsnr } else { off.lfefsnr };
        add(&mut t, a, snr_offset(off.csnr, fine));
    }
    t
}

/// The largest offsets whose mantissas fit: `csnroffst` and the frame-wide `fsnroffst` by
/// one binary search over `16·csnroffst + fsnroffst`, then the LFE's own `lfefsnroffst` as
/// high as it still fits. `None` if not even silence fits.
pub(crate) fn tune(alloc: &[Alloc], lfe: bool, budget: &Budget) -> Option<Offsets> {
    let at = |v: u32| Offsets {
        csnr: (v >> 4) as u8,
        fsnr: (v & 15) as u8,
        lfefsnr: (v & 15) as u8,
    };
    if !fits(&tally(alloc, lfe, at(0)), budget) {
        return None;
    }
    let (mut lo, mut hi) = (0u32, 1024u32);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if fits(&tally(alloc, lfe, at(mid)), budget) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mut off = at(lo);
    if lfe && !silent(off, lfe) {
        let nf = alloc.len() - 1;
        let mut base = [[0u32; 16]; 6];
        for a in &alloc[..nf] {
            add(&mut base, a, snr_offset(off.csnr, off.fsnr));
        }
        for fine in off.fsnr + 1..16 {
            let mut t = base;
            add(&mut t, &alloc[nf], snr_offset(off.csnr, fine));
            if !fits(&t, budget) {
                break;
            }
            off.lfefsnr = fine;
        }
    }
    Some(off)
}

/// Writes every set's `bap` array at the chosen offsets.
pub(crate) fn fill(alloc: &mut [Alloc], lfe: bool, off: Offsets) {
    let nf = alloc.len() - usize::from(lfe);
    let zero = silent(off, lfe);
    for (c, a) in alloc.iter_mut().enumerate() {
        let offset = snr_offset(off.csnr, if c < nf { off.fsnr } else { off.lfefsnr });
        for s in 0..6 {
            if a.set[s] != s {
                continue;
            }
            let bap = &mut a.bap[s];
            for_each_bap(&a.psd[s], &a.mask[s], a.end, offset, |i, b| {
                bap[i] = if zero { 0 } else { b };
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitalloc::curve;

    fn channel(end: usize, level: u8, sets: [usize; 6]) -> Alloc {
        let mut a = Alloc::new(end);
        a.set = sets;
        for s in 0..6 {
            for (i, e) in a.exp[s].iter_mut().enumerate() {
                *e = (level + (i % 9) as u8 / 3 + s as u8).min(24);
            }
            curve(&a.exp[s], end, &mut a.psd[s], &mut a.mask[s]);
        }
        a
    }

    #[test]
    fn the_winner_fills_without_overflow() {
        let alloc = [
            channel(253, 3, [0, 0, 0, 3, 3, 3]),
            channel(199, 6, [0, 1, 1, 1, 4, 4]),
            channel(7, 2, [0, 0, 2, 2, 2, 2]),
        ];
        for frame in [4000u32, 8000, 12000, 20000] {
            let budget = Budget {
                header: 70,
                side: [900, 40, 200, 300, 250, 40],
                trailer: 18,
                frame,
                first: frame * 5 / 8,
            };
            let off = tune(&alloc, true, &budget).expect("silence fits");
            let t = tally(&alloc, true, off);
            assert!(fits(&t, &budget), "frame {frame}: {off:?}");
            let v = 16 * u32::from(off.csnr) + u32::from(off.fsnr);
            if v < 1023 {
                let next = Offsets {
                    csnr: ((v + 1) >> 4) as u8,
                    fsnr: ((v + 1) & 15) as u8,
                    lfefsnr: ((v + 1) & 15) as u8,
                };
                assert!(!fits(&tally(&alloc, true, next), &budget), "frame {frame}");
            }
            assert!(off.lfefsnr >= off.fsnr);
        }
    }

    #[test]
    fn fill_writes_what_the_tally_counted() {
        let mut alloc = [channel(172, 4, [0, 0, 2, 2, 2, 5]), channel(7, 1, [0; 6])];
        let off = Offsets {
            csnr: 20,
            fsnr: 5,
            lfefsnr: 9,
        };
        let want = tally(&alloc, true, off);
        fill(&mut alloc, true, off);
        let mut got = [[0u32; 16]; 6];
        for a in &alloc {
            for (b, g) in got.iter_mut().enumerate() {
                for bin in 0..a.end {
                    g[usize::from(a.bap[a.set[b]][bin])] += 1;
                }
            }
        }
        assert_eq!(got, want);
        let zero = Offsets::default();
        fill(&mut alloc, true, zero);
        assert!(
            alloc
                .iter()
                .all(|a| a.bap.iter().flatten().all(|b| *b == 0))
        );
    }
}
