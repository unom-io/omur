//! Rate control: picks the SNR offsets that fill the frame (§8.2.12). Each candidate is a bit
//! tally — `bap` counts per block, priced by the grouping rule — never a `bap` array; only
//! the winner's arrays are written (docs/design.md §5).

use crate::bitalloc::{Alloc, MAX_SLOTS, Offsets, for_each_bap, snr_offset};
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
    /// The first 5/8 of the frame, which must hold syncinfo, BSI and blocks 0 and 1, and
    /// which block 5's mantissas must not start inside (§5.5).
    pub(crate) first: u32,
}

/// What skip fields can overshoot by: a byte's rounding plus one more field's 9 bits.
const SKIP_SLACK: u32 = 16;

/// `bap` counts per block.
type Tally = [[u32; 16]; 6];

/// Adds channel `a`'s `bap` counts at `offset` to every block that uses each of its sets.
fn add(tally: &mut Tally, a: &Alloc, offset: i32) {
    for s in 0..6 {
        if a.set[s] != s {
            continue;
        }
        let mut count = [0u32; 16];
        for_each_bap(&a.psd[s], &a.mask[s], a.start, a.end, offset, |_, b| {
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

/// The frame holds it, blocks 0 and 1 end inside the first 5/8, and block 5's mantissas
/// with the trailer fit the last 3/8 once skip fields move them there.
fn fits(tally: &Tally, budget: &Budget) -> bool {
    let bits: [u32; 6] = core::array::from_fn(|b| budget.side[b] + block_bits(&tally[b]));
    let total = budget.header + bits.iter().sum::<u32>() + budget.trailer;
    total <= budget.frame
        && budget.header + bits[0] + bits[1] <= budget.first
        && block_bits(&tally[5]) + budget.trailer + SKIP_SLACK <= budget.frame - budget.first
}

/// Offsets all zero mean every `bap` is zero (§7.2.2.1).
fn silent(off: &Offsets, slots: usize) -> bool {
    off.csnr == 0 && off.fsnr.iter().take(slots).all(|f| *f == 0)
}

fn tally(alloc: &[Alloc], off: &Offsets) -> Tally {
    let mut t = [[0u32; 16]; 6];
    if silent(off, alloc.len()) {
        return t;
    }
    for (a, fine) in alloc.iter().zip(off.fsnr) {
        add(&mut t, a, snr_offset(off.csnr, fine));
    }
    t
}

/// The largest offsets whose mantissas fit: `csnroffst` and one `fsnroffst` for every slot
/// by one binary search over `16·csnroffst + fsnroffst`, then the LFE's (slot `lfe`) as high
/// as it still fits. `None` if not even silence fits.
pub(crate) fn tune(alloc: &[Alloc], lfe: Option<usize>, budget: &Budget) -> Option<Offsets> {
    let at = |v: u32| Offsets {
        csnr: (v >> 4) as u8,
        fsnr: [(v & 15) as u8; MAX_SLOTS],
    };
    if !fits(&tally(alloc, &at(0)), budget) {
        return None;
    }
    let (mut lo, mut hi) = (0u32, 1024u32);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if fits(&tally(alloc, &at(mid)), budget) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mut off = at(lo);
    if let Some(l) = lfe.filter(|_| !silent(&off, alloc.len())) {
        let mut base = [[0u32; 16]; 6];
        for (c, a) in alloc.iter().enumerate().filter(|(c, _)| *c != l) {
            add(&mut base, a, snr_offset(off.csnr, off.fsnr[c]));
        }
        for fine in off.fsnr[l] + 1..16 {
            let mut t = base;
            add(&mut t, &alloc[l], snr_offset(off.csnr, fine));
            if !fits(&t, budget) {
                break;
            }
            off.fsnr[l] = fine;
        }
    }
    Some(off)
}

/// Mantissa bits of each block at `off`.
pub(crate) fn mantissa_bits(alloc: &[Alloc], off: &Offsets) -> [u32; 6] {
    let t = tally(alloc, off);
    core::array::from_fn(|b| block_bits(&t[b]))
}

/// Writes every set's `bap` array at the chosen offsets.
pub(crate) fn fill(alloc: &mut [Alloc], off: &Offsets) {
    let zero = silent(off, alloc.len());
    for (a, fine) in alloc.iter_mut().zip(off.fsnr) {
        let offset = snr_offset(off.csnr, fine);
        for s in 0..6 {
            if a.set[s] != s {
                continue;
            }
            let bap = &mut a.bap[s];
            for_each_bap(&a.psd[s], &a.mask[s], a.start, a.end, offset, |i, b| {
                bap[i] = if zero { 0 } else { b };
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitalloc::curve;

    fn channel(start: usize, end: usize, level: u8, sets: [usize; 6]) -> Alloc {
        let mut a = Alloc::new(start, end);
        a.set = sets;
        for s in 0..6 {
            for (i, e) in a.exp[s].iter_mut().enumerate() {
                *e = (level + (i % 9) as u8 / 3 + s as u8).min(24);
            }
            curve(&a.exp[s], start, end, &mut a.psd[s], &mut a.mask[s]);
        }
        a
    }

    #[test]
    fn the_winner_fills_without_overflow() {
        let alloc = [
            channel(0, 253, 3, [0, 0, 0, 3, 3, 3]),
            channel(0, 199, 6, [0, 1, 1, 1, 4, 4]),
            channel(0, 7, 2, [0, 0, 2, 2, 2, 2]),
            channel(133, 217, 4, [0, 1, 2, 3, 4, 5]),
        ];
        for frame in [4000u32, 8000, 12000, 20000] {
            let budget = Budget {
                header: 70,
                side: [900, 40, 200, 300, 250, 40],
                trailer: 18,
                frame,
                first: frame * 5 / 8,
            };
            let off = tune(&alloc, Some(2), &budget).expect("silence fits");
            let t = tally(&alloc, &off);
            assert!(fits(&t, &budget), "frame {frame}: {off:?}");
            let v = 16 * u32::from(off.csnr) + u32::from(off.fsnr[0]);
            if v < 1023 {
                let mut next = off;
                next.csnr = ((v + 1) >> 4) as u8;
                next.fsnr = [((v + 1) & 15) as u8; MAX_SLOTS];
                assert!(!fits(&tally(&alloc, &next), &budget), "frame {frame}");
            }
            assert!(off.fsnr[2] >= off.fsnr[0]);
        }
    }

    #[test]
    fn fill_writes_what_the_tally_counted() {
        let mut alloc = [
            channel(0, 172, 4, [0, 0, 2, 2, 2, 5]),
            channel(0, 7, 1, [0; 6]),
            channel(145, 253, 2, [0, 0, 0, 0, 0, 0]),
        ];
        let off = Offsets {
            csnr: 20,
            fsnr: [5, 9, 3, 0, 0, 0, 0],
        };
        let want = tally(&alloc, &off);
        fill(&mut alloc, &off);
        let mut got = [[0u32; 16]; 6];
        for a in &alloc {
            for (b, g) in got.iter_mut().enumerate() {
                for bin in a.start..a.end {
                    g[usize::from(a.bap[a.set[b]][bin])] += 1;
                }
            }
        }
        assert_eq!(got, want);
        fill(&mut alloc, &Offsets::default());
        assert!(
            alloc
                .iter()
                .all(|a| a.bap.iter().flatten().all(|b| *b == 0))
        );
    }
}
