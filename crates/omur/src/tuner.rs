//! Rate control: picks the SNR offsets that fill the frame (§8.2.12). A frame-wide search
//! over `csnroffst` and one shared `fsnroffst` comes first; then each slot's own fine offset
//! climbs, the cheapest step first, while the frame still holds it. E-AC-3, whose slots
//! share the fine offset, spends what the search leaves on fast gain codes instead. Each candidate is a bit
//! tally — packed `bap` costs per block, priced by the grouping rule — never a `bap` array;
//! only the winner's arrays are written (docs/design.md §5).

use crate::bitalloc::{
    Alloc, FGAINCOD, FLOORCOD, MAX_END, MAX_SLOTS, Offsets, for_each_address, for_each_bap,
    masking, snr_offset,
};
use crate::mantissas::{PACKED, block_bits};
use crate::tables::{BAPTAB, BNDSZ, BNDTAB, FLOORTAB, MASKTAB};

/// Where the mantissas must fit: every other bit of the frame, counted by a dry run.
pub(crate) struct Budget {
    /// syncinfo and BSI.
    pub(crate) header: u32,
    /// Each block's side information.
    pub(crate) side: [u32; 6],
    /// auxdatae, crcrsv and crc2.
    pub(crate) trailer: u32,
    pub(crate) frame: u32,
    /// The first 5/8 of an AC-3 frame, which must hold syncinfo, BSI and blocks 0 and 1,
    /// and which block 5's mantissas must not start inside (§5.5). E-AC-3 has no such rule.
    pub(crate) first: Option<u32>,
}

/// What skip fields can overshoot by: a byte's rounding plus one more field's 9 bits.
const SKIP_SLACK: u32 = 16;

/// Packed mantissa cost (`mantissas::PACKED`) per block.
pub(crate) type Tally = [u64; 6];

/// `PACKED` by bit-allocation address.
const BY_ADDRESS: [u64; 64] = {
    let mut t = [0u64; 64];
    let mut a = 0;
    while a < 64 {
        t[a] = PACKED[BAPTAB[a] as usize];
        a += 1;
    }
    t
};

/// The packed cost of one set's mantissas, `psd` against `mask`, at `offset`.
fn set_cost(psd: &[i16; 256], mask: &[i16; 50], start: usize, end: usize, offset: i32) -> u64 {
    let mut sum = 0u64;
    for_each_address(psd, mask, start, end, offset, |_, ad| {
        sum += BY_ADDRESS[ad & 63];
    });
    sum
}

/// Channel `a`'s tally at `offset`: each set's cost in every block that uses it.
fn channel(a: &Alloc, offset: i32) -> Tally {
    let mut t = [0u64; 6];
    for s in 0..a.blocks {
        if a.set[s] != s {
            continue;
        }
        let sum = set_cost(&a.psd[s], &a.mask[s], a.start, a.end, offset);
        for (blk, t) in t.iter_mut().enumerate().take(a.blocks).skip(s) {
            if a.set[blk] == s {
                *t = sum;
            }
        }
    }
    t
}

/// Channel `a`'s tally one fine step up from `offset`, from `t`, its tally at `offset`: a
/// band's masking value moves once in eight steps, so only the bands it moves in are priced
/// again.
fn climb(a: &Alloc, offset: i32, t: &Tally) -> Tally {
    let end = a.end.min(MAX_END);
    let mut out = *t;
    if a.start >= end {
        return out;
    }
    let floor = FLOORTAB[FLOORCOD as usize];
    let bands = MASKTAB[a.start] as usize..MASKTAB[end - 1] as usize + 1;
    for s in 0..a.blocks {
        if a.set[s] != s {
            continue;
        }
        let (mut add, mut sub) = (0u64, 0u64);
        for (band, mk) in a.mask[s]
            .iter()
            .enumerate()
            .take(bands.end)
            .skip(bands.start)
        {
            let m = |o: i32| ((i32::from(*mk) - o - floor).max(0) & 0x1fe0) + floor;
            let (old, new) = (m(offset), m(offset + 4));
            if old == new {
                continue;
            }
            let lo = usize::from(BNDTAB[band]).max(a.start);
            let hi = (usize::from(BNDTAB[band]) + usize::from(BNDSZ[band])).min(end);
            let cost = |p: i32, m: i32| BY_ADDRESS[((p - m) >> 5).clamp(0, 63) as usize];
            for p in a.psd[s].get(lo..hi).unwrap_or_default() {
                add += cost(i32::from(*p), new);
                sub += cost(i32::from(*p), old);
            }
        }
        for (blk, t) in out.iter_mut().enumerate().take(a.blocks).skip(s) {
            if a.set[blk] == s {
                *t = t.wrapping_add(add).wrapping_sub(sub);
            }
        }
    }
    out
}

fn sum(a: &Tally, b: &Tally) -> Tally {
    core::array::from_fn(|k| a[k] + b[k])
}

/// `a` with `old` replaced by `new`, `old` a part of `a`.
fn swap(a: &Tally, old: &Tally, new: &Tally) -> Tally {
    core::array::from_fn(|k| a[k] - old[k] + new[k])
}

fn bits(t: &Tally) -> u32 {
    t.iter().map(|b| block_bits(*b)).sum()
}

/// The frame holds it and, on AC-3, blocks 0 and 1 end inside the first 5/8, and block 5's
/// mantissas with the trailer fit the last 3/8 once skip fields move them there.
fn fits(t: &Tally, budget: &Budget) -> bool {
    let block: [u32; 6] = core::array::from_fn(|b| budget.side[b] + block_bits(t[b]));
    let total = budget.header + block.iter().sum::<u32>() + budget.trailer;
    total <= budget.frame
        && budget.first.is_none_or(|first| {
            budget.header + block[0] + block[1] <= first
                && block_bits(t[5]) + budget.trailer + SKIP_SLACK <= budget.frame - first
        })
}

/// Offsets all zero mean every `bap` is zero (§7.2.2.1).
fn silent(off: &Offsets, slots: usize) -> bool {
    off.csnr == 0 && off.fsnr.iter().take(slots).all(|f| *f == 0)
}

fn tally(alloc: &[Alloc], off: &Offsets) -> Tally {
    let mut t = [0u64; 6];
    if !silent(off, alloc.len()) {
        for (a, fine) in alloc.iter().zip(off.fsnr) {
            t = sum(&t, &channel(a, snr_offset(off.csnr, fine)));
        }
    }
    t
}

/// Fine steps every slot hands back after the shared search, for the climb to share out:
/// 4, 0.75 dB. The climb gives them to the slots whose steps are cheap, tonal and low-band
/// ones, which gain 0.5 dB on FL and FR at 320 kbit/s for 0.1 dB on the noise channels.
const GIVE: u8 = 4;

/// Every slot at `16·csnroffst + fsnroffst` = `v`.
pub(crate) fn at(v: u32) -> Offsets {
    Offsets {
        csnr: (v >> 4) as u8,
        fsnr: [(v & 15) as u8; MAX_SLOTS],
        gains: None,
    }
}

/// The largest `v = 16·csnroffst + fsnroffst` shared by every slot that the frame holds,
/// searched outwards from `hint` and then halved, with each slot's tally there. `None` if
/// not even silence fits.
pub(crate) fn search(
    alloc: &[Alloc],
    budget: &Budget,
    hint: u32,
) -> Option<(u32, [Tally; MAX_SLOTS])> {
    if !fits(&[0; 6], budget) {
        return None;
    }
    let mut here = [[0u64; 6]; MAX_SLOTS];
    let mut best = here;
    // v = 0 holds: every bap is zero (§7.2.2.1)
    let mut holds = |v: u32, keep: &mut [Tally; MAX_SLOTS]| {
        if v == 0 {
            *keep = [[0; 6]; MAX_SLOTS];
            return true;
        }
        let off = at(v);
        let mut t = [0u64; 6];
        for ((a, fine), h) in alloc.iter().zip(off.fsnr).zip(here.iter_mut()) {
            *h = channel(a, snr_offset(off.csnr, fine));
            t = sum(&t, h);
        }
        let ok = fits(&t, budget);
        if ok {
            *keep = here;
        }
        ok
    };
    // lo holds, hi does not (1024 is past the last offset)
    let (mut lo, mut hi) = (0u32, 1024u32);
    let start = hint.min(1023);
    let mut step = 1;
    if holds(start, &mut best) {
        lo = start;
        while lo + step < hi {
            if !holds(lo + step, &mut best) {
                hi = lo + step;
                break;
            }
            lo += step;
            step *= 2;
        }
    } else {
        hi = start;
        while hi > step {
            if holds(hi - step, &mut best) {
                lo = hi - step;
                break;
            }
            hi -= step;
            step *= 2;
        }
    }
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if holds(mid, &mut best) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        best = [[0; 6]; MAX_SLOTS];
    }
    Some((lo, best))
}

/// The offsets the frame is coded with: the largest `16·csnroffst + fsnroffst` shared by
/// every slot (`search`); then, unless the fine offset is `shared`, from `GIVE` steps under
/// it, one more fine step at a time for the slot whose step costs fewest bits, while the
/// frame holds it. `hint` is the last frame's shared value, set to this one's. `None` if
/// not even silence fits.
pub(crate) fn tune(
    alloc: &[Alloc],
    budget: &Budget,
    hint: &mut u32,
    shared: bool,
) -> Option<Offsets> {
    let (lo, _) = search(alloc, budget, *hint)?;
    *hint = lo;
    if lo == 0 || shared {
        return Some(at(lo));
    }
    Some(spread(alloc, budget, lo))
}

/// The per-slot climb from the shared `lo`: every slot `GIVE` fine steps under it, then one
/// more step at a time for the slot whose step costs fewest bits, while the frame holds it.
fn spread(alloc: &[Alloc], budget: &Budget, lo: u32) -> Offsets {
    let slots = alloc.len().min(MAX_SLOTS);
    let mut off = at(lo);
    let offset = |fine: u8| snr_offset(off.csnr, fine);
    let mut own = [[0u64; 6]; MAX_SLOTS];
    let mut next = [None; MAX_SLOTS];
    let mut total = [0u64; 6];
    // Grouping rounds, so a lower offset can cost a few bits more: the climb starts from
    // the lowered offsets only if they still fit.
    for give in [GIVE, 0] {
        for f in off.fsnr.iter_mut() {
            *f = ((lo & 15) as u8).saturating_sub(give);
        }
        total = [0; 6];
        for (c, a) in alloc.iter().enumerate().take(slots) {
            own[c] = channel(a, offset(off.fsnr[c]));
            total = sum(&total, &own[c]);
            next[c] = (off.fsnr[c] < 15).then(|| climb(a, offset(off.fsnr[c]), &own[c]));
        }
        if fits(&total, budget) {
            break;
        }
    }
    loop {
        let mut best: Option<(u32, usize, Tally)> = None;
        for c in 0..slots {
            let Some(up) = next[c] else { continue };
            let t = swap(&total, &own[c], &up);
            let cost = bits(&t);
            if fits(&t, budget) && best.is_none_or(|(b, ..)| cost < b) {
                best = Some((cost, c, t));
            }
        }
        let Some((_, c, t)) = best else { break };
        total = t;
        own[c] = next[c].unwrap_or(own[c]);
        off.fsnr[c] += 1;
        next[c] = (off.fsnr[c] < 15).then(|| climb(&alloc[c], offset(off.fsnr[c]), &own[c]));
    }
    off
}

/// The fast gain code a step of E-AC-3's climb moves a block to.
const RAISED: u8 = FGAINCOD + 1;

/// Fine steps times blocks of headroom a block at `RAISED` takes: a slot whose fine offset
/// sits `k` under 15 may raise `k·blocks / CAP` blocks. 16 holds the LFE within 0.2 dB of
/// AC-3's per-slot climb on the probe signal at 320 to 640 kbit/s.
const CAP: usize = 16;

/// E-AC-3's fill, the per-slot climb's counterpart for slots that share one fine offset:
/// the bits the shared search at `lo` leaves, where each slot's tally is `own`, go one
/// block of one slot at a time to fast gain code `RAISED`, the step that costs fewest bits
/// first, its codes included, while the frame holds it and the slot is under its `CAP`.
/// No slot gives bits up. `after` is set when block 0 sends codes whatever it raises, and
/// `budget` holds those. Each slot's code per block, or `None` when nothing fits.
pub(crate) fn gains(
    alloc: &[Alloc],
    budget: &Budget,
    lo: u32,
    own: &[Tally; MAX_SLOTS],
    after: bool,
) -> Option<[[u8; 6]; MAX_SLOTS]> {
    let slots = alloc.len().min(MAX_SLOTS);
    let n = alloc.first()?.blocks.min(6);
    if lo == 0 || budget.first.is_some() {
        return None;
    }
    let cap = (15 - (lo & 15) as usize) * n / CAP;
    let offset = snr_offset((lo >> 4) as u8, (lo & 15) as u8);
    // what each slot's block costs raised, packed
    let mut raise = [[0u64; 6]; MAX_SLOTS];
    for (a, r) in alloc.iter().zip(raise.iter_mut()) {
        for s in 0..n {
            if a.set[s] != s {
                continue;
            }
            let mut mask = [0i16; 50];
            masking(&a.bndpsd[s], a.start, a.end, RAISED, &mut mask);
            let up = set_cost(&a.psd[s], &mask, a.start, a.end, offset);
            for (b, r) in r.iter_mut().enumerate().take(n).skip(s) {
                if a.set[b] == s {
                    *r = up;
                }
            }
        }
    }
    let mut gain = [[FGAINCOD; 6]; MAX_SLOTS];
    let mut count = [0usize; MAX_SLOTS];
    let mut mant = [0u64; 6];
    for t in &own[..slots] {
        mant = sum(&mant, t);
    }
    // a block sends codes once it or the one before is raised, block 0 also when `after`;
    // a frame that sends any has a flag in every block
    let mut raised = [false; 6];
    let sends =
        |raised: &[bool; 6], b: usize| raised[b] || (b == 0 && after) || (b > 0 && raised[b - 1]);
    let codes = |raised: &[bool; 6]| -> u32 {
        let sent = (0..n).filter(|b| sends(raised, *b)).count() as u32;
        if sent == 0 {
            0
        } else {
            n as u32 + 3 * slots as u32 * sent
        }
    };
    let mut used = budget.header + budget.side.iter().sum::<u32>() + budget.trailer + bits(&mant);
    // the budget holds the codes the frame sends unraised
    used = used.saturating_sub(codes(&raised));
    loop {
        let base = used + codes(&raised);
        let mut best: Option<(u32, usize, usize)> = None;
        for b in 0..n {
            let mut with = raised;
            with[b] = true;
            let extra = codes(&with) - codes(&raised);
            let here = block_bits(mant[b]);
            for c in 0..slots {
                if gain[c][b] != FGAINCOD || count[c] >= cap {
                    continue;
                }
                let up = block_bits(mant[b] - own[c][b] + raise[c][b]);
                let total = base + extra + up - here;
                if total <= budget.frame && best.is_none_or(|(t, ..)| total < t) {
                    best = Some((total, c, b));
                }
            }
        }
        let Some((_, c, b)) = best else { break };
        let here = block_bits(mant[b]);
        mant[b] = mant[b] - own[c][b] + raise[c][b];
        used = used + block_bits(mant[b]) - here;
        (gain[c][b], raised[b]) = (RAISED, true);
        count[c] += 1;
    }
    count.iter().any(|k| *k > 0).then_some(gain)
}

/// Mantissa bits of each block at `off`.
pub(crate) fn mantissa_bits(alloc: &[Alloc], off: &Offsets) -> [u32; 6] {
    let t = tally(alloc, off);
    core::array::from_fn(|b| block_bits(t[b]))
}

/// Writes every set's `bap` array at the chosen offsets.
pub(crate) fn fill(alloc: &mut [Alloc], off: &Offsets) {
    let zero = silent(off, alloc.len());
    for (a, fine) in alloc.iter_mut().zip(off.fsnr) {
        let offset = snr_offset(off.csnr, fine);
        for s in 0..a.blocks {
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
    use crate::bitalloc::Gains;

    fn channel_at(start: usize, end: usize, level: u8, sets: [usize; 6]) -> Alloc {
        let mut a = Alloc::new(start, end, 6);
        a.set = sets;
        for s in 0..6 {
            for (i, e) in a.exp[s].iter_mut().enumerate() {
                *e = (level + (i % 9) as u8 / 3 + s as u8).min(24);
            }
            a.analyse(s);
        }
        a
    }

    fn frames() -> impl Iterator<Item = Budget> {
        [2000u32, 4000, 8000, 12000, 20000, 40000, 80000]
            .into_iter()
            .map(|frame| Budget {
                header: 70,
                side: [900, 40, 200, 300, 250, 40],
                trailer: 18,
                frame,
                first: Some(frame * 5 / 8),
            })
    }

    /// The winner fits, and no slot can take one more fine step: the set is maximal.
    #[test]
    fn the_winner_fills_and_no_slot_can_climb() {
        let alloc = [
            channel_at(0, 253, 3, [0, 0, 0, 3, 3, 3]),
            channel_at(0, 199, 6, [0, 1, 1, 1, 4, 4]),
            channel_at(0, 7, 2, [0, 0, 2, 2, 2, 2]),
            channel_at(133, 217, 4, [0, 1, 2, 3, 4, 5]),
        ];
        for (budget, mut hint) in frames().zip([0, 1023, 300, 77, 512, 160, 9]) {
            let frame = budget.frame;
            let off = tune(&alloc, &budget, &mut hint, false).expect("silence fits");
            assert!(
                off.csnr <= 63 && off.fsnr.iter().all(|f| *f <= 15),
                "{off:?}"
            );
            assert!(
                fits(&tally(&alloc, &off), &budget),
                "frame {frame}: {off:?}"
            );
            for c in 0..alloc.len() {
                if off.fsnr[c] < 15 && !silent(&off, alloc.len()) {
                    let mut up = off;
                    up.fsnr[c] += 1;
                    assert!(
                        !fits(&tally(&alloc, &up), &budget),
                        "frame {frame} slot {c}"
                    );
                }
            }
            let v = 16 * u32::from(off.csnr) + 15;
            if v < 1023 {
                let mut next = off;
                next.csnr += 1;
                next.fsnr = [0; MAX_SLOTS];
                assert!(!fits(&tally(&alloc, &next), &budget), "frame {frame}");
            }
        }
    }

    /// A frame of fewer blocks tallies its own and nothing past them.
    #[test]
    fn a_short_frame_tallies_its_blocks_only() {
        for blocks in [1, 2, 3] {
            let mut a = channel_at(0, 253, 3, [0, 1, 1, 0, 0, 0]);
            a.blocks = blocks;
            let t = channel(&a, snr_offset(15, 0));
            assert!(t[..blocks].iter().all(|v| *v > 0), "{blocks}: {t:?}");
            assert!(t[blocks..].iter().all(|v| *v == 0), "{blocks}: {t:?}");
        }
    }

    /// A climb prices exactly what a fresh tally one fine step up does.
    #[test]
    fn a_climb_is_the_next_offsets_tally() {
        let alloc = [
            channel_at(0, 253, 3, [0, 0, 0, 3, 3, 3]),
            channel_at(0, 7, 2, [0, 0, 2, 2, 2, 2]),
            channel_at(133, 217, 4, [0, 1, 2, 3, 4, 5]),
            channel_at(145, 253, 1, [0, 0, 0, 0, 0, 0]),
        ];
        for a in &alloc {
            for v in 0..1023 {
                let at = |v: i32| snr_offset((v >> 4) as u8, (v & 15) as u8);
                let here = channel(a, at(v));
                assert_eq!(climb(a, at(v), &here), channel(a, at(v + 1)), "{v}");
            }
        }
    }

    /// The E-AC-3 fill raises only what the shared offsets leave room for, codes included,
    /// and the regrouped sets tally what it priced.
    #[test]
    fn the_fill_spends_only_what_the_search_leaves() {
        let mut raised = 0;
        for frame in (6000u32..30000).step_by(1500) {
            let mut alloc = [
                channel_at(0, 181, 3, [0, 0, 0, 3, 3, 3]),
                channel_at(0, 7, 2, [0, 0, 0, 0, 0, 0]),
                channel_at(181, 253, 4, [0, 1, 1, 1, 4, 4]),
            ];
            let budget = Budget {
                header: 120,
                side: [900, 40, 200, 300, 250, 40],
                trailer: 18,
                frame,
                first: None,
            };
            let (lo, own) = search(&alloc, &budget, 300).unwrap();
            let Some(gain) = gains(&alloc, &budget, lo, &own, false) else {
                continue;
            };
            for (a, g) in alloc.iter_mut().zip(&gain) {
                a.regroup(g);
            }
            let codes = Gains::of(&gain[..3], 6, false).unwrap();
            let sent = codes.sent.iter().filter(|s| **s).count() as u32;
            let t = tally(&alloc, &at(lo));
            let used = budget.header + budget.side.iter().sum::<u32>() + budget.trailer;
            assert!(used + bits(&t) + 6 + 9 * sent <= frame, "frame {frame}");
            raised += 1;
        }
        assert!(raised > 4, "{raised}");
    }

    #[test]
    fn fill_writes_what_the_tally_counted() {
        let mut alloc = [
            channel_at(0, 172, 4, [0, 0, 2, 2, 2, 5]),
            channel_at(0, 7, 1, [0; 6]),
            channel_at(145, 253, 2, [0, 0, 0, 0, 0, 0]),
        ];
        let off = Offsets {
            csnr: 20,
            fsnr: [5, 9, 3, 0, 0, 0, 0],
            gains: None,
        };
        let want = tally(&alloc, &off);
        fill(&mut alloc, &off);
        let mut got = [0u64; 6];
        for a in &alloc {
            for (b, g) in got.iter_mut().enumerate() {
                for bin in a.start..a.end {
                    *g += PACKED[usize::from(a.bap[a.set[b]][bin])];
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
