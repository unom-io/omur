//! Exponents (§7.1, §8.2.7 to §8.2.10): raw exponents from the coefficients, the strategy
//! per block (D15/D25/D45/reuse), the ±2 slope limit and the grouped differential coding.

/// `chexpstr` codes (Table 7.4). `lfeexpstr` uses the first two.
pub(crate) const REUSE: u8 = 0;
pub(crate) const D15: u8 = 1;
pub(crate) const D25: u8 = 2;
pub(crate) const D45: u8 = 3;

/// The most 7-bit groups a channel sends: D15 over 253 bins.
pub(crate) const MAX_GROUPS: usize = 84;

/// One coded exponent set: the absolute first exponent and the 7-bit groups after it.
#[derive(Clone, Copy)]
pub(crate) struct Groups {
    pub(crate) absexp: u8,
    pub(crate) codes: [u8; MAX_GROUPS],
    pub(crate) count: usize,
}

impl Default for Groups {
    fn default() -> Groups {
        Groups {
            absexp: 0,
            codes: [0; MAX_GROUPS],
            count: 0,
        }
    }
}

/// Raw exponents of `coef[..end]`: leading zeros of each magnitude below 1, 0 to 24.
pub(crate) fn extract(coef: &[f32; 256], end: usize, out: &mut [u8; 256]) {
    for (e, c) in out.iter_mut().zip(coef).take(end) {
        let biased = ((c.to_bits() >> 23) & 0xff) as i32;
        *e = (126 - biased).clamp(0, 24) as u8;
    }
}

/// Bins each mapped value spans under `strategy`.
fn group_size(strategy: u8) -> usize {
    match strategy {
        D25 => 2,
        D45 => 4,
        _ => 1,
    }
}

/// 7-bit groups a set over `start..end` sends under `strategy` (§7.1.3): after the absolute
/// exponent of bin 0, or over the coupling range from its start.
pub(crate) fn group_count(strategy: u8, start: usize, end: usize) -> usize {
    let size = 3 * group_size(strategy);
    match strategy {
        REUSE => 0,
        _ if start > 0 => (end.saturating_sub(start)) / size,
        D15 => (end - 1) / 3,
        D25 => (end - 1 + 3) / 6,
        _ => (end - 1 + 9) / 12,
    }
}

/// The strategy of each of a frame's `blocks` blocks. A block starts a fresh set when its
/// raw exponents have moved from the set's first block by more exponent steps, summed over
/// the bins, than a fresh D15 set costs in bits (`fixed` plus its groups): a bin's mantissa
/// takes about a bit more for each step its envelope sits under it. Each set is sized by how
/// many blocks it covers (one: D45, two or three: D25, more: D15, §8.2.8), the 32 splits of
/// six blocks ffmpeg's streams use; D15 or reuse only where `d15_only` (the LFE). Block 0
/// always starts a set; blocks past the frame read as reuse.
pub(crate) fn plan(
    raw: &[[u8; 256]; 6],
    start: usize,
    end: usize,
    fixed: u32,
    d15_only: bool,
    blocks: usize,
) -> [u8; 6] {
    let n = blocks.min(6);
    let price = fixed + 7 * group_count(D15, start, end) as u32;
    let mut fresh = [false; 6];
    fresh[0] = true;
    let mut first = 0;
    for b in 1..n {
        let moved: u32 = raw[first][start..end]
            .iter()
            .zip(&raw[b][start..end])
            .map(|(a, c)| u32::from(a.abs_diff(*c)))
            .sum();
        if moved > price {
            fresh[b] = true;
            first = b;
        }
    }
    let mut out = [REUSE; 6];
    let mut b = 0;
    while b < n {
        let len = 1 + fresh[b + 1..n].iter().take_while(|s| !**s).count();
        out[b] = match len {
            _ if d15_only => D15,
            1 => D45,
            2 | 3 => D25,
            _ => D15,
        };
        b += len;
    }
    out
}

/// Codes the envelope `env[start..end]` under `strategy`, writing the groups and the
/// exponents a decoder rebuilds from them. Exponents only move down — shared by a pair or
/// quad, capped at 15 for bin 0, slope-limited to ±2 — so no coefficient is under-stated.
/// From bin 0 the absolute exponent is bin 0's; over the coupling range it is an even
/// reference at most 1 below the first value (`cplabsexp`, §7.1.3).
pub(crate) fn encode(
    env: &[u8; 256],
    start: usize,
    end: usize,
    strategy: u8,
    groups: &mut Groups,
    decoded: &mut [u8; 256],
) {
    let size = group_size(strategy);
    let shift = size.trailing_zeros();
    let count = group_count(strategy, start, end).min(MAX_GROUPS);
    let n = 3 * count;
    let first = start.max(1);
    let end = end.min(256);
    // v[0] is the absolute exponent; v[1..=n] the mapped values, each over `size` bins:
    // the whole spans first, then a part one, then the last value again past the end
    let mut v = [0u8; 3 * MAX_GROUPS + 1];
    v[0] = env[0].min(15);
    let bins = env.get(first..end.max(first)).unwrap_or_default();
    let whole = (bins.len() >> shift).min(n);
    match size {
        1 => v[1..=whole].copy_from_slice(&bins[..whole]),
        2 => {
            for (o, p) in v[1..=whole].iter_mut().zip(bins.as_chunks::<2>().0) {
                *o = p[0].min(p[1]);
            }
        }
        _ => {
            for (o, q) in v[1..=whole].iter_mut().zip(bins.as_chunks::<4>().0) {
                *o = q[0].min(q[1]).min(q[2].min(q[3]));
            }
        }
    }
    for k in whole + 1..=n {
        let part = bins.get((k - 1) << shift..).unwrap_or_default();
        v[k] = part.iter().take(size).min().copied().unwrap_or(v[k - 1]);
    }
    let coupled = start > 0;
    slope_limit(&mut v[usize::from(coupled)..=n]);
    if coupled {
        v[0] = v[1] & !1;
    }
    groups.absexp = if coupled { v[0] / 2 } else { v[0] };
    groups.count = count;
    for (g, code) in groups.codes.iter_mut().take(count).enumerate() {
        let m = |i: usize| v[3 * g + i + 1] + 2 - v[3 * g + i];
        *code = 25 * m(0) + 5 * m(1) + m(2);
    }
    if !coupled {
        decoded[0] = v[0];
    }
    // each value over its `size` bins, the last one on to the end
    let spans = decoded.get_mut(first..end.max(first)).unwrap_or_default();
    let mut rest = spans.chunks_mut(size).zip(&v[1..=n]);
    for (d, value) in &mut rest {
        d.fill(*value);
    }
    let done = (n * size).min(spans.len());
    spans[done..].fill(v[n]);
}

/// Limits every step of `v` to ±2 by lowering values, never raising one: up the slice,
/// each at most 2 over the one before, then down it, each at most 2 over the one after
/// (§8.2.10). As prefix minima of `v[j] ∓ 2j`, so no step waits on the last.
fn slope_limit(v: &mut [u8]) {
    let mut w = [0i16; 3 * MAX_GROUPS + 1];
    let w = &mut w[..v.len()];
    let mut low = i16::MAX;
    for (k, (w, x)) in w.iter_mut().zip(v.iter()).enumerate() {
        low = low.min(i16::from(*x) - 2 * k as i16);
        *w = low + 2 * k as i16;
    }
    let mut low = i16::MAX;
    for (k, (x, w)) in v.iter_mut().zip(w.iter()).enumerate().rev() {
        low = low.min(*w + 2 * k as i16);
        *x = (low - 2 * k as i16) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decoder's side of §7.1.3: groups back to absolute exponents over `start..end`.
    fn rebuild(groups: &Groups, strategy: u8, start: usize, end: usize) -> Vec<u8> {
        let size = group_size(strategy);
        let coupled = start > 0;
        let mut prev = groups.absexp as i32 * if coupled { 2 } else { 1 };
        let mut exp = if coupled { vec![] } else { vec![groups.absexp] };
        for &code in &groups.codes[..groups.count] {
            assert!(code <= 124, "grouped value {code}");
            for m in [code / 25, code % 25 / 5, code % 5] {
                prev += m as i32 - 2;
                assert!((0..=24).contains(&prev));
                exp.extend(std::iter::repeat_n(prev as u8, size));
            }
        }
        assert!(exp.len() >= end - start);
        exp.truncate(end - start);
        exp
    }

    fn envelopes() -> Vec<[u8; 256]> {
        let mut seed = 7u32;
        let mut rnd = move |m: u32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 16) % m
        };
        let noisy = core::array::from_fn(|_| rnd(25) as u8);
        let smooth = core::array::from_fn(|i| (4 + i / 12).min(24) as u8);
        let cliff = core::array::from_fn(|i| if i < 40 { 2 } else { 24 });
        vec![noisy, smooth, cliff, [24; 256], [0; 256]]
    }

    #[test]
    fn every_strategy_round_trips_and_never_raises_an_exponent() {
        let ranges = [
            (0, 7),
            (0, 73),
            (0, 172),
            (0, 247),
            (0, 250),
            (0, 253),
            (37, 253),
            (133, 169),
            (181, 217),
        ];
        for env in envelopes() {
            for (start, end) in ranges {
                for strategy in [D15, D25, D45] {
                    let mut groups = Groups::default();
                    let mut decoded = [0u8; 256];
                    encode(&env, start, end, strategy, &mut groups, &mut decoded);
                    let got = rebuild(&groups, strategy, start, end);
                    assert_eq!(got, decoded[start..end], "{start}..{end}");
                    for bin in start..end {
                        assert!(decoded[bin] <= env[bin], "{start}..{end} bin {bin}");
                    }
                    assert!(groups.absexp <= 15);
                }
            }
        }
    }

    #[test]
    fn extraction_counts_leading_zeros() {
        let mut coef = [0f32; 256];
        coef[..6].copy_from_slice(&[0.75, -0.5, 0.4999, 1.5, 1e-9, 0.0]);
        let mut raw = [0u8; 256];
        extract(&coef, 6, &mut raw);
        assert_eq!(raw[..6], [0, 0, 1, 0, 24, 24]);
    }

    #[test]
    fn plan_weighs_exponent_bits_against_mantissa_bits() {
        let mut raw = [[10u8; 256]; 6];
        assert_eq!(
            plan(&raw, 0, 100, 6, false, 6),
            [D15, REUSE, REUSE, REUSE, REUSE, REUSE]
        );
        // 100 bins two steps quieter: 200 bits, less than a set's 237
        raw[3..].fill([12; 256]);
        assert_eq!(plan(&raw, 0, 100, 6, false, 6)[1..], [REUSE; 5]);
        raw[3..].fill([20; 256]);
        assert_eq!(
            plan(&raw, 0, 100, 6, false, 6),
            [D25, REUSE, REUSE, D25, REUSE, REUSE]
        );
        raw[4] = [0; 256];
        assert_eq!(
            plan(&raw, 0, 100, 6, false, 6),
            [D25, REUSE, REUSE, D45, D45, D45]
        );
        // a shorter frame plans its own blocks and leaves the rest at reuse
        assert_eq!(
            plan(&raw, 0, 100, 6, false, 3),
            [D25, REUSE, REUSE, REUSE, REUSE, REUSE]
        );
        assert_eq!(
            plan(&raw, 0, 100, 6, false, 1),
            [D45, REUSE, REUSE, REUSE, REUSE, REUSE]
        );
        assert_eq!(
            plan(&raw, 0, 7, 4, true, 6),
            [D15, REUSE, REUSE, D15, D15, D15]
        );
        // a coupling channel from bin 133: 36 bins a step apart cost less than its 88 bits
        let mut cpl = [[24u8; 256]; 6];
        cpl.iter_mut().for_each(|b| b[133..169].fill(8));
        cpl[1][133..169].fill(9);
        assert_eq!(plan(&cpl, 133, 169, 4, false, 6)[1..], [REUSE; 5]);
        cpl[1][133..169].fill(11);
        assert_eq!(plan(&cpl, 133, 169, 4, false, 6)[..2], [D45, D45]);
    }
}
