//! Exponents (§7.1, §8.2.7 to §8.2.10): raw exponents from the coefficients, the strategy
//! per block (D15/D25/D45/reuse), the ±2 slope limit and the grouped differential coding.

/// `chexpstr` codes (Table 7.4). `lfeexpstr` uses the first two.
pub(crate) const REUSE: u8 = 0;
pub(crate) const D15: u8 = 1;
pub(crate) const D25: u8 = 2;
pub(crate) const D45: u8 = 3;

/// A set starts afresh when its blocks' raw exponents move by more than this per bin on
/// average from the set's first block.
// ponytail: one fixed threshold; Stage 1b weighs exponent bits against mantissa bits.
const NEW_SET_PER_BIN: u32 = 2;

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

/// The strategy of each block: a fresh set where the raw exponents moved, sized by how many
/// blocks it covers (one: D45, two or three: D25, more: D15, §8.2.8); reuse elsewhere.
/// Block 0 always starts a set.
pub(crate) fn plan(raw: &[[u8; 256]; 6], start: usize, end: usize) -> [u8; 6] {
    let mut fresh = [false; 6];
    fresh[0] = true;
    let mut first = 0;
    for b in 1..6 {
        let moved: u32 = raw[first][start..end]
            .iter()
            .zip(&raw[b][start..end])
            .map(|(a, c)| u32::from(a.abs_diff(*c)))
            .sum();
        if moved > NEW_SET_PER_BIN * (end - start) as u32 {
            fresh[b] = true;
            first = b;
        }
    }
    let mut out = [REUSE; 6];
    let mut b = 0;
    while b < 6 {
        let len = 1 + fresh[b + 1..].iter().take_while(|s| !**s).count();
        out[b] = match len {
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
    let count = group_count(strategy, start, end).min(MAX_GROUPS);
    let n = 3 * count;
    let first = start.max(1);
    // v[0] is the absolute exponent; v[1..=n] the mapped values, each over `size` bins.
    let mut v = [0u8; 3 * MAX_GROUPS + 1];
    v[0] = env[0].min(15);
    for k in 1..=n {
        let lo = first + (k - 1) * size;
        let hi = (lo + size).min(end);
        v[k] = env
            .get(lo..hi)
            .and_then(|s| s.iter().min().copied())
            .unwrap_or(v[k - 1]);
    }
    let coupled = start > 0;
    for k in 1 + usize::from(coupled)..=n {
        v[k] = v[k].min(v[k - 1] + 2);
    }
    for k in (usize::from(coupled)..n).rev() {
        v[k] = v[k].min(v[k + 1] + 2);
    }
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
    for bin in first..end.min(256) {
        decoded[bin] = v[(1 + (bin - first) / size).min(n)];
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
    fn plan_sizes_each_set_by_its_reach() {
        let mut raw = [[10u8; 256]; 6];
        assert_eq!(plan(&raw, 0, 100), [D15, REUSE, REUSE, REUSE, REUSE, REUSE]);
        raw[3..].fill([20; 256]);
        assert_eq!(plan(&raw, 0, 100), [D25, REUSE, REUSE, D25, REUSE, REUSE]);
        raw[4] = [0; 256];
        assert_eq!(plan(&raw, 0, 100), [D25, REUSE, REUSE, D45, D45, D45]);
    }
}
