//! Channel coupling (§7.4, §8.2.4, §8.2.5): above the coupling start the full-bandwidth
//! channels share one spectrum, their sum, and each keeps its envelope as one
//! coordinate per band. A channel's coordinates go out in block 0 and again in any block
//! where one of them moves by more than half an octave.

/// Most coupling bands: the 18 sub-bands of Table 7.24, none merged.
pub(crate) const MAX_BANDS: usize = 18;

/// Most coupled channels: the five full-bandwidth channels of 3/2.
pub(crate) const MAX_FBW: usize = 5;

/// A stream's coupling range and band structure (§5.4.3.11 to §5.4.3.13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Bands {
    pub(crate) begf: u8,
    pub(crate) endf: u8,
    /// `cplbndstrc` of sub-band `begf + i` at bit `i`: set merges it into the band below.
    pub(crate) strc: u32,
}

impl Bands {
    /// `cplstrtmant`: the first coupled bin, and every coupled channel's `endmant`.
    pub(crate) fn start(&self) -> usize {
        37 + 12 * usize::from(self.begf)
    }

    /// `cplendmant`: one past the last bin of the coupling channel.
    pub(crate) fn end(&self) -> usize {
        37 + 12 * (usize::from(self.endf) + 3)
    }

    /// `ncplsubnd`.
    pub(crate) fn subbands(&self) -> usize {
        3 + usize::from(self.endf) - usize::from(self.begf)
    }

    /// `ncplbnd`.
    pub(crate) fn count(&self) -> usize {
        let merged = (self.strc >> 1) & ((1 << (self.subbands() - 1)) - 1);
        self.subbands() - merged.count_ones() as usize
    }

    /// One past the last bin of each band, in order; `count` of them are set.
    pub(crate) fn edges(&self) -> [usize; MAX_BANDS] {
        let mut out = [0; MAX_BANDS];
        let mut band = 0;
        for s in 0..self.subbands().min(MAX_BANDS) {
            if s > 0 && self.strc >> s & 1 == 0 {
                band += 1;
            }
            out[band.min(MAX_BANDS - 1)] = self.start() + 12 * (s + 1);
        }
        out
    }
}

/// One channel's coordinates in a block: `mstrcplco`, then `cplcoexp` and `cplcomant` per
/// band.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct Coords {
    pub(crate) mstr: u8,
    pub(crate) exp: [u8; MAX_BANDS],
    pub(crate) mant: [u8; MAX_BANDS],
}

/// The coupling decisions of one frame: what each block sends.
pub(crate) struct Frame {
    pub(crate) bands: Bands,
    /// `cplcoe[ch]` per block.
    pub(crate) send: [[bool; MAX_FBW]; 6],
    /// The coordinates a block sends where `send` is set.
    pub(crate) coords: [[Coords; MAX_FBW]; 6],
    /// `phsflginu`: 2/0 only, set when any block of the frame flags a band.
    pub(crate) phased: bool,
    /// `phsflg[bnd]` in effect in each block: the coupling channel of a flagged band is L − R,
    /// and the decoder negates R's reconstruction there. A block whose flags differ from
    /// the block before sends both channels' coordinates, and the flags with them.
    pub(crate) phase: [[bool; MAX_BANDS]; 6],
}

impl Frame {
    pub(crate) fn new(bands: Bands) -> Frame {
        Frame {
            bands,
            send: [[false; MAX_FBW]; 6],
            coords: [[Coords::default(); MAX_FBW]; 6],
            phased: false,
            phase: [[false; MAX_BANDS]; 6],
        }
    }
}

/// How much of a band's energy `L + R` must cancel, `−2ΣLR / (ΣL² + ΣR²)`, to set its
/// flag, and how much `L − R` would, to clear it; between, the flag holds from the block
/// before, and block 0 starts clear. Unrelated content, or one channel alone, sums as well
/// either way and leaves the flags, and their bits, off.
const FLIP: f32 = 0.5;

/// Each block's phase flags for a 2/0 pair `l`, `r` over the coupling bands ending at
/// `edges`, from `start`, by the sign of each band's cross-spectrum past `FLIP`.
fn phase_flags(
    l: &[[f32; 256]; 6],
    r: &[[f32; 256]; 6],
    start: usize,
    edges: &[usize; MAX_BANDS],
    n: usize,
    blocks: usize,
) -> [[bool; MAX_BANDS]; 6] {
    let mut out = [[false; MAX_BANDS]; 6];
    for b in 0..blocks {
        let mut lo = start;
        for k in 0..n {
            let (mut x, mut el, mut er) = (0f32, 0f32, 0f32);
            for (a, c) in l[b][lo..edges[k]].iter().zip(&r[b][lo..edges[k]]) {
                x += a * c;
                el += a * a;
                er += c * c;
            }
            let corr = 2.0 * x / (el + er).max(f32::MIN_POSITIVE);
            out[b][k] = match b {
                _ if corr < -FLIP => true,
                _ if corr > FLIP => false,
                0 => false,
                _ => out[b - 1][k],
            };
            lo = edges[k];
        }
    }
    out
}

/// The grid a coordinate is chosen on: index `16·e + 15 − m` stands for `(16 + m)/32 · 2^-e`,
/// `e` 0 to 23, so one index is one quantiser step; `ZERO` is everything smaller.
const ZERO: u16 = 16 * 24;

/// `2^e`, exact, for `e` from -126 to 127.
fn exp2(e: i32) -> f32 {
    f32::from_bits(((127 + e).clamp(1, 254) as u32) << 23)
}

/// The grid index nearest to `v`, clamped to the largest coordinate, 31/32.
fn index(v: f32) -> u16 {
    if v.is_nan() || v <= 0.0 {
        return ZERO;
    }
    let v = v.min(31.0 / 32.0);
    let e = 126 - ((v.to_bits() >> 23) & 0xff) as i32; // v·2^e in [0.5, 1)
    if e > 23 {
        return ZERO;
    }
    // m = 16 rounds up into the next octave, which the same formula indexes
    let m = (v * exp2(e) * 32.0 - 15.5) as i32;
    (16 * e + 15 - m).clamp(0, i32::from(ZERO)) as u16
}

/// `cplco` as a decoder rebuilds it (§7.4.3).
#[cfg(test)]
pub(crate) fn value(mstr: u8, exp: u8, mant: u8) -> f32 {
    let shift = -i32::from(exp) - 3 * i32::from(mstr);
    if exp == 15 {
        f32::from(mant) / 16.0 * exp2(shift)
    } else {
        (f32::from(mant) + 16.0) / 32.0 * exp2(shift)
    }
}

/// Codes a channel's band indices: `mstrcplco` scales every band as far as the largest
/// allows, and a band past the exponent's reach takes the `cplcoexp` 15 form.
fn code(idx: &[u16], out: &mut Coords) {
    let top = idx.iter().map(|i| i / 16).min().unwrap_or(0);
    out.mstr = (top / 3).min(3) as u8;
    let base = 3 * u16::from(out.mstr);
    for ((i, e), m) in idx.iter().zip(out.exp.iter_mut()).zip(out.mant.iter_mut()) {
        let shift = i / 16 - base;
        if *i < ZERO && shift <= 14 {
            (*e, *m) = (shift as u8, (15 - i % 16) as u8);
        } else if *i < ZERO {
            // (16 + m)/32 · 2^-shift as a 4-bit fraction of 2^-(15 + base)
            let frac = (31 - i % 16) as f32 / 2.0 * exp2(15 - i32::from(shift));
            (*e, *m) = (15, ((frac + 0.5) as u8).min(15));
        } else {
            (*e, *m) = (15, 0);
        }
    }
}

/// The coupling channel's largest magnitude: a block whose sum would pass it is scaled
/// down, since no exponent codes a coefficient of 1 or more.
const PEAK: f32 = 0.98;

/// Forms the coupling channel of the frame's first `blocks` blocks into `cpl`, the sum of
/// `coef[..nf]` over the coupling range, and decides `frame`'s coordinates from each band's
/// energy ratio. The sum, not the average, keeps the channel clear of the hearing threshold
/// at the top of the band; a 2/0 band whose phase flag is set sums L − R. A channel's
/// coordinates hold from block to block while every band stays within `MOVE_STEPS` of the
/// run's ratio, energies summed over the run, and the flags hold.
pub(crate) fn form(
    coef: &[[[f32; 256]; 6]],
    cpl: &mut [[f32; 256]; 6],
    frame: &mut Frame,
    blocks: usize,
) {
    let blocks = blocks.clamp(1, 6);
    let bands = frame.bands;
    let (start, end) = (bands.start(), bands.end().min(256));
    let n = bands.count().min(MAX_BANDS);
    let edges = bands.edges();
    let nf = coef.len().min(MAX_FBW);
    frame.phase = match coef {
        [l, r] => phase_flags(l, r, start, &edges, n, blocks),
        _ => [[false; MAX_BANDS]; 6],
    };
    frame.phased = frame.phase[..blocks].iter().flatten().any(|f| *f);
    let mut energy = [[0f32; MAX_BANDS]; 6];
    let mut own = [[[0f32; MAX_BANDS]; 6]; MAX_FBW];
    for b in 0..blocks {
        let out = &mut cpl[b];
        out.fill(0.0);
        for c in coef.iter().take(nf) {
            for (o, v) in out[start..end].iter_mut().zip(&c[b][start..end]) {
                *o += v;
            }
        }
        if let [l, r] = coef {
            // a flagged band carries L − R: the sum with R taken back twice
            let mut lo = start;
            for (k, hi) in edges.iter().enumerate().take(n) {
                if frame.phase[b][k] {
                    for ((o, a), c) in out[lo..*hi]
                        .iter_mut()
                        .zip(&l[b][lo..*hi])
                        .zip(&r[b][lo..*hi])
                    {
                        *o = a - c;
                    }
                }
                lo = *hi;
            }
        }
        let peak = out[start..end].iter().fold(0f32, |m, v| m.max(v.abs()));
        if peak > PEAK {
            let scale = PEAK / peak;
            for o in out[start..end].iter_mut() {
                *o *= scale;
            }
        }
        let mut lo = start;
        for k in 0..n {
            let band = lo..edges[k];
            energy[b][k] = out[band.clone()].iter().map(|v| v * v).sum();
            for (o, ch) in own.iter_mut().zip(coef) {
                o[b][k] = ch[b][band.clone()].iter().map(|v| v * v).sum();
            }
            lo = edges[k];
        }
    }
    let ratio = |own: &[f32; MAX_BANDS], cpl: &[f32; MAX_BANDS]| -> [u16; MAX_BANDS] {
        core::array::from_fn(|k| {
            if k < n && cpl[k] > 0.0 {
                index((own[k] / cpl[k]).sqrt() * 0.125).min(MOVE_FLOOR)
            } else {
                MOVE_FLOOR
            }
        })
    };
    for (c, own) in own.iter().enumerate().take(nf) {
        let mut first = 0;
        let (mut acc_own, mut acc_cpl) = (own[0], energy[0]);
        for b in 1..=blocks {
            let run = ratio(&acc_own, &acc_cpl);
            let moved = b == blocks || frame.phase[b] != frame.phase[b - 1] || {
                let here = ratio(&own[b], &energy[b]);
                here.iter()
                    .zip(&run)
                    .any(|(h, r)| h.abs_diff(*r) > MOVE_STEPS)
            };
            if b < blocks {
                frame.send[b][c] = false;
            }
            if !moved {
                for k in 0..n {
                    acc_own[k] += own[b][k];
                    acc_cpl[k] += energy[b][k];
                }
                continue;
            }
            let mut idx = [ZERO; MAX_BANDS];
            for k in 0..n {
                if acc_cpl[k] > 0.0 {
                    idx[k] = index((acc_own[k] / acc_cpl[k]).sqrt() * 0.125);
                }
            }
            frame.send[first][c] = true;
            code(&idx[..n], &mut frame.coords[first][c]);
            if b < blocks {
                (first, acc_own, acc_cpl) = (b, own[b], energy[b]);
            }
        }
    }
}

/// Coordinates at or under this index (`2^-4`: the channel 6 to 12 dB under the coupling
/// channel) count as one when deciding whether a coordinate moved.
const MOVE_FLOOR: u16 = 64;

/// Grid steps a band may move before the channel resends: 8, half an octave, 3 dB. One
/// step resends in nearly every block, and those bits cost FL 0.5 dB at 320 kbit/s with
/// nothing gained on the burst channels.
const MOVE_STEPS: u16 = 8;

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(c: &Coords, n: usize) -> Vec<f32> {
        (0..n).map(|k| value(c.mstr, c.exp[k], c.mant[k])).collect()
    }

    /// The step of the grid at `v`: 1/32 of its octave.
    fn step(v: f32) -> f32 {
        let e = 126 - ((v.to_bits() >> 23) & 0xff) as i32;
        exp2(-e.max(0)) / 32.0
    }

    #[test]
    fn coordinates_round_trip_within_a_step() {
        let mut v = 0.96f32;
        while v > 1e-6 {
            for spread in [1.0, 0.3, 1e-3, 1e-6] {
                let want = [v, v * spread, v * 0.7];
                let idx: Vec<u16> = want.iter().map(|w| index(*w)).collect();
                let mut c = Coords::default();
                code(&idx, &mut c);
                for (k, (got, w)) in decoded(&c, 3).iter().zip(want).enumerate() {
                    // past the exponent's reach the 4-bit fraction of 2^-(15 + 3·mstr)
                    // rules; below the grid, 0
                    let limit = if idx[k] == ZERO {
                        exp2(-24)
                    } else if c.exp[k] == 15 {
                        exp2(-15 - 3 * i32::from(c.mstr)) / 32.0
                    } else {
                        step(w)
                    };
                    assert!((got - w).abs() <= limit, "{w} -> {got} ({c:?})");
                }
            }
            v *= 0.83;
        }
        let mut c = Coords::default();
        code(&[ZERO, index(0.5)], &mut c);
        assert_eq!(decoded(&c, 2), [0.0, 0.5]);
        assert_eq!(index(7.0), 0); // clamps to 31/32
    }

    #[test]
    fn bands_follow_the_structure() {
        // ffmpeg's 2.0 range at 192 kbit/s: sub-bands 8 to 14 in three bands
        let b = Bands {
            begf: 8,
            endf: 12,
            strc: 0b110_1100,
        };
        assert_eq!(
            (b.start(), b.end(), b.subbands(), b.count()),
            (133, 217, 7, 3)
        );
        assert_eq!(b.edges()[..3], [145, 181, 217]);
    }

    /// A band flags when L + R would cancel past `FLIP`, clears when L − R would, and holds
    /// between; a band with one channel alone never flags.
    #[test]
    fn phase_flags_follow_the_cross_spectrum() {
        let bands = Bands {
            begf: 8,
            endf: 12,
            strc: 0b110_1100,
        };
        let (start, edges) = (bands.start(), bands.edges());
        let mut l = [[0f32; 256]; 6];
        let mut r = [[0f32; 256]; 6];
        // band 0 anti-phase in blocks 0-2, in phase after; band 1 R alone; band 2 anti-phase
        // in block 0, then 30% correlated in block 1 (between the thresholds)
        for b in 0..6 {
            for i in 0..256 {
                let x = ((i * 31 + b * 7) % 17) as f32 / 17.0 - 0.5;
                let y = ((i * 13 + b * 5) % 11) as f32 / 11.0 - 0.5;
                l[b][i] = x;
                r[b][i] = match i {
                    _ if i < edges[0] => {
                        if b < 3 {
                            -x
                        } else {
                            x
                        }
                    }
                    _ if i < edges[1] => {
                        l[b][i] = 0.0;
                        x
                    }
                    _ if b == 0 => -x,
                    _ => 0.3 * x + y,
                };
            }
        }
        let f = phase_flags(&l, &r, start, &edges, bands.count(), 6);
        let column = |k: usize| f.map(|b| b[k]);
        assert_eq!(column(0), [true, true, true, false, false, false]);
        assert_eq!(column(1), [false; 6]);
        assert!(column(2)[0] && column(2)[1], "{:?}", column(2));
    }

    #[test]
    fn a_lone_channel_is_its_coupling_channel_scaled() {
        let mut coef = vec![[[0f32; 256]; 6]; 5];
        for block in coef[2].iter_mut() {
            for (i, v) in block.iter_mut().enumerate() {
                *v = ((i * 37 % 19) as f32 - 9.0) / 20.0;
            }
        }
        let mut cpl = [[0f32; 256]; 6];
        let bands = Bands {
            begf: 8,
            endf: 8,
            strc: 0b10,
        };
        let mut frame = Frame::new(bands);
        form(&coef, &mut cpl, &mut frame, 6);
        assert!(frame.send[0].iter().all(|s| *s));
        let c = &frame.coords[0][2];
        for k in 0..bands.count() {
            // the channel is the coupling channel: cplco = 1/8
            assert!((value(c.mstr, c.exp[k], c.mant[k]) - 0.125).abs() <= step(0.125));
        }
        assert_eq!(decoded(&frame.coords[0][0], 2), [0.0, 0.0]);
        // nothing moves in block 1: nothing is sent
        assert!(frame.send[1].iter().all(|s| !*s));
    }
}
