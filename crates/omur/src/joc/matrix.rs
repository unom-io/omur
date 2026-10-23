//! Reconstruction matrices (TS 103 420 clause 6): per element, data point and parameter
//! band, the regularised least-squares fit of element ≈ matrix × core over the band's QMF
//! samples, quantised coarse or fine, Huffman coded, under the frame's byte budget.
//!
//! The QMF's 64 subbands group into the 23 bands of Table 54's finest column, the "split";
//! every coarser column is a union of them. A frame sums, per subband, the core's covariance
//! and each element's correlation with the core under the smooth ramp's weights, with
//! timeslot 0 apart and a snapshot where a steep split lands; grouped into the split once a
//! frame, they give every fit below as a quadratic per split band.
//!
//! What a frame sends per element, cheapest first: `Reuse` (the decoder holds last frame's
//! matrix); one smooth data point (the matrix ramps there over the frame); two steep ones
//! (the old matrix for timeslot 0, the first to timeslot `b`, the second after) when the
//! element moved or a transient landed mid-frame. Sparse mode keeps one core channel and
//! gives the others +0.4, Pseudocode 2's offset of 50 over the centre 48; it is written for
//! one band, where its channel index reads one way, and taken where it saves bits for no
//! more error.

use super::qmf::SUBBANDS;
use super::tables::{
    ATOM_BAND, ATOM_START, COARSE_MTX_CODE, COARSE_MTX_LEN, COARSE_VEC_CODE, COARSE_VEC_LEN,
    FINE_MTX_CODE, FINE_MTX_LEN, FINE_VEC_CODE, FINE_VEC_LEN, NUM_BANDS,
};
use crate::bits::{BitWriter, Overflow};

/// Core channels the matrices read: L, R, C, Ls, Rs (Table 53, `joc_dmx_config_idx` 0).
pub(crate) const CH: usize = 5;
/// Bands of the 23-band split.
pub(crate) const SPLIT: usize = 23;
/// QMF timeslots in a six-block frame.
pub(crate) const SLOTS: usize = 24;
/// The 15 entries of a symmetric 5 × 5 matrix, row by row from the diagonal.
const PAIRS: [(usize, usize); 15] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 3),
    (3, 4),
    (4, 4),
];
/// Where `(c, d)` sits in `PAIRS`.
const fn pair(c: usize, d: usize) -> usize {
    let (a, b) = if c <= d { (c, d) } else { (d, c) };
    [0, 5, 9, 12, 14][a] + b - a
}

/// Bits of `joc_info` per object present: the present flag, `joc_num_bands_idx`,
/// `b_joc_sparse`, `joc_num_quant_idx`, `joc_slope_idx`, `joc_num_dpoints_bits`.
const INFO_BITS: u32 = 8;
/// The steep offsets of two data points.
const OFFSET_BITS: u32 = 10;
/// Bits of `joc_header` and the frame part of `joc_info`.
const HEADER_BITS: u32 = 12 + 18;

/// A quadratic error `a + lin·m + mᵀ·q·m` of a matrix row `m`.
#[derive(Clone, Copy, Default)]
struct Quad {
    a: f32,
    lin: [f32; CH],
    q: [f32; 15],
}

impl Quad {
    fn add(&mut self, o: &Quad) {
        self.a += o.a;
        for (x, y) in self.lin.iter_mut().zip(o.lin) {
            *x += y;
        }
        for (x, y) in self.q.iter_mut().zip(o.q) {
            *x += y;
        }
    }

    fn sub(&mut self, o: &Quad) {
        self.a -= o.a;
        for (x, y) in self.lin.iter_mut().zip(o.lin) {
            *x -= y;
        }
        for (x, y) in self.q.iter_mut().zip(o.q) {
            *x -= y;
        }
    }

    /// The error of holding `m` over a span with these sums: `E − 2m·C + mᵀRm`.
    fn held(e: f32, c: &[f32; CH], r: &[f32; 15]) -> Quad {
        Quad {
            a: e,
            lin: c.map(|v| -2.0 * v),
            q: *r,
        }
    }

    fn eval(&self, m: &[f32; CH]) -> f32 {
        let mut e = self.a;
        for (l, v) in self.lin.iter().zip(m) {
            e += l * v;
        }
        for (q, &(c, d)) in self.q.iter().zip(&PAIRS) {
            let twice = if c == d { 1.0 } else { 2.0 };
            e += twice * q * m[c] * m[d];
        }
        e.max(0.0)
    }
}

/// Quadratics per band, a field at a time: band `b`'s is `a[b]`, `lin[c][b]`, `q[p][b]`, so
/// that every step of a fit runs over the bands side by side.
#[derive(Clone, Copy, Default)]
struct Sums {
    a: [f32; SPLIT],
    lin: [[f32; SPLIT]; CH],
    q: [[f32; SPLIT]; 15],
}

impl Sums {
    fn add(&mut self, b: usize, o: &Quad) {
        self.a[b] += o.a;
        for (x, y) in self.lin.iter_mut().zip(o.lin) {
            x[b] += y;
        }
        for (x, y) in self.q.iter_mut().zip(o.q) {
            x[b] += y;
        }
    }

    /// Each of the first `nb` bands' minimiser, by channel and band, regularised toward
    /// `toward` by `1e-3` of the mean diagonal plus `floor`, through a Cholesky
    /// factorisation: a band the core leaves empty keeps the matrix it was pulled to. The
    /// regularisation bounds the condition number near 5,000, which f32 resolves far below
    /// a quantiser step. Each step runs over the bands.
    #[allow(clippy::needless_range_loop)] // the factorisation reads clearest by index
    fn solve(&self, toward: &[[f32; SPLIT]; CH], floor: f32, nb: usize) -> [[f32; SPLIT]; CH] {
        let nb = nb.min(SPLIT);
        let mut lambda = [0f32; SPLIT];
        for b in 0..nb {
            let trace = (0..CH).map(|c| self.q[pair(c, c)][b]).sum::<f32>();
            lambda[b] = 1e-3 * trace / CH as f32 + floor + 1e-30;
        }
        let mut l = [[[0f32; SPLIT]; CH]; CH];
        let mut inv = [[0f32; SPLIT]; CH];
        for i in 0..CH {
            for j in 0..=i {
                for b in 0..nb {
                    let mut s = self.q[pair(i, j)][b] + if i == j { lambda[b] } else { 0.0 };
                    for k in 0..j {
                        s -= l[i][k][b] * l[j][k][b];
                    }
                    if i == j {
                        l[i][i][b] = s.max(1e-30).sqrt();
                        inv[i][b] = 1.0 / l[i][i][b];
                    } else {
                        l[i][j][b] = s * inv[j][b];
                    }
                }
            }
        }
        // (q + λ) m = -lin / 2 + λ·toward
        let mut z = [[0f32; SPLIT]; CH];
        for i in 0..CH {
            for b in 0..nb {
                let mut s = -0.5 * self.lin[i][b] + lambda[b] * toward[i][b];
                for k in 0..i {
                    s -= l[i][k][b] * z[k][b];
                }
                z[i][b] = s * inv[i][b];
            }
        }
        let mut m = [[0f32; SPLIT]; CH];
        for i in (0..CH).rev() {
            for b in 0..nb {
                let mut s = z[i][b];
                for k in i + 1..CH {
                    s -= l[k][i][b] * m[k][b];
                }
                m[i][b] = s * inv[i][b];
            }
        }
        m.map(|m| m.map(|v| if v.is_finite() { v } else { 0.0 }))
    }

    /// [`Quad::eval`] of the first `nb` bands at `m`, by channel and band.
    fn eval(&self, m: &[[f32; SPLIT]; CH], nb: usize) -> [f32; SPLIT] {
        let nb = nb.min(SPLIT);
        let mut e = self.a;
        for (l, m) in self.lin.iter().zip(m) {
            for b in 0..nb {
                e[b] += l[b] * m[b];
            }
        }
        for (q, &(c, d)) in self.q.iter().zip(&PAIRS) {
            let twice = if c == d { 1.0 } else { 2.0 };
            for b in 0..nb {
                e[b] += twice * q[b] * m[c][b] * m[d][b];
            }
        }
        e.map(|e| e.max(0.0))
    }
}

/// `joc_num_quant_idx` 0 or 1 (Table 51): 96 or 192 steps of `820 / (4096·(1 + idx))`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Quant {
    fine: bool,
}

impl Quant {
    fn steps(self) -> i32 {
        if self.fine { 192 } else { 96 }
    }

    fn step(self) -> f32 {
        if self.fine {
            820.0 / 8192.0
        } else {
            820.0 / 4096.0
        }
    }

    /// The nearest value index, `q − nquant/2` of Pseudocode 5.
    fn index(self, m: f32) -> i16 {
        let half = self.steps() / 2;
        ((m / self.step() + 0.5).floor() as i32).clamp(-half, half - 1) as i16
    }

    fn value(self, v: i16) -> f32 {
        f32::from(v) * self.step()
    }

    /// `joc_mtx` code word and length of a difference of value indices (Pseudocode 3).
    fn mtx(self, delta: i32) -> (u32, u8) {
        if self.fine {
            let s = delta.rem_euclid(192) as usize;
            (FINE_MTX_CODE[s], FINE_MTX_LEN[s])
        } else {
            let s = delta.rem_euclid(96) as usize;
            (COARSE_MTX_CODE[s], COARSE_MTX_LEN[s])
        }
    }

    /// `joc_vec` code word of a sparse band's value index, counted from the sparse offset
    /// of Pseudocode 2 (50 or 100 steps, two or four above the centre).
    fn vec(self, v: i16) -> (u32, u8) {
        let s = i32::from(v) - self.sparse();
        if self.fine {
            let s = s.rem_euclid(192) as usize;
            (FINE_VEC_CODE[s], FINE_VEC_LEN[s])
        } else {
            let s = s.rem_euclid(96) as usize;
            (COARSE_VEC_CODE[s], COARSE_VEC_LEN[s])
        }
    }

    /// The value index a sparse band gives its other channels: Pseudocode 2's offset.
    fn sparse(self) -> i32 {
        if self.fine { 4 } else { 2 }
    }
}

/// How an element's matrix moves over the frame (Pseudocode 6).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// Held from last frame: `b_joc_obj_present` clear.
    Reuse,
    /// One data point, reached by linear interpolation at the frame's end.
    Smooth,
    /// One data point from timeslot 1: no lag, a step at the frame's start.
    Jump,
    /// Two data points: the first from timeslot 1, the second from timeslot `b`.
    Steep(u8),
}

/// One element's coded matrix for the frame.
#[derive(Clone, Copy)]
struct Coded {
    mode: Mode,
    quant: Quant,
    /// `joc_num_bands_idx`.
    bands: u8,
    /// The channel a sparse frame keeps.
    sparse: Option<u8>,
    /// Value indices by data point, channel and band.
    v: [[[i16; SPLIT]; CH]; 2],
    bits: u32,
    err: f32,
}

impl Coded {
    fn reuse(err: f32) -> Coded {
        Coded {
            mode: Mode::Reuse,
            quant: Quant { fine: false },
            bands: 0,
            sparse: None,
            v: [[[0; SPLIT]; CH]; 2],
            bits: 1,
            err,
        }
    }

    fn points(&self) -> usize {
        match self.mode {
            Mode::Reuse => 0,
            Mode::Smooth | Mode::Jump => 1,
            Mode::Steep(_) => 2,
        }
    }
}

/// The rungs an element steps down when the frame is over budget: fine at 23 bands, then
/// coarse at each band count of Table 50 from 23 down.
const RUNGS: [(bool, u8); 9] = [
    (true, 7),
    (false, 7),
    (false, 6),
    (false, 5),
    (false, 4),
    (false, 3),
    (false, 2),
    (false, 1),
    (false, 0),
];

fn atom_band(bands: u8, atom: usize) -> usize {
    usize::from(ATOM_BAND[usize::from(bands) * SPLIT + atom])
}

/// Sums of a subband's products over a frame.
type Bands = [f32; SUBBANDS];

/// Sums `v` over each split band.
fn group(v: &Bands) -> [f32; SPLIT] {
    std::array::from_fn(|g| {
        let (a, b) = (usize::from(ATOM_START[g]), usize::from(ATOM_START[g + 1]));
        v[a..b].iter().sum()
    })
}

/// One element's running sums: `Re Σ z·conj(x_c)` plain and weighted by `w`, `Σ |z|²`,
/// timeslot 0 apart, its energy per timeslot, and the plain sums of it and of the core
/// before `split`, the timeslot a steep second data point starts.
struct Element {
    c: [[Bands; CH]; 2],
    c0: [Bands; CH],
    e: Bands,
    e0: Bands,
    level: [f32; SLOTS],
    split: Option<u8>,
    sx: [Bands; 15],
    sc: [Bands; CH],
    se: Bands,
}

/// One element's frame as quadratics per split band: under the smooth ramp, held from
/// timeslot 1, and the two steep halves; the error timeslot 0 takes at the held matrix, the
/// error of reusing it all frame, and the element's energy.
#[derive(Clone, Copy)]
struct Fits {
    smooth: [Quad; SPLIT],
    jump: [Quad; SPLIT],
    first: [Quad; SPLIT],
    second: [Quad; SPLIT],
    pre: f32,
    reuse: f32,
    energy: f32,
    /// The cheaper of smooth and jump at the top rung, which the lower rungs keep.
    family: Mode,
}

/// What the encoder keeps per frame and across frames for the matrices.
pub(crate) struct Joc {
    elements: usize,
    /// The core's covariance `Re Σ x_c·conj(x_d)` per pair and subband: plain, weighted by
    /// `w = (t + 1)/24`, by `w²`; then timeslot 0 alone.
    x: Box<[[Bands; 15]; 3]>,
    x0: Box<[Bands; 15]>,
    sums: Box<[Element]>,
    /// The matrix the decoder ends the last frame with, per element and split band.
    prev: Box<[[[f32; CH]; SPLIT]]>,
    /// Per element and split band, its energy remembered over about a second: a band it
    /// has used keeps its matrix through a pause, one it never uses falls to zero.
    seen: Box<[[f32; SPLIT]]>,
    fits: Box<[Fits]>,
    coded: Box<[Coded]>,
    /// The rung each element stands on, and what the next one down costs.
    rung: Box<[usize]>,
    next: Box<[Option<Coded>]>,
    /// The frame's regularisation floor: a millionth of the core's mean band power.
    floor: f32,
    /// `joc_seq_count_bits` of the next frame.
    seq: u16,
}

impl Joc {
    pub(crate) fn new(elements: usize) -> Joc {
        let zero = Element {
            c: [[[0.0; SUBBANDS]; CH]; 2],
            c0: [[0.0; SUBBANDS]; CH],
            e: [0.0; SUBBANDS],
            e0: [0.0; SUBBANDS],
            level: [0.0; SLOTS],
            split: None,
            sx: [[0.0; SUBBANDS]; 15],
            sc: [[0.0; SUBBANDS]; CH],
            se: [0.0; SUBBANDS],
        };
        let q = [Quad::default(); SPLIT];
        let fits = Fits {
            smooth: q,
            jump: q,
            first: q,
            second: q,
            pre: 0.0,
            reuse: 0.0,
            energy: 0.0,
            family: Mode::Smooth,
        };
        Joc {
            elements,
            x: Box::new([[[0.0; SUBBANDS]; 15]; 3]),
            x0: Box::new([[0.0; SUBBANDS]; 15]),
            sums: (0..elements)
                .map(|_| Element { ..zero })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            prev: vec![[[0.0; CH]; SPLIT]; elements].into_boxed_slice(),
            seen: vec![[0.0; SPLIT]; elements].into_boxed_slice(),
            fits: vec![fits; elements].into_boxed_slice(),
            coded: vec![Coded::reuse(0.0); elements].into_boxed_slice(),
            rung: vec![0; elements].into_boxed_slice(),
            next: vec![None; elements].into_boxed_slice(),
            floor: 0.0,
            seq: 0,
        }
    }

    /// Takes timeslot `t`'s QMF of the core (`x`, real and imaginary) and of each element
    /// (`z`); an element marked silent in `quiet` adds nothing. An element whose `events`
    /// entry is `t`, or whose energy at `t` passes eight times its mean before, splits there.
    pub(crate) fn accumulate(
        &mut self,
        t: usize,
        x: &[[[f32; SUBBANDS]; 2]; CH],
        z: &[[[f32; SUBBANDS]; 2]],
        quiet: &[bool],
        events: &[Option<u8>],
    ) {
        let w = (t + 1) as f32 / SLOTS as f32;
        for (k, (zk, s)) in z.iter().zip(self.sums.iter_mut()).enumerate() {
            if quiet.get(k).copied().unwrap_or(true) {
                continue;
            }
            let mut e = [0f32; SUBBANDS];
            for (sb, v) in e.iter_mut().enumerate() {
                *v = zk[0][sb] * zk[0][sb] + zk[1][sb] * zk[1][sb];
            }
            let level: f32 = e.iter().sum();
            let before: f32 = s.level[..t].iter().sum();
            let top = s.level[..t].iter().cloned().fold(level, f32::max);
            let transient = t >= 2 && level > 8.0 * before / t as f32 && level > 1e-3 * top;
            let event = events.get(k).copied().flatten() == Some(t as u8);
            if s.split.is_none() && t >= 2 && (transient || event) {
                s.split = Some(t as u8);
                s.sx = self.x[0];
                s.sc = s.c[0];
                s.se = s.e;
            }
            s.level[t] = level;
            for (a, v) in s.e.iter_mut().zip(e) {
                *a += v;
            }
            let [plain, weighted] = &mut s.c;
            for ((p, q), [xr, xi]) in plain.iter_mut().zip(weighted.iter_mut()).zip(x) {
                for ((((p, q), zr), zi), (xr, xi)) in p
                    .iter_mut()
                    .zip(q.iter_mut())
                    .zip(&zk[0])
                    .zip(&zk[1])
                    .zip(xr.iter().zip(xi))
                {
                    let v = zr * xr + zi * xi;
                    *p += v;
                    *q += w * v;
                }
            }
            if t == 0 {
                (s.c0, s.e0) = (s.c[0], s.e);
            }
        }
        let ww = w * w;
        let [plain, weighted, squared] = &mut *self.x;
        for (p, &(c, d)) in PAIRS.iter().enumerate() {
            let ([cr, ci], [dr, di]) = (&x[c], &x[d]);
            let sums = plain[p]
                .iter_mut()
                .zip(weighted[p].iter_mut())
                .zip(squared[p].iter_mut());
            for (((a, b), q), (((cr, ci), dr), di)) in sums.zip(cr.iter().zip(ci).zip(dr).zip(di)) {
                let v = cr * dr + ci * di;
                *a += v;
                *b += w * v;
                *q += ww * v;
            }
        }
        if t == 0 {
            *self.x0 = self.x[0];
        }
    }

    /// Clears the sums for the next frame.
    fn reset(&mut self) {
        *self.x = [[[0.0; SUBBANDS]; 15]; 3];
        *self.x0 = [[0.0; SUBBANDS]; 15];
        for s in self.sums.iter_mut() {
            s.c = [[[0.0; SUBBANDS]; CH]; 2];
            s.c0 = [[0.0; SUBBANDS]; CH];
            (s.e, s.e0) = ([0.0; SUBBANDS], [0.0; SUBBANDS]);
            s.level = [0.0; SLOTS];
            s.split = None;
        }
    }

    /// Element `k`'s quadratics for the frame, from the core's grouped sums `r` (plain, by
    /// `w`, by `w²`) and `r0` (timeslot 0).
    fn fit(&mut self, k: usize, r: &[[[f32; 15]; SPLIT]; 3], r0: &[[f32; 15]; SPLIT]) {
        let s = &self.sums[k];
        let prev = &self.prev[k];
        let gc = |v: &[Bands; CH]| -> [[f32; CH]; SPLIT] {
            let g: [[f32; SPLIT]; CH] = std::array::from_fn(|c| group(&v[c]));
            std::array::from_fn(|b| std::array::from_fn(|c| g[c][b]))
        };
        let (c1, cw, c0) = (gc(&s.c[0]), gc(&s.c[1]), gc(&s.c0));
        let (e1, e0) = (group(&s.e), group(&s.e0));
        let f = &mut self.fits[k];
        let (mut reuse, mut pre, mut energy) = (0f32, 0f32, 0f32);
        for g in 0..SPLIT {
            let total = Quad::held(e1[g], &c1[g], &r[0][g]);
            let zero = Quad::held(e0[g], &c0[g], &r0[g]);
            reuse += total.eval(&prev[g]);
            pre += zero.eval(&prev[g]);
            energy += e1[g];
            f.jump[g] = total;
            f.jump[g].sub(&zero);
            // the ramp (1 − w)·prev + w·M: weights w², w(1 − w), (1 − w)² on the covariance
            let (rww, rw) = (&r[2][g], &r[1][g]);
            let mut o = Quad {
                a: e1[g],
                lin: [0.0; CH],
                q: *rww,
            };
            let pg = &prev[g];
            for c in 0..CH {
                let mut rp = 0.0;
                for d in 0..CH {
                    let p = pair(c, d);
                    rp += (rw[p] - rww[p]) * pg[d];
                    o.a += pg[c] * (r[0][g][p] - 2.0 * rw[p] + rww[p]) * pg[d];
                }
                o.a -= 2.0 * pg[c] * (c1[g][c] - cw[g][c]);
                o.lin[c] = -2.0 * cw[g][c] + 2.0 * rp;
            }
            f.smooth[g] = o;
        }
        (f.reuse, f.pre, f.energy) = (reuse, pre, energy);
        for (seen, e) in self.seen[k].iter_mut().zip(e1) {
            *seen = e.max(0.9 * *seen);
        }
        if s.split.is_some() {
            let rs: [[f32; SPLIT]; 15] = std::array::from_fn(|p| group(&s.sx[p]));
            let (cs, es) = (gc(&s.sc), group(&s.se));
            for g in 0..SPLIT {
                let snap: [f32; 15] = std::array::from_fn(|p| rs[p][g]);
                let before = Quad::held(es[g], &cs[g], &snap);
                f.first[g] = before;
                f.first[g].sub(&Quad::held(e0[g], &c0[g], &r0[g]));
                f.second[g] = Quad::held(e1[g], &c1[g], &r[0][g]);
                f.second[g].sub(&before);
            }
        }
    }

    /// Codes element `k` at rung `(fine, bands)` under `mode`, fitting and scoring each band
    /// of `fit` (one set per data point) and counting its bits; `pre` is error the mode
    /// fixes.
    fn code(
        &self,
        k: usize,
        mode: Mode,
        rung: (bool, u8),
        fit: [&[Quad; SPLIT]; 2],
        pre: f32,
    ) -> Coded {
        let quant = Quant { fine: rung.0 };
        let bands = rung.1;
        let nb = usize::from(NUM_BANDS[usize::from(bands)]);
        let mut c = Coded {
            mode,
            quant,
            bands,
            sparse: None,
            v: [[[0; SPLIT]; CH]; 2],
            bits: INFO_BITS
                + match mode {
                    Mode::Steep(_) => OFFSET_BITS,
                    Mode::Jump => OFFSET_BITS / 2,
                    _ => 0,
                },
            err: pre,
        };
        let points = c.points();
        let seen = &self.seen[k];
        let all: f32 = seen.iter().sum();
        for (dp, quads) in fit.iter().enumerate().take(points) {
            let mut band = Sums::default();
            let mut toward = [[0f32; SPLIT]; CH];
            let mut width = [0f32; SPLIT];
            let mut used = [0f32; SPLIT];
            for (g, q) in quads.iter().enumerate() {
                let b = atom_band(bands, g);
                band.add(b, q);
                used[b] += seen[g];
                for (t, p) in toward.iter_mut().zip(self.prev[k][g]) {
                    t[b] += p;
                }
                width[b] += 1.0;
            }
            for b in 0..SPLIT {
                let pull = used[b] > 1e-4 * all;
                for t in toward.iter_mut() {
                    t[b] = if pull { t[b] / width[b].max(1.0) } else { 0.0 };
                }
            }
            let m = band.solve(&toward, self.floor, nb);
            for ((v, m), toward) in c.v[dp].iter_mut().zip(&m).zip(&toward) {
                for b in 0..nb {
                    // within three quarters of a step of the held value, hold it
                    let held = quant.index(toward[b]);
                    let near = (m[b] / quant.step() - f32::from(held)).abs() < 0.75;
                    v[b] = if near { held } else { quant.index(m[b]) };
                }
            }
            for ch in 0..CH {
                let mut last = 0i32;
                for b in 0..nb {
                    let v = i32::from(c.v[dp][ch][b]);
                    c.bits += u32::from(quant.mtx(v - last).1);
                    last = v;
                }
            }
            let values = c.v[dp].map(|v| v.map(|v| quant.value(v)));
            for e in band.eval(&values, nb).iter().take(nb) {
                c.err += e;
            }
        }
        // one band and one channel carrying it: sparse, if no worse for fewer bits
        if nb == 1 {
            let mut sparse = c;
            sparse.bits = c.bits - (0..points).map(|dp| self.dense_bits(&c, dp)).sum::<u32>();
            sparse.err = pre;
            let mut whole = [Quad::default(); 2];
            for (w, quads) in whole.iter_mut().zip(fit).take(points) {
                for g in quads.iter() {
                    w.add(g);
                }
            }
            let mut best: Option<(u8, [i16; 2], f32)> = None;
            for keep in 0..CH {
                let (mut err, mut vs) = (0f32, [0i16; 2]);
                for (dp, q) in whole.iter().enumerate().take(points) {
                    let other = quant.value(quant.sparse() as i16);
                    let mut m = [other; CH];
                    let mut lin = q.lin[keep];
                    for d in (0..CH).filter(|d| *d != keep) {
                        lin += 2.0 * q.q[pair(keep, d)] * other;
                    }
                    let qq = q.q[pair(keep, keep)];
                    let opt = if qq > 0.0 { -lin / (2.0 * qq) } else { 0.0 };
                    vs[dp] = quant.index(opt);
                    m[keep] = quant.value(vs[dp]);
                    err += q.eval(&m);
                }
                if best.is_none_or(|b| err < b.2) {
                    best = Some((keep as u8, vs, err));
                }
            }
            if let Some((keep, vs, err)) = best {
                sparse.sparse = Some(keep);
                sparse.err += err;
                for (dp, v) in vs.iter().enumerate().take(points) {
                    sparse.v[dp] = [[quant.sparse() as i16; SPLIT]; CH];
                    sparse.v[dp][usize::from(keep)][0] = *v;
                    sparse.bits += 3 + u32::from(quant.vec(*v).1);
                }
                if sparse.err <= c.err && sparse.bits < c.bits {
                    c = sparse;
                }
            }
        }
        if mode == Mode::Smooth && self.expand(&c, 0) == self.prev[k] {
            return Coded::reuse(c.err);
        }
        c
    }

    fn dense_bits(&self, c: &Coded, dp: usize) -> u32 {
        let nb = usize::from(NUM_BANDS[usize::from(c.bands)]);
        let mut bits = 0;
        for ch in 0..CH {
            let mut last = 0i32;
            for b in 0..nb {
                let v = i32::from(c.v[dp][ch][b]);
                bits += u32::from(c.quant.mtx(v - last).1);
                last = v;
            }
        }
        bits
    }

    /// Data point `dp` of `c` as the decoder holds it per split band.
    fn expand(&self, c: &Coded, dp: usize) -> [[f32; CH]; SPLIT] {
        let mut out = [[0f32; CH]; SPLIT];
        for (g, o) in out.iter_mut().enumerate() {
            let b = if c.sparse.is_some() {
                0
            } else {
                atom_band(c.bands, g)
            };
            for (o, v) in o.iter_mut().zip(&c.v[dp]) {
                *o = c.quant.value(v[b]);
            }
        }
        out
    }

    /// Element `k` at `rung` in `mode`, from its fits.
    fn coded(&self, k: usize, mode: Mode, rung: usize) -> Coded {
        let r = RUNGS[rung.min(RUNGS.len() - 1)];
        let f = &self.fits[k];
        match mode {
            Mode::Reuse => Coded::reuse(f.reuse),
            Mode::Smooth => self.code(k, mode, r, [&f.smooth; 2], 0.0),
            Mode::Jump => self.code(k, mode, r, [&f.jump; 2], f.pre),
            Mode::Steep(_) => self.code(k, mode, r, [&f.first, &f.second], f.pre),
        }
    }

    /// Element `k` at the top rung: `Reuse` while last frame's matrix holds it 50 dB clean;
    /// else smooth; a jump at timeslot 1 if that cuts the error by a fifth; two steep points
    /// at its split if they halve it again.
    fn first(&mut self, k: usize, floor: f32) -> Coded {
        let (reuse, energy) = (self.fits[k].reuse, self.fits[k].energy);
        if reuse < 1e-5 * (energy + floor) {
            self.fits[k].family = Mode::Smooth;
            return Coded::reuse(reuse);
        }
        let smooth = self.coded(k, Mode::Smooth, 0);
        let jump = self.coded(k, Mode::Jump, 0);
        let mut best = if jump.err < 0.8 * smooth.err {
            jump
        } else {
            smooth
        };
        self.fits[k].family = if jump.err < 0.8 * smooth.err {
            Mode::Jump
        } else {
            Mode::Smooth
        };
        if let Some(b) = self.sums[k].split.filter(|b| (2..SLOTS as u8).contains(b)) {
            let steep = self.coded(k, Mode::Steep(b), 0);
            if steep.err < 0.5 * best.err {
                best = steep;
            }
        }
        best
    }

    /// The next candidate down from element `k`'s current one: a steep frame first gives
    /// up its second point, then the rungs step down, then `Reuse`.
    fn down(&self, k: usize) -> Option<Coded> {
        let family = self.fits[k].family;
        match self.coded[k].mode {
            Mode::Reuse => None,
            Mode::Steep(_) => Some(self.coded(k, family, self.rung[k])),
            _ if self.rung[k] + 1 < RUNGS.len() => Some(self.coded(k, family, self.rung[k] + 1)),
            _ => Some(Coded::reuse(self.fits[k].reuse)),
        }
    }

    /// Fits and codes every element, then writes `joc()` (clause 6.2) into `out` in at most
    /// `budget` bytes, its length returned, and clears the sums for the next frame. Over
    /// budget, the element whose next rung costs the least error relative to its own energy
    /// per bit saved steps down first.
    pub(crate) fn encode(&mut self, out: &mut [u8], budget: usize) -> Result<usize, Overflow> {
        let n = self.elements;
        let grouped = |v: &[Bands; 15]| -> [[f32; 15]; SPLIT] {
            let g: [[f32; SPLIT]; 15] = std::array::from_fn(|p| group(&v[p]));
            std::array::from_fn(|b| std::array::from_fn(|p| g[p][b]))
        };
        let r = [
            grouped(&self.x[0]),
            grouped(&self.x[1]),
            grouped(&self.x[2]),
        ];
        let r0 = grouped(&self.x0);
        let power: f32 = r[0]
            .iter()
            .map(|q| (0..CH).map(|c| q[pair(c, c)]).sum::<f32>())
            .sum();
        self.floor = 1e-6 * power / (CH * SPLIT) as f32;
        for k in 0..n {
            self.fit(k, &r, &r0);
        }
        let top = self.fits.iter().map(|f| f.energy).fold(0f32, f32::max);
        let floor = 1e-6 * top + 1e-12;
        for k in 0..n {
            self.rung[k] = 0;
            self.coded[k] = self.first(k, floor);
        }
        let limit = 8 * budget as u32;
        let total = |coded: &[Coded]| HEADER_BITS + coded.iter().map(|c| c.bits).sum::<u32>();
        if total(&self.coded).div_ceil(8) * 8 > limit {
            for k in 0..n {
                self.next[k] = self.down(k);
            }
        }
        while total(&self.coded).div_ceil(8) * 8 > limit {
            let mut pick: Option<(usize, f32)> = None;
            for k in 0..n {
                if let Some(next) = &self.next[k] {
                    let saved = self.coded[k].bits.saturating_sub(next.bits) as f32;
                    let cost =
                        (next.err - self.coded[k].err).max(0.0) / (self.fits[k].energy + floor);
                    let gain = saved / (cost + 1e-9);
                    if saved > 0.0 && pick.is_none_or(|p| gain > p.1) {
                        pick = Some((k, gain));
                    }
                }
            }
            let Some((k, _)) = pick else {
                break;
            };
            if let Some(next) = self.next[k].take() {
                if !matches!(self.coded[k].mode, Mode::Steep(_)) {
                    self.rung[k] += 1;
                }
                self.coded[k] = next;
                self.next[k] = self.down(k);
            }
        }
        if total(&self.coded).div_ceil(8) * 8 > limit {
            for c in self.coded.iter_mut() {
                *c = Coded::reuse(0.0);
            }
        }
        self.reset();
        self.write(out)
    }

    fn write(&mut self, out: &mut [u8]) -> Result<usize, Overflow> {
        let mut w = BitWriter::new(out);
        w.put(0, 3)?; // joc_dmx_config_idx: 5.X
        w.put(self.elements.saturating_sub(1) as u32, 6)?;
        w.put(0, 3)?; // joc_ext_config_idx
        w.put(4 << 5, 8)?; // joc_clipgain 1
        w.put(u32::from(self.seq), 10)?;
        for c in self.coded.iter() {
            w.put(u32::from(c.mode != Mode::Reuse), 1)?;
            if c.mode == Mode::Reuse {
                continue;
            }
            w.put(u32::from(c.bands), 3)?;
            w.put(u32::from(c.sparse.is_some()), 1)?;
            w.put(u32::from(c.quant.fine), 1)?;
            match c.mode {
                Mode::Steep(b) => {
                    w.put(0b11, 2)?; // steep, two points
                    w.put(0, 5)?; // from timeslot 1
                    w.put(u32::from(b) - 1, 5)?;
                }
                Mode::Jump => {
                    w.put(0b10, 2)?; // steep, one point
                    w.put(0, 5)?;
                }
                _ => w.put(0b00, 2)?,
            }
        }
        for c in self.coded.iter() {
            let nb = usize::from(NUM_BANDS[usize::from(c.bands)]);
            for dp in 0..c.points() {
                if let Some(keep) = c.sparse {
                    w.put(u32::from(keep), 3)?;
                    let (code, len) = c.quant.vec(c.v[dp][usize::from(keep)][0]);
                    w.put(code, u32::from(len))?;
                    continue;
                }
                for ch in 0..CH {
                    let mut last = 0i32;
                    for b in 0..nb {
                        let v = i32::from(c.v[dp][ch][b]);
                        let (code, len) = c.quant.mtx(v - last);
                        w.put(code, u32::from(len))?;
                        last = v;
                    }
                }
            }
        }
        let n = w.finish();
        for k in 0..self.elements {
            let c = self.coded[k];
            if c.mode != Mode::Reuse {
                self.prev[k] = self.expand(&c, c.points() - 1);
            }
        }
        self.seq = if self.seq == 1023 { 1 } else { self.seq + 1 };
        Ok(n)
    }

    /// Bits the payload takes with every element reused: the floor of any frame.
    pub(crate) fn min_bytes(elements: usize) -> usize {
        (HEADER_BITS as usize + elements).div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_index_the_upper_triangle() {
        for (p, &(c, d)) in PAIRS.iter().enumerate() {
            assert_eq!((pair(c, d), pair(d, c)), (p, p));
        }
    }

    #[test]
    fn the_solve_recovers_a_known_matrix() {
        // z = 1.5 x0 - 0.25 x3 over five independent channels of unequal power
        let power = [1.0f32, 0.5, 2.0, 0.25, 1.0];
        let truth = [1.5f32, 0.0, 0.0, -0.25, 0.0];
        let mut q = Quad::default();
        for c in 0..CH {
            q.q[pair(c, c)] = power[c];
            q.lin[c] = -2.0 * truth[c] * power[c];
        }
        let mut s = Sums::default();
        s.add(0, &q);
        let m = s.solve(&[[0.0; SPLIT]; CH], 0.0, 1);
        for c in 0..CH {
            assert!((m[c][0] - truth[c]).abs() < 1e-2, "{m:?}");
        }
        let quant = Quant { fine: false };
        assert_eq!(quant.index(1.5), 7); // 7 × 0.2002 = 1.401, 8 × = 1.602
        assert_eq!((quant.index(100.0), quant.index(-100.0)), (47, -48));
        assert_eq!(Quant { fine: true }.index(-0.25), -2);
    }
}
