//! The 64-band complex QMF analysis bank of TS 103 420 clause 7.2, run on every element and
//! on the rendered core: 64 samples in, one timeslot of 64 complex subband samples out.
//!
//! Pseudocode 12's modulation `Q[k] = Σ u[j]·exp(iπ(k+½)(j−½)/64)` over the 128 folded
//! samples `u` is two 64-point DCT-IVs, `X[k] = Σ x[p]·cos(π(p+½)(k+½)/64)`:
//!
//! ```text
//! x[0] = u[0] + u[1],   x[p] = u[p+1] − u[128−p]      p in 1..64
//! y[0] = u[1] − u[0],   y[p] = u[p+1] + u[128−p]
//! Re Q[k] = DCT-IV(x)[k],   Im Q[k] = (−1)^k · DCT-IV(y reversed)[k]
//! ```
//!
//! Each DCT-IV runs through a 32-point FFT with table twiddles, as `mdct` does.

use super::tables::{FFT_IM, FFT_RE, POST_IM, POST_RE, PRE_IM, PRE_RE, QWIN};

pub(crate) const SUBBANDS: usize = 64;
const TAPS: usize = 640;

/// The twiddles of FFT stages 4, 8 and 16 side by side: stage `h` reads `exp(-iπj/h)` for
/// `j < h` at `h + j`.
const STAGE_RE: [f32; 32] = stage(&FFT_RE);
const STAGE_IM: [f32; 32] = stage(&FFT_IM);

const fn stage(t: &[f32; 16]) -> [f32; 32] {
    let mut out = [0f32; 32];
    let mut h = 4;
    while h < 32 {
        let mut j = 0;
        while j < h {
            out[h + j] = t[j * (16 / h)];
            j += 1;
        }
        h *= 2;
    }
    out
}

/// One radix-2 stage over butterflies `H` apart.
#[inline(always)]
fn butterflies<const H: usize>(re: &mut [f32; 32], im: &mut [f32; 32]) {
    let (wr, wi) = (&STAGE_RE[H..2 * H], &STAGE_IM[H..2 * H]);
    for (re, im) in re.chunks_exact_mut(2 * H).zip(im.chunks_exact_mut(2 * H)) {
        let (ra, rb) = re.split_at_mut(H);
        let (ia, ib) = im.split_at_mut(H);
        for (((ra, rb), (ia, ib)), (wr, wi)) in ra
            .iter_mut()
            .zip(rb.iter_mut())
            .zip(ia.iter_mut().zip(ib.iter_mut()))
            .zip(wr.iter().zip(wi))
        {
            let tr = wr * *rb - wi * *ib;
            let ti = wr * *ib + wi * *rb;
            *rb = *ra - tr;
            *ib = *ia - ti;
            *ra += tr;
            *ia += ti;
        }
    }
}

/// Radix-2 FFT, `X[k] = Σ x[n]·exp(-2πikn/32)`, on input in bit-reversed order. The
/// first two stages' twiddles are 1 and −i, so they only add.
fn fft32(re: &mut [f32; 32], im: &mut [f32; 32]) {
    for (re, im) in re.chunks_exact_mut(4).zip(im.chunks_exact_mut(4)) {
        let (r0, r1, r2, r3) = (re[0], re[1], re[2], re[3]);
        let (i0, i1, i2, i3) = (im[0], im[1], im[2], im[3]);
        // stage 1: (0, 1) and (2, 3)
        let (a0, a1, a2, a3) = (r0 + r1, r0 - r1, r2 + r3, r2 - r3);
        let (c0, c1, c2, c3) = (i0 + i1, i0 - i1, i2 + i3, i2 - i3);
        // stage 2: (0, 2) by 1, (1, 3) by −i
        re[0] = a0 + a2;
        im[0] = c0 + c2;
        re[2] = a0 - a2;
        im[2] = c0 - c2;
        re[1] = a1 + c3;
        im[1] = c1 - a3;
        re[3] = a1 - c3;
        im[3] = c1 + a3;
    }
    butterflies::<4>(re, im);
    butterflies::<8>(re, im);
    butterflies::<16>(re, im);
}

/// The 64-point DCT-IV of `u`, unscaled: `z[n] = (u[2n] + i·u[63−2n])·exp(−iπ(4n+1)/256)`,
/// `Z = FFT₃₂(z)`, `v[k] = Z[k]·exp(−iπk/64)`, `X[2k] = Re v[k]`, `X[63−2k] = −Im v[k]`.
pub(crate) fn dct4(u: &[f32; 64], out: &mut [f32; 64]) {
    let mut zr = [0f32; 32];
    let mut zi = [0f32; 32];
    for n in 0..32 {
        let (a, b) = (u[2 * n], u[63 - 2 * n]);
        zr[n] = a * PRE_RE[n] - b * PRE_IM[n];
        zi[n] = b * PRE_RE[n] + a * PRE_IM[n];
    }
    let mut re = [0f32; 32];
    let mut im = [0f32; 32];
    for (n, (r, i)) in zr.iter().zip(zi).enumerate() {
        let slot = (n as u8).reverse_bits() as usize >> 3;
        (re[slot], im[slot]) = (*r, i);
    }
    fft32(&mut re, &mut im);
    // out[63 − 2k] is out[2m + 1] for m = 31 − k
    for k in 0..32 {
        let m = 31 - k;
        out[2 * k] = re[k] * POST_RE[k] - im[k] * POST_IM[k];
        out[2 * k + 1] = -(im[m] * POST_RE[m] + re[m] * POST_IM[m]);
    }
}

/// One signal's analysis filter state (`qana_filt`), the newest sample first: a ring of
/// `TAPS` held twice over, so the window always reads `TAPS` in a row from `head`.
pub(crate) struct Analysis {
    state: [f32; 2 * TAPS],
    head: usize,
    /// Timeslots in a row whose input was silent; at 10, the state is all zeros.
    quiet: u8,
}

impl Analysis {
    pub(crate) fn new() -> Analysis {
        Analysis {
            state: [0.0; 2 * TAPS],
            head: 0,
            quiet: 10,
        }
    }

    /// Whether the last timeslot came out all zeros.
    pub(crate) fn silent(&self) -> bool {
        self.quiet >= 10
    }

    /// One timeslot from 64 chronological samples (Pseudocode 8 to 12) into `re`, `im`.
    pub(crate) fn process(&mut self, pcm: &[f32; 64], re: &mut [f32; 64], im: &mut [f32; 64]) {
        if pcm.iter().all(|s| *s == 0.0) {
            if self.quiet >= 10 {
                re.fill(0.0);
                im.fill(0.0);
                return;
            }
            self.quiet += 1;
        } else {
            self.quiet = 0;
        }
        self.head = (self.head + TAPS - 64) % TAPS;
        let h = self.head;
        for (i, src) in pcm.iter().rev().enumerate() {
            self.state[h + i] = *src;
            self.state[h + TAPS + i] = *src;
        }
        let state = &self.state[h..h + TAPS];
        let mut u = [0f32; 128];
        for k in 0..TAPS / 128 {
            let (s, w) = (
                &state[128 * k..128 * k + 128],
                &QWIN[128 * k..128 * k + 128],
            );
            for j in 0..128 {
                u[j] += s[j] * w[j];
            }
        }
        let mut x = [0f32; 64];
        let mut y = [0f32; 64];
        x[0] = u[0] + u[1];
        y[63] = u[1] - u[0];
        for p in 1..64 {
            x[p] = u[p + 1] - u[128 - p];
            y[63 - p] = u[p + 1] + u[128 - p];
        }
        dct4(&x, re);
        dct4(&y, im);
        for v in im.iter_mut().skip(1).step_by(2) {
            *v = -*v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pseudocode 8 to 12 evaluated directly in f64, one timeslot at a time.
    struct Direct {
        state: Vec<f64>,
    }

    impl Direct {
        fn process(&mut self, pcm: &[f64]) -> Vec<(f64, f64)> {
            self.state.rotate_right(64);
            for j in 0..64 {
                self.state[j] = pcm[63 - j];
            }
            let z: Vec<f64> = (0..TAPS)
                .map(|j| self.state[j] * f64::from(QWIN[j]))
                .collect();
            let u: Vec<f64> = (0..128)
                .map(|j| (0..5).map(|k| z[j + 128 * k]).sum())
                .collect();
            (0..64)
                .map(|sb| {
                    let (mut r, mut i) = (0.0, 0.0);
                    for (j, v) in u.iter().enumerate() {
                        let t = std::f64::consts::PI * (sb as f64 + 0.5) * (j as f64 - 0.5) / 64.0;
                        r += v * t.cos();
                        i += v * t.sin();
                    }
                    (r, i)
                })
                .collect()
        }
    }

    #[test]
    fn the_fast_bank_matches_the_direct_sum_on_a_tone() {
        let mut fast = Analysis::new();
        let mut direct = Direct {
            state: vec![0.0; TAPS],
        };
        let tone = |n: usize| {
            let t = n as f64 / 48_000.0;
            0.5 * (2.0 * std::f64::consts::PI * 1_000.0 * t).sin()
                + 0.25 * (2.0 * std::f64::consts::PI * 9_100.0 * t).cos()
        };
        let (mut err, mut peak) = (0f64, 0f64);
        for slot in 0..20 {
            let pcm: Vec<f64> = (0..64).map(|i| tone(64 * slot + i)).collect();
            let pcm32: [f32; 64] = std::array::from_fn(|i| pcm[i] as f32);
            let (mut re, mut im) = ([0f32; 64], [0f32; 64]);
            fast.process(&pcm32, &mut re, &mut im);
            for (sb, (r, i)) in direct.process(&pcm).into_iter().enumerate() {
                err = err.max(
                    (f64::from(re[sb]) - r)
                        .abs()
                        .max((f64::from(im[sb]) - i).abs()),
                );
                peak = peak.max(r.hypot(i));
            }
        }
        assert!(err < 1e-5 * peak, "error {err:e} against peak {peak:e}");
        // 1 kHz is subband 2 (375 Hz each), 9.1 kHz subband 24
        let (mut re, mut im) = ([0f32; 64], [0f32; 64]);
        fast.process(
            &std::array::from_fn(|i| tone(1280 + i) as f32),
            &mut re,
            &mut im,
        );
        let mag: Vec<f32> = (0..64).map(|k| re[k].hypot(im[k])).collect();
        let top = mag.iter().cloned().fold(0f32, f32::max);
        assert!(mag[2] == top || mag[24] == top);
        assert!(mag[40] < 1e-3 * top);
    }

    #[test]
    fn silence_costs_nothing_once_the_state_drains() {
        let mut a = Analysis::new();
        let (mut re, mut im) = ([1f32; 64], [1f32; 64]);
        a.process(&[0.0; 64], &mut re, &mut im);
        assert!(a.silent() && re.iter().chain(&im).all(|v| *v == 0.0));
        a.process(&[0.5; 64], &mut re, &mut im);
        assert!(!a.silent());
        for _ in 0..10 {
            a.process(&[0.0; 64], &mut re, &mut im);
        }
        assert!(a.silent() && re.iter().chain(&im).all(|v| v.abs() < 1e-9));
    }
}
