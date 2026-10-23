//! Forward MDCT (§8.2.3): the 512-point long transform and the 256-point short pair, as a
//! TDAC fold into a DCT-IV evaluated by a half-length complex FFT with table twiddles. The
//! O(N²) spec sum stays here as the test oracle.
//!
//! An M-point DCT-IV through an M/2-point FFT, `H = M/2`:
//!
//! ```text
//! z[n]  = (u[2n] + i·u[M-1-2n]) · exp(-iπ(4n+1)/(4M))      n < H
//! Z     = FFT_H(z)
//! v[k]  = Z[k] · exp(-iπk/M)                                k < H
//! X[2k] = Re v[k],   X[M-1-2k] = -Im v[k]
//! ```
//!
//! The post-twiddle is `exp(-iπk/M)`, not the pre-twiddle's mirror: the constant
//! `exp(-iπ/(4M))` the two would share sits on the pre side only.

use crate::tables_gen::{
    FFT_IM, FFT_RE, POST_LONG_IM, POST_LONG_RE, POST_SHORT_IM, POST_SHORT_RE, PRE_LONG_IM,
    PRE_LONG_RE, PRE_SHORT_IM, PRE_SHORT_RE, WINDOW,
};

/// In-place radix-2 FFT, `X[k] = Σ x[n]·exp(-2πikn/N)`, on input already in bit-reversed
/// order. N is 64 or 128; the twiddles come from the 128-point table.
fn fft<const N: usize>(re: &mut [f32; N], im: &mut [f32; N]) {
    let mut half = 1;
    while half < N {
        let stride = 64 / half;
        let mut base = 0;
        while base < N {
            for j in 0..half {
                let (wr, wi) = (FFT_RE[j * stride], FFT_IM[j * stride]);
                let (a, b) = (base + j, base + j + half);
                let tr = wr * re[b] - wi * im[b];
                let ti = wr * im[b] + wi * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
            base += 2 * half;
        }
        half *= 2;
    }
}

/// M-point DCT-IV of `u` into `out`, `M = 2·H`, through an H-point FFT.
fn dct4<const M: usize, const H: usize>(
    u: &[f32; M],
    out: &mut [f32],
    pre: (&[f32; H], &[f32; H]),
    post: (&[f32; H], &[f32; H]),
) {
    let bits = H.trailing_zeros();
    let mut re = [0f32; H];
    let mut im = [0f32; H];
    for n in 0..H {
        let (a, b) = (u[2 * n], u[M - 1 - 2 * n]);
        let (pr, pi) = (pre.0[n], pre.1[n]);
        let slot = (n as u8).reverse_bits() as usize >> (8 - bits);
        re[slot] = a * pr - b * pi;
        im[slot] = b * pr + a * pi;
    }
    fft(&mut re, &mut im);
    for k in 0..H {
        let (qr, qi) = (post.0[k], post.1[k]);
        out[2 * k] = re[k] * qr - im[k] * qi;
        out[M - 1 - 2 * k] = -(im[k] * qr + re[k] * qi);
    }
}

/// The window at `n` of the 512-sample block.
fn window(n: usize) -> f32 {
    WINDOW[if n < 256 { n } else { 511 - n }]
}

/// The 512-point transform of `x` windowed, into 256 coefficients.
pub(crate) fn long(x: &[f32; 512], out: &mut [f32; 256]) {
    let mut w = [0f32; 512];
    for n in 0..512 {
        w[n] = x[n] * window(n);
    }
    // MDCT(a, b, c, d) = DCT-IV(-c_r - d, a - b_r) over 128-sample quarters.
    let mut u = [0f32; 256];
    for n in 0..128 {
        u[n] = -w[383 - n] - w[384 + n];
        u[128 + n] = w[n] - w[255 - n];
    }
    dct4(
        &u,
        out,
        (&PRE_LONG_RE, &PRE_LONG_IM),
        (&POST_LONG_RE, &POST_LONG_IM),
    );
}

/// The two 256-point transforms of `x` windowed (alpha -1 on the first half, +1 on the
/// second), interleaved: `out[2k]` from the first, `out[2k + 1]` from the second.
pub(crate) fn short_pair(x: &[f32; 512], out: &mut [f32; 256]) {
    let mut w = [0f32; 512];
    for n in 0..512 {
        w[n] = x[n] * window(n);
    }
    let mut u1 = [0f32; 128];
    let mut u2 = [0f32; 128];
    for m in 0..128 {
        u1[m] = w[m] - w[255 - m];
        u2[m] = -(w[383 - m] + w[384 + m]);
    }
    let mut x1 = [0f32; 128];
    let mut x2 = [0f32; 128];
    let pre = (&PRE_SHORT_RE, &PRE_SHORT_IM);
    let post = (&POST_SHORT_RE, &POST_SHORT_IM);
    dct4(&u1, &mut x1, pre, post);
    dct4(&u2, &mut x2, pre, post);
    for k in 0..128 {
        out[2 * k] = x1[k];
        out[2 * k + 1] = x2[k];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §8.2.3.2 summed directly, in f64: N-point transform of the windowed `x` with `alpha`.
    fn spec_sum(x: &[f64], alpha: f64) -> Vec<f64> {
        let n = x.len() as f64;
        let pi = std::f64::consts::PI;
        (0..x.len() / 2)
            .map(|k| {
                let k1 = (2 * k + 1) as f64;
                let s: f64 = x
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let t = 2.0 * pi / (4.0 * n) * (2 * i + 1) as f64 * k1
                            + pi / 4.0 * k1 * (1.0 + alpha);
                        v * t.cos()
                    })
                    .sum();
                -2.0 / n * s
            })
            .collect()
    }

    fn signals() -> Vec<[f32; 512]> {
        let mut seed = 0x2545_f491u32;
        let mut noise = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let mut impulse = [0f32; 512];
        impulse[173] = 1.0;
        let tone = core::array::from_fn(|i| (i as f32 * 0.37).sin() * 0.5 + 0.2);
        let white = core::array::from_fn(|_| noise());
        vec![impulse, tone, white, [1.0; 512]]
    }

    fn assert_close(got: &[f32], want: &[f64]) {
        let peak = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(want)
            .fold(0f64, |m, (g, w)| m.max((*g as f64 - w).abs()));
        assert!(
            err <= 1e-4 * peak,
            "max error {err:e} against peak {peak:e}"
        );
    }

    #[test]
    fn long_matches_the_spec_sum() {
        for x in signals() {
            let windowed: Vec<f64> = (0..512).map(|n| (x[n] * window(n)) as f64).collect();
            let mut out = [0f32; 256];
            long(&x, &mut out);
            assert_close(&out, &spec_sum(&windowed, 0.0));
        }
    }

    #[test]
    fn short_pair_matches_the_spec_sum() {
        for x in signals() {
            let windowed: Vec<f64> = (0..512).map(|n| (x[n] * window(n)) as f64).collect();
            let first = spec_sum(&windowed[..256], -1.0);
            let second = spec_sum(&windowed[256..], 1.0);
            let want: Vec<f64> = (0..256)
                .map(|i| {
                    if i % 2 == 0 {
                        first[i / 2]
                    } else {
                        second[i / 2]
                    }
                })
                .collect();
            let mut out = [0f32; 256];
            short_pair(&x, &mut out);
            assert_close(&out, &want);
        }
    }

    #[test]
    fn window_is_power_complementary_and_matches_table_7_33() {
        for n in 0..256 {
            let (a, b) = (window(n), window(n + 256));
            assert!((a * a + b * b - 1.0).abs() < 1e-6, "n {n}");
        }
        // Table 7.33 prints five decimals; the full-precision window rounds to them.
        for (n, want) in SPEC_WINDOW.iter().enumerate() {
            assert!((WINDOW[n] - want).abs() <= 0.5e-5 + 1e-7, "n {n}");
        }
        for (ours, theirs) in WINDOW.iter().zip(oxideav_ac3::tables::WINDOW) {
            assert!((ours - theirs).abs() < 1e-5);
        }
    }

    /// Forward here, oxideav's IMDCT + window + the decoder's 2× overlap-add: the input.
    #[test]
    fn decoder_overlap_add_reproduces_the_input() {
        let len = 256 * 12;
        let pcm: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.05).sin() * 0.4 + (i as f32 * 0.61).sin() * 0.2)
            .collect();
        for short in [false, true] {
            let mut delay = [0f32; 256];
            let mut worst = 0f32;
            for b in 0..len / 256 - 1 {
                let x: [f32; 512] = core::array::from_fn(|n| pcm[256 * b + n]);
                let mut coef = [0f32; 256];
                let mut y = [0f32; 512];
                if short {
                    short_pair(&x, &mut coef);
                    oxideav_ac3::imdct::imdct_256_pair_fft(&coef, &mut y);
                } else {
                    long(&x, &mut coef);
                    oxideav_ac3::imdct::imdct_512_fft(&coef, &mut y);
                }
                for (n, v) in y.iter_mut().enumerate() {
                    *v *= window(n);
                }
                if b > 0 {
                    for n in 0..256 {
                        let out = 2.0 * (y[n] + delay[n]);
                        worst = worst.max((out - pcm[256 * b + n]).abs());
                    }
                }
                delay.copy_from_slice(&y[256..]);
            }
            assert!(worst < 1e-4, "short {short}: worst {worst}");
        }
    }

    #[rustfmt::skip]
    #[allow(clippy::approx_constant)]
    const SPEC_WINDOW: [f32; 256] = [
        0.00014, 0.00024, 0.00037, 0.00051, 0.00067, 0.00086, 0.00107, 0.00130,
        0.00157, 0.00187, 0.00220, 0.00256, 0.00297, 0.00341, 0.00390, 0.00443,
        0.00501, 0.00564, 0.00632, 0.00706, 0.00785, 0.00871, 0.00962, 0.01061,
        0.01166, 0.01279, 0.01399, 0.01526, 0.01662, 0.01806, 0.01959, 0.02121,
        0.02292, 0.02472, 0.02662, 0.02863, 0.03073, 0.03294, 0.03527, 0.03770,
        0.04025, 0.04292, 0.04571, 0.04862, 0.05165, 0.05481, 0.05810, 0.06153,
        0.06508, 0.06878, 0.07261, 0.07658, 0.08069, 0.08495, 0.08935, 0.09389,
        0.09859, 0.10343, 0.10842, 0.11356, 0.11885, 0.12429, 0.12988, 0.13563,
        0.14152, 0.14757, 0.15376, 0.16011, 0.16661, 0.17325, 0.18005, 0.18699,
        0.19407, 0.20130, 0.20867, 0.21618, 0.22382, 0.23161, 0.23952, 0.24757,
        0.25574, 0.26404, 0.27246, 0.28100, 0.28965, 0.29841, 0.30729, 0.31626,
        0.32533, 0.33450, 0.34376, 0.35311, 0.36253, 0.37204, 0.38161, 0.39126,
        0.40096, 0.41072, 0.42054, 0.43040, 0.44030, 0.45023, 0.46020, 0.47019,
        0.48020, 0.49022, 0.50025, 0.51028, 0.52031, 0.53033, 0.54033, 0.55031,
        0.56026, 0.57019, 0.58007, 0.58991, 0.59970, 0.60944, 0.61912, 0.62873,
        0.63827, 0.64774, 0.65713, 0.66643, 0.67564, 0.68476, 0.69377, 0.70269,
        0.71150, 0.72019, 0.72877, 0.73723, 0.74557, 0.75378, 0.76186, 0.76981,
        0.77762, 0.78530, 0.79283, 0.80022, 0.80747, 0.81457, 0.82151, 0.82831,
        0.83496, 0.84145, 0.84779, 0.85398, 0.86001, 0.86588, 0.87160, 0.87716,
        0.88257, 0.88782, 0.89291, 0.89785, 0.90264, 0.90728, 0.91176, 0.91610,
        0.92028, 0.92432, 0.92822, 0.93197, 0.93558, 0.93906, 0.94240, 0.94560,
        0.94867, 0.95162, 0.95444, 0.95713, 0.95971, 0.96217, 0.96451, 0.96674,
        0.96887, 0.97089, 0.97281, 0.97463, 0.97635, 0.97799, 0.97953, 0.98099,
        0.98236, 0.98366, 0.98488, 0.98602, 0.98710, 0.98811, 0.98905, 0.98994,
        0.99076, 0.99153, 0.99225, 0.99291, 0.99353, 0.99411, 0.99464, 0.99513,
        0.99558, 0.99600, 0.99639, 0.99674, 0.99706, 0.99736, 0.99763, 0.99788,
        0.99811, 0.99831, 0.99850, 0.99867, 0.99882, 0.99895, 0.99908, 0.99919,
        0.99929, 0.99938, 0.99946, 0.99953, 0.99959, 0.99965, 0.99969, 0.99974,
        0.99978, 0.99981, 0.99984, 0.99986, 0.99988, 0.99990, 0.99992, 0.99993,
        0.99994, 0.99995, 0.99996, 0.99997, 0.99998, 0.99998, 0.99998, 0.99999,
        0.99999, 0.99999, 0.99999, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000,
        1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000,
    ];
}
