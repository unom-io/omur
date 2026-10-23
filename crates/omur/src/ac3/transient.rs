//! Transient detection (§8.2.2): each full-bandwidth channel, high-passed at 8 kHz, is cut
//! into 256-sample segments, each a tree of one 256, two 128 and four 64-sample pieces. A
//! piece whose peak jumps past its predecessor's by the level's ratio marks a transient,
//! and the block whose second half the segment is switches to the short transform pair.

use super::SPAN;
use super::coupling::MAX_FBW;
use crate::BLOCK_SAMPLES;
use crate::tables_gen::HIGHPASS;

/// A piece of level 1, 2, 3 holds a transient when its predecessor's peak is under this
/// fraction of its own (§8.2.2 step 4).
const RATIO: [f32; 3] = [0.1, 0.075, 0.05];

/// Under this peak a segment is silent and never switches.
const SILENCE: f32 = 100.0 / 32768.0;

/// The detector's state across segments and frames, per full-bandwidth channel.
#[derive(Clone, Copy, Default)]
pub(crate) struct Detector {
    /// Direct form II state of each biquad: w[n-1], then w[n-2], per channel.
    state: [[[f32; MAX_FBW]; 2]; 2],
    /// The last piece's peak of each level, from the segment before.
    last: [[f32; 3]; MAX_FBW],
}

/// The input of a channel the layout lacks.
static SILENT: [f32; SPAN] = [0.0; SPAN];

impl Detector {
    /// `blksw` of the frame's six blocks for each of `pcm`'s channels: whether the second
    /// half of the block's window, the 256 samples the frame brings in after it, holds a
    /// transient. The channels run side by side, so their filters do not wait on each other.
    pub(crate) fn detect(&mut self, pcm: &[[f32; SPAN]], blksw: &mut [[bool; 6]]) {
        let rows: [&[f32; SPAN]; MAX_FBW] = core::array::from_fn(|c| pcm.get(c).unwrap_or(&SILENT));
        let [mut s1, mut s2] = [self.state[0], self.state[1]];
        let k: [&[f32]; 2] = [&HIGHPASS[..5], &HIGHPASS[5..]];
        for b in 0..6 {
            let mut peak = [[0f32; MAX_FBW]; 4];
            for (q, pk) in peak.iter_mut().enumerate() {
                let at = BLOCK_SAMPLES * (b + 1) + 64 * q;
                for n in at..at + 64 {
                    let mut v: [f32; MAX_FBW] =
                        core::array::from_fn(|c| rows[c].get(n).copied().unwrap_or(0.0));
                    for (j, h) in k.iter().enumerate() {
                        for c in 0..MAX_FBW {
                            let w = v[c] - h[3] * s1[j][c] - h[4] * s2[j][c];
                            v[c] = h[0] * w + h[1] * s1[j][c] + h[2] * s2[j][c];
                            s2[j][c] = s1[j][c];
                            s1[j][c] = w;
                        }
                    }
                    for (p, v) in pk.iter_mut().zip(v) {
                        *p = p.max(v.abs());
                    }
                }
            }
            for (c, sw) in blksw.iter_mut().enumerate().take(pcm.len().min(MAX_FBW)) {
                let p3 = [peak[0][c], peak[1][c], peak[2][c], peak[3][c]];
                let p2 = [p3[0].max(p3[1]), p3[2].max(p3[3])];
                let p1 = p2[0].max(p2[1]);
                let last = self.last[c];
                let jump = |level: usize, before: f32, now: f32| now * RATIO[level] > before;
                sw[b] = p1 >= SILENCE
                    && (jump(0, last[0], p1)
                        || jump(1, last[1], p2[0])
                        || jump(1, p2[0], p2[1])
                        || jump(2, last[2], p3[0])
                        || p3.windows(2).any(|w| jump(2, w[0], w[1])));
                self.last[c] = [p1, p2[1], p3[3]];
            }
        }
        self.state = [s1, s2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(x: &[f32]) -> Vec<bool> {
        let mut d = Detector::default();
        let mut out = Vec::new();
        let mut pcm = vec![[0f32; SPAN]; 1];
        for frame in x.chunks(1536) {
            pcm[0].copy_within(1536.., 0);
            pcm[0][256..256 + frame.len()].copy_from_slice(frame);
            let mut sw = [[false; 6]];
            d.detect(&pcm, &mut sw);
            out.extend(sw[0]);
        }
        out
    }

    #[test]
    fn an_impulse_switches_its_block_only() {
        let mut x = vec![0f32; 4 * 1536];
        x[1536 + 3 * 256 + 100] = 0.5;
        let sw = run(&x);
        let hits: Vec<usize> = (0..sw.len()).filter(|b| sw[*b]).collect();
        assert_eq!(hits, [6 + 3]);
    }

    #[test]
    fn a_steady_tone_never_switches() {
        for f in [0.01f32, 0.3, 1.9, 2.9] {
            let x: Vec<f32> = (0..8 * 1536).map(|i| 0.7 * (i as f32 * f).sin()).collect();
            let sw = run(&x);
            // the first segment rises out of silence
            assert!(sw[1..].iter().all(|s| !s), "{f}: {sw:?}");
        }
    }
}
