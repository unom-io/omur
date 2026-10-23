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

/// Channels a filter vector runs side by side; 3/2 takes two, the fifth channel's idle
/// lanes filtering silence.
const LANES: usize = 4;
const GROUPS: usize = MAX_FBW.div_ceil(LANES);

type Lanes = [f32; LANES];

/// The detector's state across segments and frames, per full-bandwidth channel.
#[derive(Clone, Copy, Default)]
pub(crate) struct Detector {
    /// Direct form II state of each group's biquads: w[n-1], then w[n-2], per lane.
    state: [[[Lanes; 2]; 2]; GROUPS],
    /// The last piece's peak of each level, from the segment before.
    last: [[f32; 3]; MAX_FBW],
}

/// `HIGHPASS` as its two biquads.
const STAGES: [[f32; 5]; 2] = {
    let mut t = [[0f32; 5]; 2];
    let mut i = 0;
    while i < 10 {
        t[i / 5][i % 5] = HIGHPASS[i];
        i += 1;
    }
    t
};

/// One sample of every lane through biquad `h` in direct form II, state `s`.
#[inline(always)]
fn biquad(v: Lanes, s: &mut [Lanes; 2], h: &[f32; 5]) -> Lanes {
    let [s1, s2] = *s;
    let w: Lanes = core::array::from_fn(|c| v[c] - h[3] * s1[c] - h[4] * s2[c]);
    *s = [w, s1];
    core::array::from_fn(|c| h[0] * w[c] + h[1] * s1[c] + h[2] * s2[c])
}

/// The input of a channel the layout lacks.
static SILENT: [f32; SPAN] = [0.0; SPAN];

/// The high-passed peak of every piece of the first `n` blocks, per lane of `G` groups:
/// the groups run in one loop, so their recursions overlap.
fn filter<const G: usize>(
    rows: &[[&[f32; SPAN]; LANES]; G],
    state: &mut [[[Lanes; 2]; 2]; G],
    peak: &mut [[[Lanes; GROUPS]; 4]; 6],
    n: usize,
) {
    for (b, pb) in peak.iter_mut().enumerate().take(n) {
        for (q, pk) in pb.iter_mut().enumerate() {
            let at = BLOCK_SAMPLES * (b + 1) + 64 * q;
            let piece: [[&[f32]; LANES]; G] = core::array::from_fn(|g| {
                rows[g].map(|r| r.get(at..at + 64).unwrap_or(&SILENT[..64]))
            });
            let mut top = [[0f32; LANES]; G];
            for i in 0..64 {
                for g in 0..G {
                    let mut v: Lanes =
                        core::array::from_fn(|c| piece[g][c].get(i).copied().unwrap_or(0.0));
                    for (sj, h) in state[g].iter_mut().zip(&STAGES) {
                        v = biquad(v, sj, h);
                    }
                    top[g] = core::array::from_fn(|c| top[g][c].max(v[c].abs()));
                }
            }
            pk[..G].copy_from_slice(&top);
        }
    }
}

impl Detector {
    /// `blksw` of the frame's first `blocks` blocks for each of `pcm`'s channels: whether the
    /// second half of the block's window, the 256 samples the frame brings in after it, holds
    /// a transient. The channels run four to a vector, so their filters do not wait on each
    /// other.
    pub(crate) fn detect(&mut self, pcm: &[[f32; SPAN]], blksw: &mut [[bool; 6]], blocks: usize) {
        let n = blocks.min(6);
        let pcm = pcm.get(..MAX_FBW).unwrap_or(pcm);
        let nch = pcm.len();
        let rows: [[&[f32; SPAN]; LANES]; GROUPS] = core::array::from_fn(|g| {
            core::array::from_fn(|c| pcm.get(LANES * g + c).unwrap_or(&SILENT))
        });
        let mut peak = [[[[0f32; LANES]; GROUPS]; 4]; 6];
        if nch > LANES {
            filter(&rows, &mut self.state, &mut peak, n);
        } else {
            let ([first, ..], [state, ..]) = (&rows, &mut self.state);
            filter::<1>(
                core::array::from_ref(first),
                core::array::from_mut(state),
                &mut peak,
                n,
            );
        }
        for (b, pb) in peak.iter().enumerate().take(n) {
            for (c, sw) in blksw.iter_mut().enumerate().take(nch) {
                let p3 = pb.map(|p| p[c / LANES][c % LANES]);
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
            d.detect(&pcm, &mut sw, 6);
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
