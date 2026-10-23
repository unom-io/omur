//! The AC-3 frame: six blocks, the per-frame decisions that feed exponents, allocation and
//! packing. Channel coupling (§8.2.4) is Stage 1b; `Coupling` is where its decision lands.

use crate::bitalloc::{Alloc, Offsets, curve};
use crate::bits::{BitWriter, Overflow, crc_shift_back};
use crate::exponents::{D15, Groups, REUSE, encode, extract, plan};
use crate::mantissas::{MAX_CH, Mantissas};
use crate::pack::{self, Coupling, Side, Stream};
use crate::tables::BITRATES;
use crate::tuner::{self, Budget};
use crate::{BLOCK_SAMPLES, Layout, mdct};

/// Samples a frame's six windows span: the frame and the 256 past it.
pub(crate) const SPAN: usize = 6 * BLOCK_SAMPLES + BLOCK_SAMPLES;

/// `endmant` of the LFE (§7.1.3).
const LFE_END: usize = 7;

/// One channel's frame up to the SNR offsets: transforms, raw exponents, the exponent
/// sets and their masking curves. A set covers the blocks up to the next that sends one,
/// with each bin's smallest exponent over them.
fn analyse(
    pcm: &[f32; SPAN],
    coef: &mut [[f32; 256]; 6],
    raw: &mut [[u8; 256]; 6],
    side: &mut Side,
    alloc: &mut Alloc,
    lfe: bool,
) {
    let end = alloc.end;
    for b in 0..6 {
        let Some(x) = pcm[BLOCK_SAMPLES * b..].first_chunk::<512>() else {
            continue;
        };
        if side.blksw[b] {
            mdct::short_pair(x, &mut coef[b]);
        } else {
            mdct::long(x, &mut coef[b]);
        }
        extract(&coef[b], end, &mut raw[b]);
    }
    side.strategy = plan(raw, end);
    if lfe {
        for st in side.strategy.iter_mut().filter(|st| **st != REUSE) {
            *st = D15;
        }
    }
    let mut b = 0;
    while b < 6 {
        let len = 1 + side.strategy[b + 1..]
            .iter()
            .take_while(|st| **st == REUSE)
            .count();
        let mut env = raw[b];
        for r in &raw[b + 1..b + len] {
            for (e, v) in env.iter_mut().zip(r).take(end) {
                *e = (*e).min(*v);
            }
        }
        let strategy = side.strategy[b];
        encode(&env, end, strategy, &mut side.groups[b], &mut alloc.exp[b]);
        curve(&alloc.exp[b], end, &mut alloc.psd[b], &mut alloc.mask[b]);
        alloc.set[b..b + len].fill(b);
        b += len;
    }
}

pub(crate) struct Ac3 {
    stream: Stream,
    /// Per channel in packing order (L C R Ls Rs, LFE last): `SPAN` samples from the
    /// frame's first.
    pcm: Box<[[f32; SPAN]]>,
    coef: Box<[[[f32; 256]; 6]]>,
    raw: Box<[[[u8; 256]; 6]]>,
    side: Box<[Side]>,
    alloc: Box<[Alloc]>,
    mant: Box<Mantissas>,
    coupling: Coupling,
    /// Bits through block 1 of the last frame, which crc1 must cover.
    #[cfg(test)]
    first_bits: usize,
    /// Where block 5's mantissas start in the last frame, at or past the 5/8 point.
    #[cfg(test)]
    last_start: usize,
}

/// `chbwcod` from the rate each full-bandwidth channel gets: 15 kHz at 64 kbit/s,
/// 19.5 kHz at 96, the full 23.7 kHz from 124 up.
// ponytail: a straight line; Stage 1b's coupling sets the band edge per rate instead.
fn bandwidth_code(kbps: u32, nfchans: usize) -> u8 {
    let per = kbps / nfchans.max(1) as u32;
    (per.saturating_sub(4) / 2).min(60) as u8
}

impl Ac3 {
    pub(crate) fn new(layout: Layout, kbps: u32) -> Ac3 {
        let (acmod, nfchans, lfe) = match layout {
            Layout::Stereo => (2, 2, false),
            _ => (7, 5, true),
        };
        let nch = nfchans + usize::from(lfe);
        let rate = BITRATES.iter().position(|r| *r == kbps).unwrap_or(0);
        let frame_bytes = 4 * BITRATES[rate] as usize;
        let words = frame_bytes / 2;
        let crc1_end = 2 * ((words >> 1) + (words >> 3));
        let chbwcod = bandwidth_code(kbps, nfchans);
        let end_of = |c: usize| {
            if c < nfchans {
                37 + 3 * (usize::from(chbwcod) + 12)
            } else {
                LFE_END
            }
        };
        Ac3 {
            stream: Stream {
                acmod,
                lfe,
                nfchans,
                frmsizecod: 2 * rate as u8,
                chbwcod,
                frame_bytes,
                crc1_end,
                crc1_back: crc_shift_back(8 * (crc1_end - 2)),
            },
            pcm: vec![[0.0; SPAN]; nch].into_boxed_slice(),
            coef: vec![[[0.0; 256]; 6]; nch].into_boxed_slice(),
            raw: vec![[[24; 256]; 6]; nch].into_boxed_slice(),
            side: (0..nch)
                .map(|c| Side {
                    strategy: [REUSE; 6],
                    groups: [Groups::default(); 6],
                    blksw: [false; 6],
                    end: end_of(c),
                })
                .collect(),
            alloc: (0..nch).map(|c| Alloc::new(end_of(c))).collect(),
            mant: Box::new(Mantissas::new()),
            coupling: Coupling::default(),
            #[cfg(test)]
            first_bits: 0,
            #[cfg(test)]
            last_start: 0,
        }
    }

    pub(crate) fn frame_bytes(&self) -> usize {
        self.stream.frame_bytes
    }

    /// The input of every channel, in packing order.
    pub(crate) fn pcm_mut(&mut self) -> &mut [[f32; SPAN]] {
        &mut self.pcm
    }

    /// Keeps the 256 samples the next frame's first window shares with this one.
    pub(crate) fn advance(&mut self) {
        for p in self.pcm.iter_mut() {
            p.copy_within(6 * BLOCK_SAMPLES.., 0);
        }
    }

    /// Encodes the frame in `pcm` into `out`, `frame_bytes` long.
    pub(crate) fn encode(&mut self, out: &mut [u8]) -> Result<(), Overflow> {
        let s = &self.stream;
        for c in 0..self.pcm.len() {
            analyse(
                &self.pcm[c],
                &mut self.coef[c],
                &mut self.raw[c],
                &mut self.side[c],
                &mut self.alloc[c],
                c >= s.nfchans,
            );
        }

        // Side information does not depend on the offsets' values: a dry run sizes it.
        let mut budget = Budget {
            header: 0,
            side: [0; 6],
            trailer: 18,
            frame: 8 * s.frame_bytes as u32,
            first: 8 * s.crc1_end as u32,
        };
        let mut w = BitWriter::new(out);
        pack::header(&mut w, s)?;
        budget.header = w.position() as u32;
        for b in 0..6 {
            let at = w.position();
            let off = Offsets::default();
            pack::block_side(&mut w, s, &self.side, self.coupling, off, b, None)?;
            budget.side[b] = (w.position() - at) as u32;
        }
        let lfe = s.lfe;
        let off = tuner::tune(&self.alloc, lfe, &budget).ok_or(Overflow)?;
        tuner::fill(&mut self.alloc, lfe, off);
        let mant = tuner::mantissa_bits(&self.alloc, lfe, off);
        let last_start =
            budget.header + (0..5).map(|b| budget.side[b] + mant[b]).sum::<u32>() + budget.side[5];
        let skips = pack::skips(budget.first.saturating_sub(last_start));

        let mut w = BitWriter::new(out);
        pack::header(&mut w, s)?;
        for (b, skip) in skips.into_iter().enumerate() {
            pack::block_side(&mut w, s, &self.side, self.coupling, off, b, skip)?;
            #[cfg(test)]
            if b == 5 {
                self.last_start = w.position();
            }
            self.mant.begin_block();
            for (c, a) in self.alloc.iter().enumerate().take(MAX_CH) {
                let set = a.set[b];
                self.mant
                    .channel(c, &self.coef[c][b], &a.exp[set], &a.bap[set], a.end);
            }
            pack::mantissas(&mut w, &self.mant, &self.side)?;
            #[cfg(test)]
            if b == 1 {
                self.first_bits = w.position();
            }
        }
        pack::close(w, s)?;
        pack::seal(out, s);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_ac3::{audblk, bsi, decoder, syncinfo};

    impl Ac3 {
        /// §5.5: blocks 0 and 1 end inside the first 5/8; block 5's mantissas start past it.
        fn assert_constraints(&self, what: &str) {
            let five_eighths = 8 * self.stream.crc1_end;
            assert!(self.first_bits <= five_eighths, "{what}: blocks 0-1");
            assert!(self.last_start >= five_eighths, "{what}: block 5");
        }
    }

    /// Silence and a tone 80 dB down leave most of the frame unused; skip fields, not
    /// auxdata, take the slack ahead of block 5.
    #[test]
    fn quiet_frames_keep_both_five_eighths_constraints() {
        for layout in [Layout::Stereo, Layout::Surround51] {
            for &kbps in crate::Config::bitrates(crate::Codec::Ac3, layout) {
                for level in [0.0, 1e-4] {
                    let mut enc = Ac3::new(layout, kbps);
                    let mut out = vec![0u8; enc.frame_bytes()];
                    for frame in 0..2 {
                        for p in enc.pcm_mut().iter_mut() {
                            for (i, v) in p.iter_mut().enumerate().skip(BLOCK_SAMPLES) {
                                let t = (frame * 1536 + i) as f32;
                                *v = level * (t * 0.05).sin();
                            }
                        }
                        enc.encode(&mut out).unwrap();
                        enc.assert_constraints(&format!("{layout:?} {kbps} level {level}"));
                        assert!(decoder::verify_packet_crc(&out).unwrap().all_ok());
                        enc.advance();
                    }
                }
            }
        }
    }

    /// oxideav's decoder, fed our frames, arrives at our exponents, PSD, masking curve and
    /// `bap` values; and blocks 0 and 1 end inside crc1's 5/8.
    #[test]
    fn oxideav_rebuilds_the_allocation() {
        for (layout, kbps) in [
            (Layout::Stereo, 128),
            (Layout::Surround51, 256),
            (Layout::Surround51, 640),
        ] {
            let mut enc = Ac3::new(layout, kbps);
            let mut state = audblk::Ac3State::new();
            let mut out = vec![0u8; enc.frame_bytes()];
            let mut pcm = vec![0f32; 1536 * enc.pcm.len()];
            let mut seed = 1u32;
            for frame in 0..4 {
                for (c, p) in enc.pcm_mut().iter_mut().enumerate() {
                    for (i, v) in p.iter_mut().enumerate().skip(BLOCK_SAMPLES) {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let t = (frame * 1536 + i) as f32;
                        let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 1.0;
                        *v = 0.3 * (t * 0.01 * (c + 1) as f32).sin() + 0.1 * noise;
                    }
                }
                enc.encode(&mut out).unwrap();
                enc.assert_constraints(&format!("{layout:?} {kbps}"));
                let si = syncinfo::parse(&out).unwrap();
                let b = bsi::parse(&out[5..]).unwrap();
                audblk::decode_frame(&mut state, &si, &b, &out, &mut pcm).unwrap();
                let nf = enc.stream.nfchans;
                for (c, a) in enc.alloc.iter().enumerate() {
                    let ox = &state.channels[if c < nf { c } else { audblk::MAX_FBW + 1 }];
                    let (s, end) = (a.set[5], a.end);
                    assert_eq!(ox.exp[..end], a.exp[s][..end], "channel {c}");
                    assert_eq!(ox.psd[..end], a.psd[s][..end], "channel {c}");
                    let bands = crate::tables::MASKTAB[end - 1] as usize + 1;
                    assert_eq!(ox.mask[..bands], a.mask[s][..bands], "channel {c}");
                    assert_eq!(ox.bap[..end], a.bap[s][..end], "channel {c}");
                }
                enc.advance();
            }
        }
    }
}
