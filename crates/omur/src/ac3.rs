//! The AC-3 frame: six blocks, the per-frame decisions that feed exponents, allocation and
//! packing, and channel coupling (§8.2.4) in `coupling`.

pub(crate) mod coupling;

use crate::bitalloc::{Alloc, MAX_SLOTS, Offsets, curve};
use crate::bits::{BitWriter, Overflow, crc_shift_back};
use crate::exponents::{D15, Groups, REUSE, encode, extract, plan};
use crate::mantissas::Mantissas;
use crate::pack::{self, Side, Stream};
use crate::tables::BITRATES;
use crate::tuner::{self, Budget};
use crate::{BLOCK_SAMPLES, Layout, mdct};
use coupling::Bands;

/// Samples a frame's six windows span: the frame and the 256 past it.
pub(crate) const SPAN: usize = 6 * BLOCK_SAMPLES + BLOCK_SAMPLES;

/// `endmant` of the LFE (§7.1.3).
const LFE_END: usize = 7;

/// The six blocks' transforms of one channel: one 512-point or a short pair each.
fn transform(pcm: &[f32; SPAN], blksw: &[bool; 6], coef: &mut [[f32; 256]; 6]) {
    for (b, out) in coef.iter_mut().enumerate() {
        let Some(x) = pcm[BLOCK_SAMPLES * b..].first_chunk::<512>() else {
            continue;
        };
        if blksw[b] {
            mdct::short_pair(x, out);
        } else {
            mdct::long(x, out);
        }
    }
}

/// One slot's frame up to the SNR offsets: raw exponents, the exponent sets and their
/// masking curves over `alloc.start..alloc.end`. A set covers the blocks up to the next
/// that sends one, with each bin's smallest exponent over them.
fn envelope(
    coef: &[[f32; 256]; 6],
    raw: &mut [[u8; 256]; 6],
    side: &mut Side,
    alloc: &mut Alloc,
    lfe: bool,
) {
    let (start, end) = (alloc.start, alloc.end);
    for (c, r) in coef.iter().zip(raw.iter_mut()) {
        extract(c, end, r);
    }
    side.strategy = plan(raw, start, end);
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
            for (e, v) in env.iter_mut().zip(r).take(end).skip(start) {
                *e = (*e).min(*v);
            }
        }
        let strategy = side.strategy[b];
        encode(
            &env,
            start,
            end,
            strategy,
            &mut side.groups[b],
            &mut alloc.exp[b],
        );
        curve(
            &alloc.exp[b],
            start,
            end,
            &mut alloc.psd[b],
            &mut alloc.mask[b],
        );
        alloc.set[b..b + len].fill(b);
        b += len;
    }
}

/// The coupling range per rate, or the bandwidth code when nothing is coupled: the
/// choices of ffmpeg's streams, as `probe dump` reads them.
fn coupling_plan(layout: Layout, kbps: u32) -> (Option<Bands>, u8) {
    let cpl = |begf, endf, strc| (Some(Bands { begf, endf, strc }), 0);
    match (layout, kbps) {
        (Layout::Stereo, ..=128) => cpl(1, 6, 0b1000_0000),
        (Layout::Stereo, ..=160) => cpl(7, 8, 0b1010),
        (Layout::Stereo, ..=192) => cpl(8, 12, 0b110_1100),
        (Layout::Stereo, ..=224) => cpl(11, 12, 0b1100),
        (Layout::Stereo, ..=256) => cpl(12, 12, 0b110),
        (Layout::Stereo, _) => (None, 48),
        (_, ..=256) => cpl(6, 5, 0),
        (_, ..=320) => cpl(8, 8, 0b100),
        (_, ..=384) => cpl(11, 10, 0),
        (_, ..=512) => cpl(12, 12, 0b110),
        _ => (None, 48),
    }
}

pub(crate) struct Ac3 {
    stream: Stream,
    /// Per channel in slot order (L C R Ls Rs, then the LFE): `SPAN` samples from the
    /// frame's first.
    pcm: Box<[[f32; SPAN]]>,
    /// Per slot: the channels', then the coupling channel's coefficients.
    coef: Box<[[[f32; 256]; 6]]>,
    raw: Box<[[[u8; 256]; 6]]>,
    side: Box<[Side]>,
    alloc: Box<[Alloc]>,
    mant: Box<Mantissas>,
    coupling: Option<Box<coupling::Frame>>,
    /// Bits through block 1 of the last frame, which crc1 must cover.
    #[cfg(test)]
    first_bits: usize,
    /// Where block 5's mantissas start in the last frame, at or past the 5/8 point.
    #[cfg(test)]
    last_start: usize,
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
        let (cpl, chbwcod) = coupling_plan(layout, kbps);
        let stream = Stream {
            acmod,
            lfe,
            nfchans,
            frmsizecod: 2 * rate as u8,
            chbwcod,
            cpl,
            frame_bytes,
            crc1_end,
            crc1_back: crc_shift_back(8 * (crc1_end - 2)),
        };
        let range = |c: usize| match cpl {
            Some(b) if c == nch => (b.start(), b.end()),
            _ if c >= nfchans => (0, LFE_END),
            Some(b) => (0, b.start()),
            None => (0, 37 + 3 * (usize::from(chbwcod) + 12)),
        };
        let slots = stream.slots();
        Ac3 {
            stream,
            pcm: vec![[0.0; SPAN]; nch].into_boxed_slice(),
            coef: vec![[[0.0; 256]; 6]; slots].into_boxed_slice(),
            raw: vec![[[24; 256]; 6]; slots].into_boxed_slice(),
            side: (0..slots)
                .map(|c| Side {
                    strategy: [REUSE; 6],
                    groups: [Groups::default(); 6],
                    blksw: [false; 6],
                    start: range(c).0,
                    end: range(c).1,
                })
                .collect(),
            alloc: (0..slots)
                .map(|c| Alloc::new(range(c).0, range(c).1))
                .collect(),
            mant: Box::new(Mantissas::new()),
            coupling: cpl.map(|b| Box::new(coupling::Frame::new(b))),
            #[cfg(test)]
            first_bits: 0,
            #[cfg(test)]
            last_start: 0,
        }
    }

    pub(crate) fn frame_bytes(&self) -> usize {
        self.stream.frame_bytes
    }

    /// The input of every channel, in slot order.
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
        let nf = s.nfchans;
        for (c, p) in self.pcm.iter().enumerate() {
            if let (Some(coef), Some(side)) = (self.coef.get_mut(c), self.side.get(c)) {
                transform(p, &side.blksw, coef);
            }
        }
        if let (Some(f), Some(cs)) = (self.coupling.as_deref_mut(), s.cpl_slot()) {
            let (fbw, rest) = self.coef.split_at_mut(nf);
            if let Some(cpl) = rest.get_mut(cs - nf) {
                coupling::form(fbw, cpl, f);
            }
        }
        let slots = s.slots();
        for c in 0..slots {
            envelope(
                &self.coef[c],
                &mut self.raw[c],
                &mut self.side[c],
                &mut self.alloc[c],
                Some(c) == s.lfe_slot(),
            );
        }

        // Side information does not depend on the offsets' values: a dry run sizes it.
        let cpl = self.coupling.as_deref();
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
            pack::block_side(&mut w, s, &self.side, cpl, &off, b, None)?;
            budget.side[b] = (w.position() - at) as u32;
        }
        let off = tuner::tune(&self.alloc, s.lfe_slot(), &budget).ok_or(Overflow)?;
        tuner::fill(&mut self.alloc, &off);
        let mant = tuner::mantissa_bits(&self.alloc, &off);
        let last_start =
            budget.header + (0..5).map(|b| budget.side[b] + mant[b]).sum::<u32>() + budget.side[5];
        let skips = pack::skips(budget.first.saturating_sub(last_start));

        let (order, n) = s.order();
        let order = &order[..n.min(MAX_SLOTS)];
        let mut w = BitWriter::new(out);
        pack::header(&mut w, s)?;
        for (b, skip) in skips.into_iter().enumerate() {
            pack::block_side(&mut w, s, &self.side, cpl, &off, b, skip)?;
            #[cfg(test)]
            if b == 5 {
                self.last_start = w.position();
            }
            self.mant.begin_block();
            for &c in order {
                let (Some(a), Some(coef)) = (self.alloc.get(c), self.coef.get(c)) else {
                    continue;
                };
                let set = a.set[b];
                self.mant
                    .channel(c, &coef[b], &a.exp[set], &a.bap[set], a.start, a.end);
            }
            pack::mantissas(&mut w, &self.mant, &self.side, order)?;
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

    /// Each channel's coupling coordinates, as oxideav decodes them, are the square root of
    /// its band energy over the coupling channel's, summed over the blocks they hold for,
    /// an eighth of it, within a step.
    #[test]
    fn coordinates_decode_to_the_band_energy_ratio() {
        let mut enc = Ac3::new(Layout::Surround51, 320);
        let mut state = audblk::Ac3State::new();
        let mut out = vec![0u8; enc.frame_bytes()];
        let mut pcm = vec![0f32; 1536 * 6];
        let mut seed = 7u32;
        for frame in 0..3 {
            for (c, p) in enc.pcm_mut().iter_mut().enumerate() {
                for v in p.iter_mut().skip(BLOCK_SAMPLES) {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *v = [0.3, 0.02, 0.1, 0.05, 0.2, 0.0][c] * ((seed >> 9) as f32 / 4e6 - 1.0);
                }
            }
            enc.encode(&mut out).unwrap();
            let si = syncinfo::parse(&out).unwrap();
            let b = bsi::parse(&out[5..]).unwrap();
            audblk::decode_frame(&mut state, &si, &b, &out, &mut pcm).unwrap();
            let f = enc.coupling.as_deref().unwrap();
            let cs = enc.stream.cpl_slot().unwrap();
            let edges = f.bands.edges();
            for c in 0..5 {
                let blk = (0..6).rev().find(|b| f.send[*b][c]).unwrap();
                let mut lo = f.bands.start();
                for (k, hi) in edges.iter().take(f.bands.count()).enumerate() {
                    let energy = |s: usize| -> f32 {
                        enc.coef[s][blk..]
                            .iter()
                            .flat_map(|b| &b[lo..*hi])
                            .map(|v| v * v)
                            .sum()
                    };
                    let want = (energy(c) / energy(cs)).sqrt() / 8.0;
                    let got = state.cpl_coord[c][k];
                    assert!(
                        (got - want).abs() <= want / 16.0,
                        "frame {frame} ch {c} band {k}: {got} for {want}"
                    );
                    lo = *hi;
                }
            }
            enc.advance();
        }
    }

    /// oxideav's decoder, fed our frames, arrives at our exponents, PSD, masking curve and
    /// `bap` values; and blocks 0 and 1 end inside crc1's 5/8.
    #[test]
    fn oxideav_rebuilds_the_allocation() {
        for (layout, kbps) in [
            (Layout::Stereo, 128),
            (Layout::Stereo, 192),
            (Layout::Stereo, 320),
            (Layout::Surround51, 256),
            (Layout::Surround51, 320),
            (Layout::Surround51, 448),
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
                    let ox = match c {
                        _ if Some(c) == enc.stream.cpl_slot() => audblk::MAX_FBW,
                        _ if c >= nf => audblk::MAX_FBW + 1,
                        _ => c,
                    };
                    let ox = &state.channels[ox];
                    let (s, start, end) = (a.set[5], a.start, a.end);
                    let what = format!("{layout:?} {kbps} slot {c}");
                    assert_eq!(ox.exp[start..end], a.exp[s][start..end], "{what}");
                    assert_eq!(ox.psd[start..end], a.psd[s][start..end], "{what}");
                    let bands = crate::tables::MASKTAB[start] as usize
                        ..crate::tables::MASKTAB[end - 1] as usize + 1;
                    assert_eq!(ox.mask[bands.clone()], a.mask[s][bands], "{what}");
                    assert_eq!(ox.bap[start..end], a.bap[s][start..end], "{what}");
                }
                enc.advance();
            }
        }
    }
}
