//! The coded frame: its blocks (six on AC-3; one, two, three or six on E-AC-3), the
//! per-frame decisions that feed exponents, allocation and packing, and channel coupling
//! (§8.2.4) in `coupling`. Every decision restarts at the frame's block 0.

pub(crate) mod coupling;
mod transient;

use crate::bitalloc::{Alloc, MAX_SLOTS, Offsets, curve};
use crate::bits::{BitWriter, Overflow};
use crate::exponents::{D15, D45, Groups, REUSE, encode, extract, group_count, plan};
use crate::mantissas::Mantissas;
use crate::pack::{self, Choices, Side, Stream, Syntax};
use crate::tuner::{self, Budget};
use crate::{BLOCK_SAMPLES, Layout, eac3, mdct};
use coupling::Bands;

/// Whether transients switch blocks to the short transform pair (§8.2.2). A build-time
/// choice for the listening pass, which decides the default.
const BLOCK_SWITCHING: bool = true;

/// Samples the longest frame's windows span: six blocks and the 256 past them. A frame of
/// `n` blocks uses the first `256·n + 256`.
pub(crate) const SPAN: usize = 6 * BLOCK_SAMPLES + BLOCK_SAMPLES;

/// `endmant` of the LFE (§7.1.3).
const LFE_END: usize = 7;

/// Bytes one skip field holds (`skipl`).
const SKIP_BYTES: usize = 511;

/// Block `b`'s share of `ext`: its next `SKIP_BYTES`, or nothing past the end.
fn skip_of(ext: &[u8], b: usize) -> Option<&[u8]> {
    let chunk = ext.get(SKIP_BYTES * b..ext.len().min(SKIP_BYTES * (b + 1)))?;
    (!chunk.is_empty()).then_some(chunk)
}

/// The first `blocks` blocks' transforms of one channel: one 512-point or a short pair each.
fn transform(pcm: &[f32; SPAN], blksw: &[bool; 6], coef: &mut [[f32; 256]; 6], blocks: usize) {
    for (b, out) in coef.iter_mut().enumerate().take(blocks) {
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
/// masking curves over `alloc.start..alloc.end`, for the frame's `alloc.blocks` blocks. A
/// set covers the blocks up to the next that sends one, with each bin's smallest exponent
/// over them; `fixed` is the bits a set sends besides its groups.
fn envelope(
    coef: &[[f32; 256]; 6],
    raw: &mut [[u8; 256]; 6],
    side: &mut Side,
    alloc: &mut Alloc,
    fixed: u32,
    lfe: bool,
) {
    let (start, end, n) = (alloc.start, alloc.end, alloc.blocks);
    for (c, r) in coef.iter().zip(raw.iter_mut()).take(n) {
        extract(c, end, r);
    }
    side.strategy = plan(raw, start, end, fixed, lfe, n);
    let mut b = 0;
    while b < n {
        let len = 1 + side.strategy[b + 1..n]
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

/// Rematrixes a 2/0 frame (§8.2.6): in each of the first `bands` rematrixing bands below
/// `top`, the frame's L and R become (L + R)/2 and (L − R)/2 where that pair codes for
/// less: noise at N in L and R needs N/2 in both of the pair, so the pair wins when four
/// times the product of its energies is under the product of L's and R's. `l` and `r` hold
/// the frame's blocks. Returns the flags.
fn rematrix(l: &mut [[f32; 256]], r: &mut [[f32; 256]], bands: usize, top: usize) -> [bool; 4] {
    const EDGES: [usize; 5] = [13, 25, 37, 61, 253];
    let mut flags = [false; 4];
    for (k, flag) in flags.iter_mut().enumerate().take(bands) {
        let (lo, hi) = (EDGES[k], EDGES[k + 1].min(top));
        if lo >= hi {
            continue;
        }
        let mut e = [0f32; 4];
        for (lb, rb) in l.iter().zip(r.iter()) {
            for (a, b) in lb[lo..hi].iter().zip(&rb[lo..hi]) {
                let (s, d) = (0.5 * (a + b), 0.5 * (a - b));
                e[0] += a * a;
                e[1] += b * b;
                e[2] += s * s;
                e[3] += d * d;
            }
        }
        *flag = 4.0 * e[2] * e[3] < e[0] * e[1];
        if *flag {
            for (lb, rb) in l.iter_mut().zip(r.iter_mut()) {
                for (a, b) in lb[lo..hi].iter_mut().zip(&mut rb[lo..hi]) {
                    (*a, *b) = (0.5 * (*a + *b), 0.5 * (*a - *b));
                }
            }
        }
    }
    flags
}

/// The coupling range per rate, or the bandwidth code when nothing is coupled: the
/// choices of ffmpeg's 2.0 and 5.1 streams, as `probe dump` reads them. Other channel
/// counts take the 5.1 range of the same rate per channel.
fn coupling_plan(nfchans: usize, kbps: u32) -> (Option<Bands>, u8) {
    let cpl = |begf, endf, strc| (Some(Bands { begf, endf, strc }), 0);
    if nfchans == 2 {
        return match kbps {
            0..=128 => cpl(1, 6, 0b1000_0000),
            129..=160 => cpl(7, 8, 0b1010),
            161..=192 => cpl(8, 12, 0b110_1100),
            193..=224 => cpl(11, 12, 0b1100),
            225..=256 => cpl(12, 12, 0b110),
            _ => (None, 48),
        };
    }
    match kbps * 5 / nfchans.max(1) as u32 {
        0..=256 => cpl(6, 5, 0),
        257..=320 => cpl(8, 8, 0b100),
        321..=384 => cpl(11, 10, 0),
        385..=512 => cpl(12, 12, 0b110),
        _ => (None, 48),
    }
}

/// What one core codes, fixed at construction.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Core {
    /// 2 (2/0), 6 (2/2, 7.1's dependent substream) or 7 (3/2).
    pub(crate) acmod: u8,
    pub(crate) lfe: bool,
    /// The core's own rate, which picks its coupling range.
    pub(crate) kbps: u32,
    pub(crate) frame_bytes: usize,
    pub(crate) blocks: usize,
    pub(crate) syntax: Syntax,
}

impl Core {
    /// AC-3 2.0 or 5.1 at `kbps`, a rate of Table 5.18.
    pub(crate) fn ac3(layout: Layout, kbps: u32) -> Core {
        let (acmod, lfe) = match layout {
            Layout::Stereo => (2, false),
            _ => (7, true),
        };
        Core {
            acmod,
            lfe,
            kbps,
            frame_bytes: 4 * kbps as usize,
            blocks: 6,
            syntax: Syntax::ac3(kbps),
        }
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
    /// The frame's rematrixing flags, 2/0 only.
    remat: [bool; 4],
    /// Where the last frame's shared offset search ended, where the next one starts.
    hint: u32,
    /// Blocks from the start of the six-block group to this frame's first.
    group: usize,
    /// What the next frame carries in its skip fields, `ext_len` bytes of it, spread over
    /// the blocks in order; E-AC-3 only.
    ext: Box<[u8]>,
    ext_len: usize,
    /// The most `ext_len` may be: what a frame whose every block sends fresh exponents
    /// and coordinates leaves.
    ext_room: usize,
    transients: transient::Detector,
    /// Bits through block 1 of the last frame, which crc1 must cover.
    #[cfg(test)]
    first_bits: usize,
    /// Where block 5's mantissas start in the last frame, at or past the 5/8 point.
    #[cfg(test)]
    last_start: usize,
    /// Frames that went out at the all-zero offsets.
    #[cfg(test)]
    fallbacks: usize,
}

impl Ac3 {
    pub(crate) fn new(core: Core) -> Ac3 {
        let Core { acmod, lfe, .. } = core;
        let blocks = core.blocks.clamp(1, 6);
        let nfchans = match acmod {
            2 => 2,
            6 => 4,
            _ => 5,
        };
        let nch = nfchans + usize::from(lfe);
        let (cpl, chbwcod) = coupling_plan(nfchans, core.kbps);
        let stream = Stream {
            syntax: core.syntax,
            blocks,
            acmod,
            lfe,
            nfchans,
            chbwcod,
            cpl,
            frame_bytes: core.frame_bytes,
        };
        let range = |c: usize| match cpl {
            Some(b) if c == nch => (b.start(), b.end()),
            _ if c >= nfchans => (0, LFE_END),
            Some(b) => (0, b.start()),
            None => (0, 37 + 3 * (usize::from(chbwcod) + 12)),
        };
        let slots = stream.slots();
        let eac3 = stream.eac3();
        let mut core = Ac3 {
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
                .map(|c| Alloc::new(range(c).0, range(c).1, blocks))
                .collect(),
            mant: Box::new(Mantissas::new()),
            coupling: cpl.map(|b| Box::new(coupling::Frame::new(b))),
            remat: [false; 4],
            hint: 0,
            group: 0,
            ext: vec![0; if eac3 { SKIP_BYTES * blocks } else { 0 }].into_boxed_slice(),
            ext_len: 0,
            ext_room: 0,
            transients: transient::Detector::default(),
            #[cfg(test)]
            first_bits: 0,
            #[cfg(test)]
            last_start: 0,
            #[cfg(test)]
            fallbacks: 0,
        };
        if eac3 {
            core.ext_room = core.room();
        }
        core
    }

    /// Skip-field bytes every frame holds whatever its audio: the bits past the side
    /// information of every block sending fresh exponents (D45, the LFE D15: the most a
    /// set costs per block it covers) and coordinates, less 9 bits of `skipl` per block.
    fn room(&mut self) -> usize {
        let n = self.stream.blocks;
        let lfe = self.stream.lfe_slot();
        for (c, side) in self.side.iter_mut().enumerate() {
            let strategy = if Some(c) == lfe { D15 } else { D45 };
            side.strategy = [strategy; 6];
            for g in side.groups.iter_mut() {
                g.count = group_count(strategy, side.start, side.end);
            }
        }
        if let Some(f) = self.coupling.as_deref_mut() {
            f.send = [[true; coupling::MAX_FBW]; 6];
        }
        let mut scratch = vec![0u8; self.stream.frame_bytes];
        let Ok(budget) = self.size(&mut scratch) else {
            return 0;
        };
        let used = budget.header + budget.side.iter().sum::<u32>() + budget.trailer + 9 * n as u32;
        let left = budget.frame.saturating_sub(used) as usize / 8;
        left.min(self.ext.len())
    }

    /// Carries `bytes` in the next frame's skip fields, spread over its blocks in order.
    /// Refuses more than any frame of the stream can hold.
    pub(crate) fn set_extension(&mut self, bytes: &[u8]) -> Result<(), crate::Error> {
        let dst = self
            .ext
            .get_mut(..bytes.len())
            .filter(|_| bytes.len() <= self.ext_room)
            .ok_or(crate::Error::Length)?;
        dst.copy_from_slice(bytes);
        self.ext_len = bytes.len();
        Ok(())
    }

    pub(crate) fn frame_bytes(&self) -> usize {
        self.stream.frame_bytes
    }

    /// Samples per channel the frame's windows span: its blocks and the 256 past them.
    pub(crate) fn span(&self) -> usize {
        self.stream.blocks * BLOCK_SAMPLES + BLOCK_SAMPLES
    }

    /// The input of every channel, in slot order, `span` samples from the frame's first
    /// window.
    pub(crate) fn pcm_mut(&mut self) -> &mut [[f32; SPAN]] {
        &mut self.pcm
    }

    /// Keeps the 256 samples the next frame's first window shares with this one.
    pub(crate) fn advance(&mut self) {
        let n = self.stream.blocks;
        for p in self.pcm.iter_mut() {
            p.copy_within(n * BLOCK_SAMPLES..(n + 1) * BLOCK_SAMPLES, 0);
        }
        self.group = (self.group + n) % 6;
    }

    /// Encodes the frame in `pcm` into `out`, `frame_bytes` long, with the extension set
    /// for it. A frame the tuner's offsets do not fit goes out at the all-zero offsets
    /// instead: every `bap` 0, no mantissas, which every rate holds.
    pub(crate) fn encode(&mut self, out: &mut [u8]) {
        self.analyse();
        if let Ok(budget) = self.size(out) {
            let shared = self.stream.eac3();
            let tuned = tuner::tune(&self.alloc, &budget, &mut self.hint, shared);
            let written = tuned.map(|off| self.write(out, &budget, &off));
            if !matches!(written, Some(Ok(()))) {
                let _ = self.write(out, &budget, &Offsets::default());
                #[cfg(test)]
                {
                    self.fallbacks += 1;
                }
            }
        }
        self.ext_len = 0;
    }

    /// Everything up to the SNR offsets: transforms, the coupling channel, exponents and
    /// masking curves.
    fn analyse(&mut self) {
        let s = &self.stream;
        let (nf, n) = (s.nfchans, s.blocks);
        if BLOCK_SWITCHING {
            let mut blksw = [[false; 6]; coupling::MAX_FBW];
            self.transients.detect(&self.pcm[..nf], &mut blksw, n);
            for (side, sw) in self.side.iter_mut().zip(blksw).take(nf) {
                side.blksw = sw;
            }
        }
        for (c, p) in self.pcm.iter().enumerate() {
            if let (Some(coef), Some(side)) = (self.coef.get_mut(c), self.side.get(c)) {
                transform(p, &side.blksw, coef, n);
            }
        }
        if s.acmod == 2
            && let [l, r, ..] = &mut self.coef[..]
        {
            let top = s.cpl.map_or(self.alloc[0].end, |b| b.start());
            self.remat = rematrix(&mut l[..n], &mut r[..n], s.remat_bands(), top);
        }
        if let (Some(f), Some(cs)) = (self.coupling.as_deref_mut(), s.cpl_slot()) {
            let (fbw, rest) = self.coef.split_at_mut(nf);
            if let Some(cpl) = rest.get_mut(cs - nf) {
                coupling::form(fbw, cpl, f, n);
            }
        }
        for c in 0..s.slots() {
            // absexp, then gainrng and chbwcod where sent
            let fixed = match c {
                _ if c >= nf => 4,
                _ if s.cpl.is_some() => 6,
                _ => 12,
            };
            envelope(
                &self.coef[c],
                &mut self.raw[c],
                &mut self.side[c],
                &mut self.alloc[c],
                fixed,
                Some(c) == s.lfe_slot(),
            );
        }
    }

    /// syncinfo, BSI and, on E-AC-3, audfrm with the offsets.
    fn header(&self, w: &mut BitWriter, off: &Offsets) -> Result<(), Overflow> {
        let s = &self.stream;
        let group = self.group == 0;
        match s.syntax {
            Syntax::Ac3 { frmsizecod, .. } => pack::header(w, s, frmsizecod),
            Syntax::Eac3 { chanmap } => eac3::header(w, s, chanmap, &self.side, group, off),
        }
    }

    /// Where the mantissas must fit. Side information does not depend on the offsets'
    /// values: a dry run into `out` sizes it.
    fn size(&self, out: &mut [u8]) -> Result<Budget, Overflow> {
        let s = &self.stream;
        let choices = Choices {
            cpl: self.coupling.as_deref(),
            remat: self.remat,
        };
        let mut budget = Budget {
            header: 0,
            side: [0; 6],
            trailer: 18,
            frame: 8 * s.frame_bytes as u32,
            first: s.five_eighths(),
        };
        let mut w = BitWriter::new(out);
        self.header(&mut w, &Offsets::default())?;
        budget.header = w.position() as u32;
        let ext = &self.ext[..self.ext_len];
        for b in 0..s.blocks {
            let at = w.position();
            let off = Offsets::default();
            pack::block_side(&mut w, s, &self.side, choices, &off, b, skip_of(ext, b))?;
            budget.side[b] = (w.position() - at) as u32;
        }
        Ok(budget)
    }

    /// Writes the frame at `off`.
    fn write(&mut self, out: &mut [u8], budget: &Budget, off: &Offsets) -> Result<(), Overflow> {
        let s = &self.stream;
        let choices = Choices {
            cpl: self.coupling.as_deref(),
            remat: self.remat,
        };
        tuner::fill(&mut self.alloc, off);
        let mant = tuner::mantissa_bits(&self.alloc, off);
        let pad = budget.first.map_or([None; 6], |first| {
            let last_start = budget.header
                + (0..5).map(|b| budget.side[b] + mant[b]).sum::<u32>()
                + budget.side[5];
            pack::skips(first.saturating_sub(last_start))
        });
        let ext = &self.ext[..self.ext_len];

        let (order, n) = s.order();
        let order = &order[..n.min(MAX_SLOTS)];
        let mut w = BitWriter::new(out);
        self.header(&mut w, off)?;
        for (b, pad) in pad.into_iter().enumerate().take(s.blocks) {
            let skip = pad.map_or(skip_of(ext, b), |n| pack::ZEROS.get(..usize::from(n)));
            pack::block_side(&mut w, s, &self.side, choices, off, b, skip)?;
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
        /// The tuned frame fits; on AC-3 (§5.5), blocks 0 and 1 end inside the first 5/8 and
        /// block 5's mantissas start past it.
        fn assert_constraints(&self, what: &str) {
            assert_eq!(self.fallbacks, 0, "{what}: the tuned frame overflowed");
            if let Some(five_eighths) = self.stream.five_eighths() {
                let five_eighths = five_eighths as usize;
                assert!(self.first_bits <= five_eighths, "{what}: blocks 0-1");
                assert!(self.last_start >= five_eighths, "{what}: block 5");
            }
        }
    }

    /// Every codec, layout and rate a core codes on its own.
    fn configs() -> Vec<crate::Config> {
        let codecs = [
            crate::Blocks::One,
            crate::Blocks::Two,
            crate::Blocks::Three,
            crate::Blocks::Six,
        ]
        .map(|blocks| crate::Codec::Eac3 { blocks });
        let mut out = vec![];
        for codec in [crate::Codec::Ac3].into_iter().chain(codecs) {
            for layout in [Layout::Stereo, Layout::Surround51] {
                for &kbps in crate::Config::bitrates(codec, layout) {
                    out.extend(crate::Config::new(codec, layout, kbps));
                }
            }
        }
        out
    }

    /// Nearly equal L and R are coded as sum and difference in every band, and oxideav,
    /// undoing it, gets the input back.
    #[test]
    fn correlated_stereo_is_rematrixed() {
        let mut enc = Ac3::new(Core::ac3(Layout::Stereo, 192));
        let mut state = audblk::Ac3State::new();
        let mut out = vec![0u8; enc.frame_bytes()];
        let mut pcm = vec![0f32; 1536 * 2];
        let (mut want, mut got) = (Vec::<f32>::new(), Vec::<f32>::new());
        for frame in 0..4 {
            for (c, p) in enc.pcm_mut().iter_mut().enumerate() {
                for (i, v) in p.iter_mut().enumerate().skip(BLOCK_SAMPLES) {
                    let t = (frame * 1536 + i) as f32;
                    let x = 0.3 * (t * 0.031).sin() + 0.2 * (t * 0.17).sin();
                    *v = x * [1.0, 0.9][c] + [0.0, 0.01][c] * (t * 0.05).sin();
                }
            }
            enc.encode(&mut out);
            let si = syncinfo::parse(&out).unwrap();
            let b = bsi::parse(&out[5..]).unwrap();
            let side = audblk::parse_frame_side_info(&si, &b, &out).unwrap();
            assert_eq!(side[0].rematflg, [true; 4], "frame {frame}");
            audblk::decode_frame(&mut state, &si, &b, &out, &mut pcm).unwrap();
            if frame > 0 {
                // a frame decodes to its span's first 1536 samples
                got.extend(pcm.iter().step_by(2));
                want.extend(enc.pcm[0][..1536].iter());
            }
            enc.advance();
        }
        let (mut s, mut e) = (0f64, 0f64);
        for (w, g) in want.iter().zip(&got) {
            s += f64::from(*w).powi(2);
            e += f64::from(w - g).powi(2);
        }
        assert!(
            10.0 * (s / e).log10() > 40.0,
            "{} dB",
            10.0 * (s / e).log10()
        );
    }

    /// The fallback frame, all offsets zero, fits every codec and rate with the side
    /// information at its largest: noise that jumps 60 dB from block to block in every
    /// channel, so every block sends exponents and coordinates; on E-AC-3, with as large
    /// an extension as the stream takes.
    #[test]
    fn silence_fits_every_rate() {
        for config in configs() {
            let mut enc = Ac3::new(config.cores().0);
            if enc.stream.eac3() {
                assert!(enc.ext_room > 0, "{config:?}");
                let full = vec![0x5a; enc.ext_room + 1];
                assert!(enc.set_extension(&full).is_err(), "{config:?}");
                enc.set_extension(&full[1..]).unwrap();
            }
            let mut out = vec![0u8; enc.frame_bytes()];
            let mut seed = 3u32;
            for (c, p) in enc.pcm_mut().iter_mut().enumerate() {
                for (i, v) in p.iter_mut().enumerate() {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let loud = (i / BLOCK_SAMPLES + c).is_multiple_of(2);
                    let noise = (seed >> 9) as f32 / (1u32 << 22) as f32 - 1.0;
                    *v = noise * if loud { 0.9 } else { 9e-4 };
                }
            }
            enc.analyse();
            let what = format!("{config:?}");
            let budget = enc.size(&mut out).expect(&what);
            assert!(
                tuner::tune(&enc.alloc, &budget, &mut 0, false).is_some(),
                "{what}"
            );
            enc.write(&mut out, &budget, &Offsets::default())
                .expect(&what);
            enc.assert_constraints(&what);
            assert!(decoder::verify_packet_crc(&out).unwrap().all_ok(), "{what}");
        }
    }

    /// Silence and a tone 80 dB down leave most of the frame unused; skip fields, not
    /// auxdata, take the slack ahead of block 5.
    #[test]
    fn quiet_frames_keep_both_five_eighths_constraints() {
        for layout in [Layout::Stereo, Layout::Surround51] {
            for &kbps in crate::Config::bitrates(crate::Codec::Ac3, layout) {
                for level in [0.0, 1e-4] {
                    let mut enc = Ac3::new(Core::ac3(layout, kbps));
                    let mut out = vec![0u8; enc.frame_bytes()];
                    for frame in 0..2 {
                        for p in enc.pcm_mut().iter_mut() {
                            for (i, v) in p.iter_mut().enumerate().skip(BLOCK_SAMPLES) {
                                let t = (frame * 1536 + i) as f32;
                                *v = level * (t * 0.05).sin();
                            }
                        }
                        enc.encode(&mut out);
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
        let mut enc = Ac3::new(Core::ac3(Layout::Surround51, 320));
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
            enc.encode(&mut out);
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
            let mut enc = Ac3::new(Core::ac3(layout, kbps));
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
                enc.encode(&mut out);
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
