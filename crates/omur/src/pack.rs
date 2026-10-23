//! Syncframe assembly (§5.3): syncinfo, BSI, the audio blocks in syntax order, auxdata
//! padding and both CRCs (§7.10.1).

use crate::ac3::coupling::{self, Bands};
use crate::bitalloc::{
    CPLFLEAK, CPLSLEAK, DBPBCOD, FDCYCOD, FGAINCOD, FLOORCOD, MAX_SLOTS, Offsets, SDCYCOD, SGAINCOD,
};
use crate::bits::{BitWriter, Overflow, crc16, crc16_leading};
use crate::exponents::{Groups, REUSE};
use crate::mantissas::Mantissas;

/// What the syntax needs of the stream, fixed at construction. Slots hold the
/// full-bandwidth channels, then the LFE, then the coupling channel.
pub(crate) struct Stream {
    /// Blocks per frame: 6 on AC-3; 1, 2, 3 or 6 on E-AC-3.
    pub(crate) blocks: usize,
    pub(crate) acmod: u8,
    pub(crate) lfe: bool,
    pub(crate) nfchans: usize,
    pub(crate) frmsizecod: u8,
    /// Shared by every full-bandwidth channel when none is coupled.
    pub(crate) chbwcod: u8,
    /// The coupling range; every full-bandwidth channel is in it.
    pub(crate) cpl: Option<Bands>,
    pub(crate) frame_bytes: usize,
    /// Bytes from the frame's start to the 5/8 point crc1 closes (Table 7.34).
    pub(crate) crc1_end: usize,
    /// `crc_shift_back` over the 5/8 region past the syncword.
    pub(crate) crc1_back: u16,
}

impl Stream {
    pub(crate) fn lfe_slot(&self) -> Option<usize> {
        self.lfe.then_some(self.nfchans)
    }

    pub(crate) fn cpl_slot(&self) -> Option<usize> {
        self.cpl.map(|_| self.nfchans + usize::from(self.lfe))
    }

    /// `nrematbd` (Table 4.10).
    pub(crate) fn remat_bands(&self) -> usize {
        match self.cpl.map(|b| b.begf) {
            None | Some(3..) => 4,
            Some(1 | 2) => 3,
            Some(_) => 2,
        }
    }

    pub(crate) fn slots(&self) -> usize {
        self.nfchans + usize::from(self.lfe) + usize::from(self.cpl.is_some())
    }

    /// Slots in packing order (§5.4.3.61 to §5.4.3.63): channel 0, the coupling channel
    /// after the first coupled channel, the other full-bandwidth channels, the LFE.
    pub(crate) fn order(&self) -> ([usize; MAX_SLOTS], usize) {
        let mut out = [0; MAX_SLOTS];
        let mut n = 0;
        for c in 0..self.nfchans {
            out[n] = c;
            n += 1;
            if c == 0
                && let Some(cs) = self.cpl_slot()
            {
                out[n] = cs;
                n += 1;
            }
        }
        if let Some(l) = self.lfe_slot() {
            out[n] = l;
            n += 1;
        }
        (out, n)
    }
}

/// One channel's side information over the frame. `groups[b]` is sent where `strategy[b]`
/// is not reuse; its mantissas cover `start..end`.
pub(crate) struct Side {
    pub(crate) strategy: [u8; 6],
    pub(crate) groups: [Groups; 6],
    pub(crate) blksw: [bool; 6],
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// syncinfo with a zero crc1 (`seal` sets it), and the BSI (§5.3.1, §5.3.2).
pub(crate) fn header(w: &mut BitWriter, s: &Stream) -> Result<(), Overflow> {
    w.put(0x0b77, 16)?;
    w.put(0, 16)?; // crc1
    w.put(0, 2)?; // fscod: 48 kHz
    w.put(u32::from(s.frmsizecod), 6)?;
    w.put(8, 5)?; // bsid
    w.put(0, 3)?; // bsmod: complete main
    w.put(u32::from(s.acmod), 3)?;
    if s.acmod & 1 != 0 && s.acmod != 1 {
        w.put(0, 2)?; // cmixlev: -3 dB
    }
    if s.acmod & 4 != 0 {
        w.put(0, 2)?; // surmixlev: -3 dB
    }
    if s.acmod == 2 {
        w.put(0, 2)?; // dsurmod: not indicated
    }
    w.put(u32::from(s.lfe), 1)?;
    w.put(31, 5)?; // dialnorm: -31 dB
    // compre, langcode, audprodie, copyrightb, origbs, timecod1e, timecod2e, addbsie
    w.put(0b0000_1000, 8)
}

/// The frame's choices beyond each channel's: its coupling when the stream couples, and
/// which rematrixing bands of a 2/0 stream hold sum and difference.
#[derive(Clone, Copy)]
pub(crate) struct Choices<'a> {
    pub(crate) cpl: Option<&'a coupling::Frame>,
    pub(crate) remat: [bool; 4],
}

/// Block `blk`'s fields ahead of its mantissas (§5.3.3), with `skip` bytes of skip field.
/// Everything frame-wide goes in block 0 and is reused after.
pub(crate) fn block_side(
    w: &mut BitWriter,
    s: &Stream,
    side: &[Side],
    choices: Choices,
    off: &Offsets,
    blk: usize,
    skip: Option<u16>,
) -> Result<(), Overflow> {
    let first = blk == 0;
    let cpl = choices.cpl;
    let fbw = &side[..s.nfchans.min(side.len())];
    let lfe = s.lfe_slot().and_then(|l| side.get(l));
    let cs = s.cpl_slot().and_then(|c| side.get(c).map(|sd| (c, sd)));
    let cpl = cpl.zip(cs).map(|(f, (c, sd))| (f, c, sd));
    for c in fbw {
        w.put(u32::from(c.blksw[blk]), 1)?;
    }
    for _ in fbw {
        w.put(1, 1)?; // dithflag
    }
    w.put(0, 1)?; // dynrnge
    w.put(u32::from(first), 1)?; // cplstre
    if first {
        w.put(u32::from(cpl.is_some()), 1)?; // cplinu
        if let Some((f, ..)) = cpl {
            for _ in fbw {
                w.put(1, 1)?; // chincpl
            }
            if s.acmod == 2 {
                w.put(0, 1)?; // phsflginu
            }
            w.put(u32::from(f.bands.begf), 4)?;
            w.put(u32::from(f.bands.endf), 4)?;
            for sbnd in 1..f.bands.subbands() {
                w.put(f.bands.strc >> sbnd & 1, 1)?; // cplbndstrc
            }
        }
    }
    if let Some((f, ..)) = cpl {
        let n = f.bands.count();
        for (send, co) in f.send[blk].iter().zip(&f.coords[blk]).take(fbw.len()) {
            w.put(u32::from(*send), 1)?; // cplcoe
            if *send {
                w.put(u32::from(co.mstr), 2)?;
                for (e, m) in co.exp.iter().zip(&co.mant).take(n) {
                    w.put(u32::from(*e), 4)?;
                    w.put(u32::from(*m), 4)?;
                }
            }
        }
    }
    if s.acmod == 2 {
        w.put(u32::from(first), 1)?; // rematstr
        if first {
            for flag in choices.remat.iter().take(s.remat_bands()) {
                w.put(u32::from(*flag), 1)?; // rematflg
            }
        }
    }
    if let Some((.., c)) = cpl {
        w.put(u32::from(c.strategy[blk]), 2)?; // cplexpstr
    }
    for c in fbw {
        w.put(u32::from(c.strategy[blk]), 2)?;
    }
    if let Some(c) = lfe {
        w.put(u32::from(c.strategy[blk] != REUSE), 1)?;
    }
    if cpl.is_none() {
        for c in fbw {
            if c.strategy[blk] != REUSE {
                w.put(u32::from(s.chbwcod), 6)?;
            }
        }
    }
    if let Some((.., c)) = cpl
        && c.strategy[blk] != REUSE
    {
        exponents(w, &c.groups[blk])?; // cplabsexp, cplexps
    }
    for c in fbw {
        if c.strategy[blk] != REUSE {
            exponents(w, &c.groups[blk])?;
            w.put(0, 2)?; // gainrng
        }
    }
    if let Some(c) = lfe
        && c.strategy[blk] != REUSE
    {
        exponents(w, &c.groups[blk])?;
    }
    w.put(u32::from(first), 1)?; // baie
    if first {
        w.put(u32::from(SDCYCOD), 2)?;
        w.put(u32::from(FDCYCOD), 2)?;
        w.put(u32::from(SGAINCOD), 2)?;
        w.put(u32::from(DBPBCOD), 2)?;
        w.put(u32::from(FLOORCOD), 3)?;
    }
    w.put(u32::from(first), 1)?; // snroffste
    if first {
        w.put(u32::from(off.csnr), 6)?;
        if let Some((_, cs, _)) = cpl {
            w.put(u32::from(off.fsnr[cs]), 4)?;
            w.put(u32::from(FGAINCOD), 3)?;
        }
        for fine in off.fsnr.iter().take(fbw.len()) {
            w.put(u32::from(*fine), 4)?;
            w.put(u32::from(FGAINCOD), 3)?;
        }
        if let Some(l) = s.lfe_slot() {
            w.put(u32::from(off.fsnr[l]), 4)?;
            w.put(u32::from(FGAINCOD), 3)?;
        }
    }
    if cpl.is_some() {
        w.put(u32::from(first), 1)?; // cplleake
        if first {
            w.put(u32::from(CPLFLEAK), 3)?;
            w.put(u32::from(CPLSLEAK), 3)?;
        }
    }
    w.put(0, 1)?; // deltbaie
    w.put(u32::from(skip.is_some()), 1)?; // skiple
    if let Some(n) = skip {
        w.put(u32::from(n), 9)?;
        w.zeros(8 * usize::from(n))?;
    }
    Ok(())
}

/// Skip fields in blocks 2 to 4 that move block 5's mantissas `deficit` bits later, to the
/// 5/8 point (§5.5 constraint 2). Each costs 9 bits plus its bytes, 511 at most.
pub(crate) fn skips(deficit: u32) -> [Option<u16>; 6] {
    let mut out = [None; 6];
    let mut left = deficit;
    for s in out[2..5].iter_mut() {
        if left == 0 {
            break;
        }
        let bytes = left.saturating_sub(9).div_ceil(8).min(511);
        *s = Some(bytes as u16);
        left = left.saturating_sub(9 + 8 * bytes);
    }
    out
}

fn exponents(w: &mut BitWriter, g: &Groups) -> Result<(), Overflow> {
    w.put(u32::from(g.absexp), 4)?;
    for code in g.codes.iter().take(g.count) {
        w.put(u32::from(*code), 7)?;
    }
    Ok(())
}

/// The block's mantissas, slots in packing order.
pub(crate) fn mantissas(
    w: &mut BitWriter,
    m: &Mantissas,
    side: &[Side],
    order: &[usize],
) -> Result<(), Overflow> {
    for &c in order {
        let (Some(code), Some(bits), Some(s)) = (m.code.get(c), m.bits.get(c), side.get(c)) else {
            break;
        };
        // a bin without bits puts nothing: no branch to mispredict on bap 0
        for (v, n) in code.iter().zip(bits).take(s.end).skip(s.start) {
            w.put(u32::from(*v), u32::from(*n))?;
        }
    }
    Ok(())
}

/// Pads the auxiliary field with zeros up to auxdatae, crcrsv and crc2, all left zero.
pub(crate) fn close(mut w: BitWriter, s: &Stream) -> Result<(), Overflow> {
    let tail = s.frame_bytes * 8 - 18;
    let pad = tail.checked_sub(w.position()).ok_or(Overflow)?;
    w.zeros(pad)?;
    w.put(0, 18)?; // auxdatae, crcrsv, crc2
    w.finish();
    Ok(())
}

/// Sets crc1 and crc2 in a frame `close` has written. crc1 leaves the register at zero
/// over the first 5/8, so crc2 runs over the rest alone. crc2 never equals the syncword:
/// crcrsv flips when it would.
pub(crate) fn seal(frame: &mut [u8], s: &Stream) {
    let n = s.frame_bytes.min(frame.len());
    if n < 6 || s.crc1_end + 2 > n {
        return;
    }
    let crc1 = crc16_leading(&frame[4..s.crc1_end], s.crc1_back);
    frame[2..4].copy_from_slice(&crc1.to_be_bytes());
    let mut crc2 = crc16(&frame[s.crc1_end..n - 2]);
    if crc2 == 0x0b77 {
        frame[n - 3] ^= 0x01; // crcrsv, the bit before crc2
        crc2 = crc16(&frame[s.crc1_end..n - 2]);
    }
    frame[n - 2..n].copy_from_slice(&crc2.to_be_bytes());
}
