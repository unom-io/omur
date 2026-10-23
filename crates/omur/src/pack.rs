//! Syncframe assembly (§5.3): syncinfo, BSI, the audio blocks in syntax order, auxdata
//! padding and both CRCs (§7.10.1).

use crate::bitalloc::{DBPBCOD, FDCYCOD, FGAINCOD, FLOORCOD, Offsets, SDCYCOD, SGAINCOD};
use crate::bits::{BitWriter, Overflow, crc16, crc16_leading};
use crate::exponents::{Groups, REUSE};
use crate::mantissas::Mantissas;

/// What the syntax needs of the stream, fixed at construction.
pub(crate) struct Stream {
    pub(crate) acmod: u8,
    pub(crate) lfe: bool,
    pub(crate) nfchans: usize,
    pub(crate) frmsizecod: u8,
    /// Shared by every full-bandwidth channel: none is coupled.
    pub(crate) chbwcod: u8,
    pub(crate) frame_bytes: usize,
    /// Bytes from the frame's start to the 5/8 point crc1 closes (Table 7.34).
    pub(crate) crc1_end: usize,
    /// `crc_shift_back` over the 5/8 region past the syncword.
    pub(crate) crc1_back: u16,
}

/// The frame's coupling strategy. Never in use: the coupling syntax is Stage 1b.
#[derive(Clone, Copy, Default)]
pub(crate) struct Coupling {
    pub(crate) in_use: bool,
}

/// One channel's side information over the frame. `groups[b]` is sent where `strategy[b]`
/// is not reuse.
pub(crate) struct Side {
    pub(crate) strategy: [u8; 6],
    pub(crate) groups: [Groups; 6],
    pub(crate) blksw: [bool; 6],
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

/// Block `blk`'s fields ahead of its mantissas (§5.3.3), with `skip` bytes of skip field.
/// Everything frame-wide goes in block 0 and is reused after.
pub(crate) fn block_side(
    w: &mut BitWriter,
    s: &Stream,
    side: &[Side],
    cpl: Coupling,
    off: Offsets,
    blk: usize,
    skip: Option<u16>,
) -> Result<(), Overflow> {
    let first = blk == 0;
    let fbw = &side[..s.nfchans.min(side.len())];
    for c in fbw {
        w.put(u32::from(c.blksw[blk]), 1)?;
    }
    for _ in fbw {
        w.put(1, 1)?; // dithflag
    }
    w.put(0, 1)?; // dynrnge
    w.put(u32::from(first), 1)?; // cplstre
    if first {
        w.put(u32::from(cpl.in_use), 1)?; // cplinu
    }
    if s.acmod == 2 {
        w.put(u32::from(first), 1)?; // rematstr
        if first {
            w.put(0, 4)?; // rematflg: no band rematrixed
        }
    }
    for c in fbw {
        w.put(u32::from(c.strategy[blk]), 2)?;
    }
    let lfe = side.get(s.nfchans).filter(|_| s.lfe);
    if let Some(c) = lfe {
        w.put(u32::from(c.strategy[blk] != REUSE), 1)?;
    }
    for c in fbw {
        if c.strategy[blk] != REUSE {
            w.put(u32::from(s.chbwcod), 6)?;
        }
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
        for _ in fbw {
            w.put(u32::from(off.fsnr), 4)?;
            w.put(u32::from(FGAINCOD), 3)?;
        }
        if lfe.is_some() {
            w.put(u32::from(off.lfefsnr), 4)?;
            w.put(u32::from(FGAINCOD), 3)?;
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

/// The block's mantissas in channel order, LFE last.
pub(crate) fn mantissas(w: &mut BitWriter, m: &Mantissas, side: &[Side]) -> Result<(), Overflow> {
    for (c, s) in side.iter().enumerate() {
        let (Some(code), Some(bits)) = (m.code.get(c), m.bits.get(c)) else {
            break;
        };
        for (v, n) in code.iter().zip(bits).take(s.end) {
            if *n > 0 {
                w.put(u32::from(*v), u32::from(*n))?;
            }
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

/// Sets crc1 and crc2 in a frame `close` has written. crc2 never equals the syncword:
/// crcrsv flips when it would.
pub(crate) fn seal(frame: &mut [u8], s: &Stream) {
    let n = s.frame_bytes.min(frame.len());
    if n < 6 || s.crc1_end > n {
        return;
    }
    let crc1 = crc16_leading(&frame[4..s.crc1_end], s.crc1_back);
    frame[2..4].copy_from_slice(&crc1.to_be_bytes());
    let mut crc2 = crc16(&frame[2..n - 2]);
    if crc2 == 0x0b77 {
        frame[n - 3] ^= 0x01; // crcrsv, the bit before crc2
        crc2 = crc16(&frame[2..n - 2]);
    }
    frame[n - 2..n].copy_from_slice(&crc2.to_be_bytes());
}
