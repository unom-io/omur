//! Annex E: the `bsid` 16 header — BSI and the audfrm strategy block — for the
//! `numblkscod` frame lengths, and the channel map of the dependent substream that carries
//! 7.1's extra channels.

use crate::bitalloc::Offsets;
use crate::bits::{BitWriter, Overflow};
use crate::exponents::REUSE;
use crate::pack::{Side, Stream};

/// `chanmap` of 7.1's dependent substream (Table E.1.4, bit 0 the MSB): Ls and Rs, which
/// replace the independent substream's, and the Lrs/Rrs pair, which extends it (E.2.8.2).
/// Coded as 2/2 in that order.
pub(crate) const CHANMAP_71: u16 = 1 << (15 - 3) | 1 << (15 - 4) | 1 << (15 - 6);

/// `numblkscod` of a frame of `blocks` blocks (Table E.1.3).
fn numblkscod(blocks: usize) -> u32 {
    match blocks {
        1 => 0,
        2 => 1,
        3 => 2,
        _ => 3,
    }
}

/// A channel's exponent strategies as the row of Table E.1.9 a converter to AC-3 reads for
/// the six-block group the frame starts: bit `5 − k` set where block `k` sends exponents.
/// The table's strategy per set is the one `exponents::plan` picks. Blocks past a shorter
/// frame count as sending: the frames after it send in their block 0, and may in any other.
fn convexpstr(strategy: &[u8; 6], blocks: usize) -> u32 {
    (1..6).fold(0, |row, k| {
        row << 1 | u32::from(k >= blocks || strategy[k] != REUSE)
    })
}

/// syncinfo, BSI and audfrm (E.1.2.1 to E.1.2.3) of a frame whose channels decided `side`,
/// coded at `off`, one fine offset for every channel. `group` is set when the frame starts
/// a group of six blocks, which the converter fields of an independent substream mark.
/// The tools are off; on are per-block exponent strategies, block switching, dither flags,
/// bit allocation parameters and skip fields. The offsets go in audfrm (`snroffststr` 0):
/// ffmpeg 9.0.1 reads `snroffste` in block 0 and in no other, against E.1.2.4, so the
/// per-block strategies decode wrongly there.
pub(crate) fn header(
    w: &mut BitWriter,
    s: &Stream,
    chanmap: Option<u16>,
    side: &[Side],
    group: bool,
    off: &Offsets,
) -> Result<(), Overflow> {
    let six = s.blocks == 6;
    w.put(0x0b77, 16)?;
    w.put(u32::from(chanmap.is_some()), 2)?; // strmtyp: independent or dependent
    w.put(0, 3)?; // substreamid
    w.put((s.frame_bytes / 2).saturating_sub(1) as u32, 11)?; // frmsiz
    w.put(0, 2)?; // fscod: 48 kHz
    w.put(numblkscod(s.blocks), 2)?;
    w.put(u32::from(s.acmod), 3)?;
    w.put(u32::from(s.lfe), 1)?;
    w.put(16, 5)?; // bsid
    w.put(31, 5)?; // dialnorm: -31 dB
    // compre marks the last dependent substream of a programme (E.2.8.5); its compr is 0 dB
    if chanmap.is_some() {
        w.put(1, 1)?;
        w.put(0, 8)?;
    } else {
        w.put(0, 1)?;
    }
    if let Some(map) = chanmap {
        w.put(1, 1)?; // chanmape
        w.put(u32::from(map), 16)?;
    }
    w.put(0, 2)?; // mixmdate, infomdate
    if chanmap.is_none() && !six {
        w.put(u32::from(group), 1)?; // convsync
    }
    addbsi(w, s)?;

    if six {
        w.put(0b10, 2)?; // expstre, ahte
    }
    w.put(0, 2)?; // snroffststr: the frame's offsets, in audfrm
    // transproce, blkswe, dithflage, bamode, frmfgaincode, dbaflde, skipflde, spxattene
    w.put(0b0111_0010, 8)?;
    let fbw = &side[..s.nfchans.min(side.len())];
    let cpl = s.cpl_slot().and_then(|c| side.get(c));
    if s.acmod > 1 {
        w.put(u32::from(cpl.is_some()), 1)?; // cplinu of block 0
        w.put(0, s.blocks as u32 - 1)?; // cplstre: block 0's holds
    }
    for b in 0..s.blocks {
        if let Some(c) = cpl {
            w.put(u32::from(c.strategy[b]), 2)?; // cplexpstr
        }
        for c in fbw {
            w.put(u32::from(c.strategy[b]), 2)?; // chexpstr
        }
    }
    if let Some(c) = s.lfe_slot().and_then(|l| side.get(l)) {
        for b in 0..s.blocks {
            w.put(u32::from(c.strategy[b] != REUSE), 1)?; // lfeexpstr
        }
    }
    if chanmap.is_none() {
        if !six {
            w.put(u32::from(group), 1)?; // convexpstre
        }
        if six || group {
            for c in fbw {
                w.put(convexpstr(&c.strategy, s.blocks), 5)?;
            }
        }
    }
    w.put(u32::from(off.csnr), 6)?; // frmcsnroffst
    w.put(u32::from(off.fsnr[0]), 4)?; // frmfsnroffst
    if s.blocks > 1 {
        w.put(0, 1)?; // blkstrtinfoe
    }
    Ok(())
}

/// `addbsie` and, on an object-audio stream, `addbsil` and the two bytes of TS 103 420
/// clause 8.3 that signal the extension.
#[cfg(feature = "joc")]
fn addbsi(w: &mut BitWriter, s: &Stream) -> Result<(), Overflow> {
    match s.syntax {
        crate::pack::Syntax::Eac3 {
            objects: Some(n), ..
        } => {
            w.put(1, 1)?;
            w.put(1, 6)?; // addbsil: two bytes
            for b in crate::joc::addbsi(n) {
                w.put(u32::from(b), 8)?;
            }
            Ok(())
        }
        _ => w.put(0, 1),
    }
}

#[cfg(not(feature = "joc"))]
fn addbsi(w: &mut BitWriter, _: &Stream) -> Result<(), Overflow> {
    w.put(0, 1) // addbsie
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exponents::{D15, D25, D45};

    #[test]
    fn converter_rows_follow_table_e_1_9() {
        let r = REUSE;
        assert_eq!(convexpstr(&[D15, r, r, r, r, r], 6), 0);
        assert_eq!(convexpstr(&[D15, r, r, r, r, D45], 6), 1);
        assert_eq!(convexpstr(&[D25, r, D15, r, r, r], 6), 8);
        assert_eq!(convexpstr(&[D45, D25, r, r, D25, r], 6), 18);
        assert_eq!(convexpstr(&[D45; 6], 6), 31);
        // one block a frame: every block sends
        assert_eq!(convexpstr(&[D45, r, r, r, r, r], 1), 31);
        // three: the next frame's block 0 is block 3 of the group
        assert_eq!(convexpstr(&[D25, r, r, r, r, r], 3), 7);
    }

    #[test]
    fn the_71_map_names_ls_rs_and_the_rear_pair() {
        assert_eq!(CHANMAP_71, 0b0001_1010_0000_0000);
    }

    /// An object-audio stream's BSI carries TS 103 420's two bytes, as oxideav reads them.
    #[cfg(feature = "joc")]
    #[test]
    fn an_object_stream_signals_extension_type_a() {
        use crate::{Blocks, Codec, Config, Encoder, Layout};
        use oxideav_ac3::eac3::{bsi, joc::parse_ec3_extension_type_a};
        let six = Codec::Eac3 {
            blocks: Blocks::Six,
        };
        let config = Config::new(six, Layout::Surround51, 768).unwrap();
        let mut enc = Encoder::objects(config, 11);
        enc.push(&[0.25; 6 * 1536]).unwrap();
        enc.push(&[0.25; 6 * 256]).unwrap();
        let frame = enc.pull().unwrap();
        let b = bsi::parse(&frame.bytes[2..]).unwrap();
        let addbsi = b.addbsi.as_ref().unwrap().payload();
        assert_eq!(addbsi, [1, 11]);
        assert!(parse_ec3_extension_type_a(addbsi).unwrap().is_some());
        assert!(
            oxideav_ac3::decoder::verify_packet_crc(frame.bytes)
                .unwrap()
                .all_ok()
        );
    }
}
