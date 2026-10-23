//! The EMDF container (TS 102 366 Annex H) that carries the OAMD and JOC payloads into the
//! core frame's skip fields (TS 103 420 clause 8.2), and the `addbsi` extension signal
//! (clause 8.3).

use crate::bits::{BitWriter, Overflow};

/// `emdf_payload_id` of each payload (Table 55).
pub(crate) const OAMD: u32 = 11;
pub(crate) const JOC: u32 = 14;

/// Bytes one container takes at most: one skip field (`skipl`, 9 bits). Annex H carries a
/// container in one reserved-data field, and a decoder looks for it in one.
pub(crate) const MAX: usize = 511;

/// The `addbsi` bytes of clause 8.3.1: seven reserved bits, `flag_ec3_extension_type_a`
/// set, and `complexity_index_type_a`, the objects the program carries (16 at most).
pub(crate) fn addbsi(complexity: u8) -> [u8; 2] {
    [1, complexity.min(16)]
}

/// `variable_bits(n)` (H.2.1.2.1), also TS 103 420's `variable_bits_max` (clause 5.5.1)
/// within its group limit: n-bit groups, each but the last followed by a set `read_more`,
/// the value less the offsets the extra groups add.
pub(crate) fn variable_bits(w: &mut BitWriter, value: u32, n: u32) -> Result<(), Overflow> {
    let (mut groups, mut offset) = (1u32, 0u64);
    while u64::from(value) - offset >= 1u64 << (n * groups) {
        offset += 1u64 << (n * groups);
        groups += 1;
    }
    let coded = u64::from(value) - offset;
    for g in (0..groups).rev() {
        w.put(((coded >> (n * g)) & ((1 << n) - 1)) as u32, n)?;
        w.put(u32::from(g > 0), 1)?;
    }
    Ok(())
}

/// Bits `variable_bits(n)` takes for `value`.
pub(crate) fn variable_len(value: u32, n: u32) -> usize {
    let (mut groups, mut offset) = (1u32, 0u64);
    while u64::from(value) - offset >= 1u64 << (n * groups) {
        offset += 1u64 << (n * groups);
        groups += 1;
    }
    (groups * (n + 1)) as usize
}

/// `emdf_payload_config` as Table 56 fixes it: no sample offset or duration, group 0, codec
/// data present and zero, kept on transcode, frame aligned, priority 0, no processing.
fn payload_config(w: &mut BitWriter) -> Result<(), Overflow> {
    w.put(0b001, 3)?; // smploffste, duratione, groupide
    variable_bits(w, 0, 2)?; // groupid
    w.put(1, 1)?; // codecdatae
    w.put(0, 8)?;
    w.put(0b0100, 4)?; // discard_unknown_payload, payload_frame_aligned, create/remove duplicate
    w.put(0, 7) // priority, proc_allowed
}

/// Bytes of a container holding payloads of `oamd` and `joc` bytes.
pub(crate) fn len(oamd: usize, joc: usize) -> usize {
    let payload = |n: usize| 5 + 26 + variable_len(n as u32, 8) + 8 * n;
    // version, key_id, the payloads, the end id, 8 bits of primary protection
    let bits = 2 + 3 + payload(oamd) + payload(joc) + 5 + 4 + 8;
    4 + bits.div_ceil(8)
}

/// Writes `emdf_sync` and the container holding `oamd` then `joc` into `out`, returning
/// its length. The protection field is zero: Annex H leaves its value to the implementation.
pub(crate) fn write(out: &mut [u8], oamd: &[u8], joc: &[u8]) -> Result<usize, Overflow> {
    let body = out.get_mut(4..).ok_or(Overflow)?;
    let mut w = BitWriter::new(body);
    w.put(0, 5)?; // emdf_version, key_id
    for (id, bytes) in [(OAMD, oamd), (JOC, joc)] {
        w.put(id, 5)?;
        payload_config(&mut w)?;
        variable_bits(&mut w, bytes.len() as u32, 8)?;
        for b in bytes {
            w.put(u32::from(*b), 8)?;
        }
    }
    w.put(0, 5)?; // the end of the payloads
    w.put(0b0100, 4)?; // protection_length_primary 8 bits, no secondary
    w.put(0, 8)?;
    let n = w.finish();
    let head = out.get_mut(..4).ok_or(Overflow)?;
    head.copy_from_slice(&[0x58, 0x38, (n >> 8) as u8, n as u8]);
    Ok(4 + n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_ac3::eac3::joc::{parse_ec3_extension_type_a, parse_joc_emdf};

    /// A JOC payload of one object, one band, every coefficient symbol 0 (clause 6.2): clip
    /// gain 1 at bit 12, the object's present flag at bit 30.
    const JOC_ONE: [u8; 6] = [0x00, 0x08, 0x00, 0x02, 0x00, 0x00];

    #[test]
    fn variable_bits_round_trip_table_h_2_1() {
        for (value, n, len) in [
            (0, 8, 9),
            (255, 8, 9),
            (256, 8, 18),
            (65_791, 8, 18),
            (3, 2, 3),
        ] {
            assert_eq!(variable_len(value, n), len, "{value}");
            let mut buf = [0u8; 8];
            let mut w = BitWriter::new(&mut buf);
            variable_bits(&mut w, value, n).unwrap();
            assert_eq!(w.position(), len);
            w.finish();
            // H.2.1.2.1's reader
            let bit = |i: usize| u32::from(buf[i / 8] >> (7 - i % 8) & 1);
            let (mut at, mut got) = (0, 0u32);
            loop {
                let mut read = 0;
                for _ in 0..n {
                    read = read << 1 | bit(at);
                    at += 1;
                }
                got += read;
                at += 1;
                if bit(at - 1) == 0 {
                    break;
                }
                got = (got << n) + (1 << n);
            }
            assert_eq!((got, at), (value, len));
        }
    }

    #[test]
    fn oxideav_reads_the_container_and_the_signal() {
        let signal = parse_ec3_extension_type_a(&addbsi(11)).unwrap().unwrap();
        for oamd in [&[0x12u8, 0x34][..], &[0x55; 300]] {
            let mut out = [0u8; MAX];
            let n = write(&mut out, oamd, &JOC_ONE).unwrap();
            assert_eq!(n, len(oamd.len(), JOC_ONE.len()));
            let meta = parse_joc_emdf(&out[..n], signal).unwrap();
            let dbg = format!("{meta:?}");
            assert!(dbg.contains("complexity_index: 11"), "{dbg}");
            assert!(dbg.contains("num_objects: 1,"), "{dbg}");
        }
        assert!(parse_ec3_extension_type_a(&addbsi(17)).unwrap().is_some());
    }
}
