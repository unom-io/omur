//! The bit writer over one frame buffer, and the frame's CRC-16 (§7.10.1). Fixed capacity:
//! a write past the end is a tuner bug, surfaced as an error the frame drops on, never a
//! panic.

/// A write would pass the end of the frame buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Overflow;

/// MSB-first writer over a byte slice. Bits gather in a 64-bit accumulator and are stored
/// 32 at a time; `finish` stores the rest, zero-padded.
pub(crate) struct BitWriter<'a> {
    buf: &'a mut [u8],
    /// Bytes stored so far.
    byte: usize,
    acc: u64,
    /// Bits held in `acc`, below 32 between calls.
    held: u32,
}

impl<'a> BitWriter<'a> {
    pub(crate) fn new(buf: &'a mut [u8]) -> BitWriter<'a> {
        BitWriter {
            buf,
            byte: 0,
            acc: 0,
            held: 0,
        }
    }

    /// Bits written so far.
    pub(crate) fn position(&self) -> usize {
        self.byte * 8 + self.held as usize
    }

    /// Writes the low `bits` bits of `value`, `bits` at most 32.
    #[inline]
    pub(crate) fn put(&mut self, value: u32, bits: u32) -> Result<(), Overflow> {
        if self.position() + bits as usize > self.buf.len() * 8 {
            return Err(Overflow);
        }
        let mask = (1u64 << bits) - 1;
        self.acc = (self.acc << bits) | (u64::from(value) & mask);
        self.held += bits;
        if self.held >= 32 {
            self.held -= 32;
            let word = (self.acc >> self.held) as u32;
            if let Some(dst) = self.buf.get_mut(self.byte..self.byte + 4) {
                dst.copy_from_slice(&word.to_be_bytes());
            }
            self.byte += 4;
        }
        Ok(())
    }

    /// Writes `bits` zero bits.
    pub(crate) fn zeros(&mut self, mut bits: usize) -> Result<(), Overflow> {
        while bits > 0 {
            let n = bits.min(32);
            self.put(0, n as u32)?;
            bits -= n;
        }
        Ok(())
    }

    /// Stores the bits still held, the last byte zero-padded, and returns the bytes written.
    pub(crate) fn finish(mut self) -> usize {
        while self.held > 0 {
            let take = self.held.min(8);
            self.held -= take;
            let byte = ((self.acc >> self.held) << (8 - take)) as u8;
            if let Some(b) = self.buf.get_mut(self.byte) {
                *b = byte;
            }
            self.byte += 1;
        }
        self.byte
    }
}

/// The generator x^16 + x^15 + x^2 + 1 without its x^16 term.
const POLY: u16 = 0x8005;

const CRC_TABLE: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = (i as u16) << 8;
        let mut k = 0;
        while k < 8 {
            c = if c & 0x8000 != 0 {
                (c << 1) ^ POLY
            } else {
                c << 1
            };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// CRC-16 of `data`, register cleared, MSB first: the value that, appended, leaves the
/// register at zero.
pub(crate) fn crc16(data: &[u8]) -> u16 {
    data.iter().fold(0u16, |c, &b| {
        (c << 8) ^ CRC_TABLE[usize::from((c >> 8) as u8 ^ b)]
    })
}

/// `a · b` modulo the generator, as polynomials over GF(2).
fn mul_mod(a: u16, b: u16) -> u16 {
    let (mut r, mut a) = (0u16, a);
    for i in 0..16 {
        if (b >> i) & 1 != 0 {
            r ^= a;
        }
        a = if a & 0x8000 != 0 {
            (a << 1) ^ POLY
        } else {
            a << 1
        };
    }
    r
}

/// `x^-bits` modulo the generator: the factor that turns a trailing CRC into one that sits
/// `bits` bits earlier, ahead of the data it covers.
pub(crate) fn crc_shift_back(bits: usize) -> u16 {
    let mut r = 1u16;
    for _ in 0..bits {
        r = if r & 1 != 0 {
            ((r ^ POLY) >> 1) | 0x8000
        } else {
            r >> 1
        };
    }
    r
}

/// The 16-bit word that, placed ahead of `data`, leaves the register at zero after both.
/// `back` is `crc_shift_back(16 + 8 × data.len())`.
pub(crate) fn crc16_leading(data: &[u8], back: u16) -> u16 {
    mul_mod(crc16(data), back)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The register a decoder runs: data shifted in at the low end, zero when the CRC holds.
    fn residue(data: &[u8]) -> u16 {
        let mut c = 0u32;
        for &b in data {
            for i in (0..8).rev() {
                let top = c >> 15 & 1;
                c = ((c << 1) & 0xffff) | u32::from(b >> i & 1);
                if top != 0 {
                    c ^= u32::from(POLY);
                }
            }
        }
        c as u16
    }

    #[test]
    fn crc16_matches_the_check_value() {
        // CRC-16 over 0x8005, cleared, unreflected: the catalogue's "CRC-16/UMTS".
        assert_eq!(crc16(b"123456789"), 0xfee8);
    }

    #[test]
    fn trailing_and_leading_crcs_zero_the_register() {
        let data: Vec<u8> = (0..300u32).map(|i| (i * 37 + 11) as u8).collect();
        let mut tail = data.clone();
        tail.extend_from_slice(&crc16(&data).to_be_bytes());
        assert_eq!(residue(&tail), 0);
        let lead = crc16_leading(&data, crc_shift_back(16 + 8 * data.len()));
        let mut head = lead.to_be_bytes().to_vec();
        head.extend_from_slice(&data);
        assert_eq!(residue(&head), 0);
    }

    #[test]
    fn writer_packs_msb_first_and_refuses_overflow() {
        let mut buf = [0xffu8; 3];
        let mut w = BitWriter::new(&mut buf);
        w.put(0b101, 3).unwrap();
        w.put(0x0b77, 16).unwrap();
        assert_eq!(w.position(), 19);
        assert_eq!(w.put(0, 6), Err(Overflow));
        w.zeros(5).unwrap();
        assert_eq!(w.put(1, 1), Err(Overflow));
        assert_eq!(w.finish(), 3);
        // 101 0000 1011 0111 0111 00000
        assert_eq!(buf, [0b1010_0001, 0b0110_1110, 0b1110_0000]);
    }
}
