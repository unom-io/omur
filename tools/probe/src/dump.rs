// `probe dump <stream.ac3> [summary]`: what an AC-3 encoder decided, frame by frame.
// syncinfo and BSI come from oxideav's parsers; the audio blocks are walked here because the
// bits per channel need each block's bap arrays, and every block is checked against
// oxideav's own side-information parse.
use std::collections::BTreeMap;
use std::fmt::Write as _;

use oxideav_ac3::{audblk, bsi, syncinfo, tables as t};
use oxideav_core::Result;
use oxideav_core::bits::BitReader;

/// Slots: five fbw channels, the coupling channel, the LFE.
const CPL: usize = 5;
const LFE: usize = 6;

/// What persists from block to block inside a frame.
struct State {
    nfchans: usize,
    lfeon: bool,
    acmod: u8,
    cplinu: bool,
    chincpl: [bool; 5],
    phsflginu: bool,
    cplbegf: usize,
    cplendf: usize,
    nsub: usize,
    nbnd: usize,
    exp: [[u8; 256]; 7],
    start: [usize; 7],
    end: [usize; 7],
    /// sdcycod fdcycod sgaincod dbpbcod floorcod
    bai: [u8; 5],
    csnr: u8,
    fsnr: [u8; 7],
    fgain: [u8; 7],
    leak: (u8, u8),
    /// Per fbw channel and the coupling channel (slot 5): segments as (offset, len, ba).
    delt: [Vec<(u8, u8, u8)>; 6],
}

/// `mstrcplco`, then `(cplcoexp, cplcomant)` per band.
type Coords = (u8, Vec<(u8, u8)>);

/// One block as read, for printing and the summary.
#[derive(Default, Clone)]
struct Block {
    blksw: [bool; 5],
    dith: [bool; 5],
    cplstr: Option<(bool, usize, usize, String)>,
    cplcoe: [bool; 5],
    cplco: [Option<Coords>; 5],
    remat: Option<Vec<bool>>,
    cplexpstr: u8,
    chexpstr: [u8; 5],
    lfeexpstr: u8,
    chbwcod: [Option<u8>; 5],
    baie: Option<[u8; 5]>,
    snr: Option<(u8, [u8; 7], [u8; 7])>,
    leak: Option<(u8, u8)>,
    deltbaie: bool,
    skipl: Option<u32>,
    exp_bits: [u32; 7],
    mant_bits: [u32; 7],
    /// Bit position of the first mantissa, counted from the syncword.
    mant_start: u64,
    /// Bit position past the last mantissa.
    end: u64,
}

impl State {
    fn new(nfchans: usize, lfeon: bool, acmod: u8) -> State {
        State {
            nfchans,
            lfeon,
            acmod,
            cplinu: false,
            chincpl: [false; 5],
            phsflginu: false,
            cplbegf: 0,
            cplendf: 0,
            nsub: 0,
            nbnd: 0,
            exp: [[0; 256]; 7],
            start: [0; 7],
            end: [0; 7],
            bai: [0; 5],
            csnr: 0,
            fsnr: [0; 7],
            fgain: [0; 7],
            leak: (0, 0),
            delt: Default::default(),
        }
    }
}

fn grpsize(expstr: u8) -> usize {
    match expstr {
        2 => 2,
        3 => 4,
        _ => 1,
    }
}

/// The 7-bit groups after `absexp`, rebuilt into `exp[first..]` with `size` bins a value.
fn exponents(
    br: &mut BitReader,
    absexp: i32,
    ngrps: usize,
    size: usize,
    exp: &mut [u8; 256],
    first: usize,
    end: usize,
) -> Result<()> {
    let mut prev = absexp;
    for g in 0..ngrps {
        let code = br.read_u32(7)? as i32;
        for (k, m) in [code / 25, code % 25 / 5, code % 5].into_iter().enumerate() {
            prev += m - 2;
            for j in 0..size {
                let bin = first + (3 * g + k) * size + j;
                if bin < end.min(256) {
                    exp[bin] = prev.clamp(0, 24) as u8;
                }
            }
        }
    }
    Ok(())
}

fn logadd(a: i32, b: i32) -> i32 {
    let c = a - b;
    let add = i32::from(t::LATAB[((c.abs() >> 1) as usize).min(255)]);
    if c >= 0 { a + add } else { b + add }
}

fn lowcomp(a: i32, b0: i32, b1: i32, band: usize) -> i32 {
    if band < 20 {
        if b0 + 256 == b1 {
            if band < 7 { 384 } else { 320 }
        } else if b0 > b1 {
            (a - 64).max(0)
        } else {
            a
        }
    } else {
        (a - 128).max(0)
    }
}

/// §7.2.2 for slot `ch`, as printed.
fn bap(st: &State, ch: usize) -> [u8; 256] {
    let mut out = [0u8; 256];
    let (start, end) = (st.start[ch], st.end[ch].min(253));
    if end <= start {
        return out;
    }
    let psd: Vec<i32> = (0..256)
        .map(|b| 3072 - (i32::from(st.exp[ch][b]) << 7))
        .collect();
    let mut bndpsd = [0i32; 50];
    let (mut j, mut k) = (start, t::MASKTAB[start] as usize);
    loop {
        let last = (t::BNDTAB[k] + t::BNDSZ[k]) as usize;
        let last = last.min(end);
        bndpsd[k] = psd[j];
        j += 1;
        while j < last {
            bndpsd[k] = logadd(bndpsd[k], psd[j]);
            j += 1;
        }
        k += 1;
        if end <= last {
            break;
        }
    }
    let sdecay = t::SLOWDEC[st.bai[0] as usize];
    let fdecay = t::FASTDEC[st.bai[1] as usize];
    let sgain = t::SLOWGAIN[st.bai[2] as usize];
    let dbknee = t::DBPBTAB[st.bai[3] as usize];
    let floor = t::FLOORTAB[st.bai[4] as usize];
    let fgain = t::FASTGAIN[st.fgain[ch] as usize];
    let snroffset = (((i32::from(st.csnr) - 15) << 4) + i32::from(st.fsnr[ch])) << 2;
    let bndstrt = t::MASKTAB[start] as usize;
    let bndend = t::MASKTAB[end - 1] as usize + 1;
    let mut excite = [0i32; 50];
    let (mut fast, mut slow, begin);
    if bndstrt == 0 {
        let lfe = bndend == 7;
        let mut low = lowcomp(0, bndpsd[0], bndpsd[1], 0);
        excite[0] = bndpsd[0] - fgain - low;
        low = lowcomp(low, bndpsd[1], bndpsd[2], 1);
        excite[1] = bndpsd[1] - fgain - low;
        let mut b0 = 7;
        (fast, slow) = (0, 0);
        for b in 2..7 {
            if !(lfe && b == 6) {
                low = lowcomp(low, bndpsd[b], bndpsd[b + 1], b);
            }
            fast = bndpsd[b] - fgain;
            slow = bndpsd[b] - sgain;
            excite[b] = fast - low;
            if !(lfe && b == 6) && bndpsd[b] <= bndpsd[b + 1] {
                b0 = b + 1;
                break;
            }
        }
        for b in b0..bndend.min(22) {
            if !(lfe && b == 6) {
                low = lowcomp(low, bndpsd[b], bndpsd[b + 1], b);
            }
            fast = (fast - fdecay).max(bndpsd[b] - fgain);
            slow = (slow - sdecay).max(bndpsd[b] - sgain);
            excite[b] = (fast - low).max(slow);
        }
        begin = 22;
    } else {
        fast = (i32::from(st.leak.0) << 8) + 768;
        slow = (i32::from(st.leak.1) << 8) + 768;
        begin = bndstrt;
    }
    for b in begin..bndend {
        fast = (fast - fdecay).max(bndpsd[b] - fgain);
        slow = (slow - sdecay).max(bndpsd[b] - sgain);
        excite[b] = fast.max(slow);
    }
    let mut mask = [0i32; 50];
    for b in bndstrt..bndend {
        if bndpsd[b] < dbknee {
            excite[b] += (dbknee - bndpsd[b]) >> 2;
        }
        mask[b] = excite[b].max(i32::from(t::HTH[0][b]));
    }
    if ch != LFE {
        let mut band = 0;
        for &(off, len, ba) in &st.delt[ch] {
            band += off as usize;
            let delta = if ba >= 4 {
                i32::from(ba) - 3
            } else {
                i32::from(ba) - 4
            } << 7;
            for _ in 0..len {
                if band < 50 {
                    mask[band] += delta;
                }
                band += 1;
            }
        }
    }
    let (mut i, mut j) = (start, bndstrt);
    loop {
        let last = ((t::BNDTAB[j] + t::BNDSZ[j]) as usize).min(end);
        let m = ((mask[j] - snroffset - floor).max(0) & 0x1fe0) + floor;
        while i < last {
            out[i] = t::BAPTAB[((psd[i] - m) >> 5).clamp(0, 63) as usize];
            i += 1;
        }
        j += 1;
        if end <= last {
            break;
        }
    }
    out
}

/// Grouped classes open across channels within a block (§7.3.5).
#[derive(Default)]
struct Groups([u32; 3]);

impl Groups {
    fn bits(&mut self, bap: u8) -> u32 {
        let (class, size, bits) = match bap {
            0 => return 0,
            1 => (0, 3, 5),
            2 => (1, 3, 7),
            4 => (2, 2, 7),
            b => return u32::from(t::QUANTIZATION_BITS[b as usize]),
        };
        let open = &mut self.0[class];
        let cost = if *open == 0 { bits } else { 0 };
        *open = (*open + 1) % size;
        cost
    }
}

fn block(st: &mut State, br: &mut BitReader, blk: usize) -> Result<Block> {
    let mut b = Block::default();
    let nf = st.nfchans;
    for ch in 0..nf {
        b.blksw[ch] = br.read_u32(1)? != 0;
    }
    for ch in 0..nf {
        b.dith[ch] = br.read_u32(1)? != 0;
    }
    if br.read_u32(1)? != 0 {
        br.skip(8)?; // dynrng
    }
    if st.acmod == 0 && br.read_u32(1)? != 0 {
        br.skip(8)?;
    }
    if br.read_u32(1)? != 0 {
        st.cplinu = br.read_u32(1)? != 0;
        let mut strc = String::new();
        if st.cplinu {
            for ch in 0..nf {
                st.chincpl[ch] = br.read_u32(1)? != 0;
            }
            st.phsflginu = st.acmod == 2 && br.read_u32(1)? != 0;
            st.cplbegf = br.read_u32(4)? as usize;
            st.cplendf = br.read_u32(4)? as usize;
            st.nsub = (3 + st.cplendf).saturating_sub(st.cplbegf);
            st.nbnd = st.nsub;
            for _ in 1..st.nsub {
                let v = br.read_u32(1)? != 0;
                strc.push(if v { '1' } else { '0' });
                st.nbnd -= usize::from(v);
            }
            st.start[CPL] = 37 + 12 * st.cplbegf;
            st.end[CPL] = 37 + 12 * (st.cplendf + 3);
        } else {
            st.chincpl = [false; 5];
        }
        b.cplstr = Some((st.cplinu, st.cplbegf, st.cplendf, strc));
    }
    if st.cplinu {
        let mut any = false;
        for ch in 0..nf {
            if st.chincpl[ch] {
                b.cplcoe[ch] = br.read_u32(1)? != 0;
                if b.cplcoe[ch] {
                    any = true;
                    let mstr = br.read_u32(2)? as u8;
                    let mut co = Vec::new();
                    for _ in 0..st.nbnd {
                        co.push((br.read_u32(4)? as u8, br.read_u32(4)? as u8));
                    }
                    b.cplco[ch] = Some((mstr, co));
                }
            }
        }
        if st.acmod == 2 && st.phsflginu && any {
            br.skip(st.nbnd as u32)?;
        }
    }
    if st.acmod == 2 && br.read_u32(1)? != 0 {
        let n = match (st.cplinu, st.cplbegf) {
            (false, _) | (true, 3..) => 4,
            (true, 1 | 2) => 3,
            (true, 0) => 2,
        };
        b.remat = Some(
            (0..n)
                .map(|_| br.read_u32(1).map(|v| v != 0))
                .collect::<Result<_>>()?,
        );
    }
    if st.cplinu {
        b.cplexpstr = br.read_u32(2)? as u8;
    }
    for ch in 0..nf {
        b.chexpstr[ch] = br.read_u32(2)? as u8;
    }
    if st.lfeon {
        b.lfeexpstr = br.read_u32(1)? as u8;
    }
    for ch in 0..nf {
        if b.chexpstr[ch] != 0 && !st.chincpl[ch] {
            b.chbwcod[ch] = Some(br.read_u32(6)? as u8);
        }
    }
    if st.cplinu && b.cplexpstr != 0 {
        let abs = (br.read_u32(4)? as i32) << 1;
        let size = grpsize(b.cplexpstr);
        let n = (st.end[CPL] - st.start[CPL]) / (3 * size);
        let (first, end) = (st.start[CPL], st.end[CPL]);
        exponents(br, abs, n, size, &mut st.exp[CPL], first, end)?;
        b.exp_bits[CPL] = 4 + 7 * n as u32;
    }
    for ch in 0..nf {
        if b.chexpstr[ch] == 0 {
            continue;
        }
        st.end[ch] = match b.chbwcod[ch] {
            Some(bw) => 37 + 3 * (bw as usize + 12),
            None => st.start[CPL],
        };
        let end = st.end[ch];
        let size = grpsize(b.chexpstr[ch]);
        let n = match size {
            1 => (end - 1) / 3,
            2 => (end - 1 + 3) / 6,
            _ => (end - 1 + 9) / 12,
        };
        let abs = br.read_u32(4)? as i32;
        st.exp[ch][0] = abs as u8;
        exponents(br, abs, n, size, &mut st.exp[ch], 1, end)?;
        br.skip(2)?; // gainrng
        b.exp_bits[ch] = 4 + 7 * n as u32 + 2;
    }
    if st.lfeon && b.lfeexpstr != 0 {
        st.end[LFE] = 7;
        let abs = br.read_u32(4)? as i32;
        st.exp[LFE][0] = abs as u8;
        exponents(br, abs, 2, 1, &mut st.exp[LFE], 1, 7)?;
        b.exp_bits[LFE] = 18;
    }
    if br.read_u32(1)? != 0 {
        for (v, n) in st.bai.iter_mut().zip([2, 2, 2, 2, 3]) {
            *v = br.read_u32(n)? as u8;
        }
        b.baie = Some(st.bai);
    }
    if br.read_u32(1)? != 0 {
        st.csnr = br.read_u32(6)? as u8;
        if st.cplinu {
            st.fsnr[CPL] = br.read_u32(4)? as u8;
            st.fgain[CPL] = br.read_u32(3)? as u8;
        }
        for ch in 0..nf {
            st.fsnr[ch] = br.read_u32(4)? as u8;
            st.fgain[ch] = br.read_u32(3)? as u8;
        }
        if st.lfeon {
            st.fsnr[LFE] = br.read_u32(4)? as u8;
            st.fgain[LFE] = br.read_u32(3)? as u8;
        }
        b.snr = Some((st.csnr, st.fsnr, st.fgain));
    }
    if st.cplinu && br.read_u32(1)? != 0 {
        st.leak = (br.read_u32(3)? as u8, br.read_u32(3)? as u8);
        b.leak = Some(st.leak);
    }
    b.deltbaie = br.read_u32(1)? != 0;
    if b.deltbaie {
        let cplbae = if st.cplinu { br.read_u32(2)? } else { 0 };
        let mut bae = [0u32; 5];
        for v in bae.iter_mut().take(nf) {
            *v = br.read_u32(2)?;
        }
        let mut slots = vec![];
        if st.cplinu {
            slots.push((CPL, cplbae));
        }
        slots.extend((0..nf).map(|ch| (ch, bae[ch])));
        for (slot, e) in slots {
            match e {
                1 => {
                    let n = br.read_u32(3)? + 1;
                    st.delt[slot].clear();
                    for _ in 0..n {
                        let seg = (br.read_u32(5)? as u8, br.read_u32(4)? as u8);
                        st.delt[slot].push((seg.0, seg.1, br.read_u32(3)? as u8));
                    }
                }
                2 => st.delt[slot].clear(),
                _ => {}
            }
        }
    } else if blk == 0 {
        st.delt.iter_mut().for_each(Vec::clear);
    }
    if br.read_u32(1)? != 0 {
        let n = br.read_u32(9)?;
        br.skip(8 * n)?;
        b.skipl = Some(n);
    }
    b.mant_start = br.bit_position() + 40;
    let zero = st.csnr == 0
        && (0..nf).all(|c| st.fsnr[c] == 0)
        && (!st.cplinu || st.fsnr[CPL] == 0)
        && (!st.lfeon || st.fsnr[LFE] == 0);
    let mut groups = Groups::default();
    let mut got_cpl = false;
    let mut order: Vec<usize> = Vec::new();
    for ch in 0..nf {
        order.push(ch);
        if st.cplinu && st.chincpl[ch] && !got_cpl {
            order.push(CPL);
            got_cpl = true;
        }
    }
    if st.lfeon {
        order.push(LFE);
    }
    for slot in order {
        if zero {
            continue;
        }
        let baps = bap(st, slot);
        let mut bits = 0;
        for &v in &baps[st.start[slot]..st.end[slot].min(256)] {
            bits += groups.bits(v);
        }
        br.skip(bits)?;
        b.mant_bits[slot] = bits;
    }
    b.end = br.bit_position() + 40;
    Ok(b)
}

fn expstr_char(s: u8) -> char {
    ['R', '1', '2', '4'][s as usize & 3]
}

#[derive(Default)]
struct Summary {
    frames: usize,
    patterns: Vec<BTreeMap<String, usize>>,
    cpl: BTreeMap<String, usize>,
    cplcoe: BTreeMap<String, usize>,
    blksw: [usize; 5],
    params: BTreeMap<String, usize>,
    bw: Vec<BTreeMap<u8, usize>>,
    csnr: BTreeMap<u8, usize>,
    fsnr: Vec<BTreeMap<(u8, u8), usize>>,
    exp_bits: [u64; 7],
    mant_bits: [u64; 7],
    skips: usize,
    remat: BTreeMap<String, usize>,
    late5: usize,
    over01: usize,
    mismatches: usize,
}

fn histogram<K: std::fmt::Debug>(m: &BTreeMap<K, usize>, top: usize) -> String {
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    v.iter()
        .take(top)
        .map(|(k, n)| format!("{k:?}×{n}"))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn dump(path: &str, summary_only: bool) {
    let data = std::fs::read(path).unwrap();
    let mut pos = 0;
    let mut sum = Summary::default();
    let mut nf = 0;
    let mut lfeon = false;
    let mut out = String::new();
    while pos + 7 < data.len() {
        let Ok(si) = syncinfo::parse(&data[pos..]) else {
            pos += 1;
            continue;
        };
        let len = si.frame_length as usize;
        let frame = &data[pos..(pos + len).min(data.len())];
        let b = bsi::parse(&frame[5..]).unwrap();
        nf = b.nfchans as usize;
        lfeon = b.lfeon;
        let mut st = State::new(nf, lfeon, b.acmod);
        let mut br = BitReader::new(&frame[5..]);
        br.skip(b.bits_consumed as u32).unwrap();
        let mut blocks = Vec::new();
        for blk in 0..6 {
            match block(&mut st, &mut br, blk) {
                Ok(x) => blocks.push(x),
                Err(e) => {
                    writeln!(out, "frame {}: block {blk} does not parse: {e}", sum.frames).unwrap();
                    break;
                }
            }
        }
        if let Ok(ox) = audblk::parse_frame_side_info(&si, &b, frame) {
            for (k, (o, m)) in ox.iter().zip(&blocks).enumerate() {
                let same = o.blksw[..nf] == m.blksw[..nf]
                    && o.chexpstr[..nf] == m.chexpstr[..nf]
                    && o.cplexpstr == m.cplexpstr
                    && o.lfeexpstr == m.lfeexpstr
                    && o.baie == m.baie.is_some()
                    && o.snroffste == m.snr.is_some()
                    && o.deltbaie == m.deltbaie
                    && o.skiple == m.skipl.is_some()
                    && o.cplcoe[..nf] == m.cplcoe[..nf];
                if !same {
                    sum.mismatches += 1;
                    writeln!(
                        out,
                        "frame {}: block {k} disagrees with oxideav",
                        sum.frames
                    )
                    .unwrap();
                }
            }
        }
        let crc1_end = {
            let words = len / 2;
            16 * ((words >> 1) + (words >> 3))
        } as u64;
        if blocks.len() == 6 {
            if blocks[5].mant_start < crc1_end {
                sum.late5 += 1;
            }
            if blocks[1].end > crc1_end {
                sum.over01 += 1;
            }
        }
        if !summary_only {
            writeln!(
                out,
                "frame {} ({} bytes, 5/8 at bit {crc1_end})",
                sum.frames, len
            )
            .unwrap();
        }
        // a channel's exponent strategies over the frame, cpl and LFE last
        let mut pat = vec![String::new(); nf + 2];
        let mut coe = String::new();
        for (k, x) in blocks.iter().enumerate() {
            for (ch, p) in pat.iter_mut().enumerate().take(nf) {
                p.push(expstr_char(x.chexpstr[ch]));
                if x.blksw[ch] {
                    sum.blksw[ch] += 1;
                }
                if let Some(bw) = x.chbwcod[ch] {
                    if sum.bw.len() < nf {
                        sum.bw.resize(nf, BTreeMap::new());
                    }
                    *sum.bw[ch].entry(bw).or_default() += 1;
                }
            }
            if st.cplinu {
                pat[nf].push(expstr_char(x.cplexpstr));
            }
            if lfeon {
                pat[nf + 1].push(expstr_char(x.lfeexpstr));
            }
            coe.push(if x.cplcoe.iter().any(|c| *c) {
                'c'
            } else {
                '.'
            });
            if x.skipl.is_some() {
                sum.skips += 1;
            }
            if let Some((inu, beg, end, strc)) = &x.cplstr {
                *sum.cpl
                    .entry(format!("{inu} {beg}-{end} {strc}"))
                    .or_default() += 1;
            }
            if let Some(r) = &x.remat {
                let s: String = r.iter().map(|v| if *v { '1' } else { '0' }).collect();
                *sum.remat.entry(s).or_default() += 1;
            }
            if let Some(p) = x.baie {
                *sum.params.entry(format!("baie {p:?}")).or_default() += 1;
            }
            if let Some(l) = x.leak {
                *sum.params.entry(format!("cplleak {l:?}")).or_default() += 1;
            }
            if let Some((c, f, g)) = x.snr {
                *sum.csnr.entry(c).or_default() += 1;
                if sum.fsnr.len() < 7 {
                    sum.fsnr.resize(7, BTreeMap::new());
                }
                for s in 0..7 {
                    *sum.fsnr[s].entry((f[s], g[s])).or_default() += 1;
                }
            }
            for s in 0..7 {
                sum.exp_bits[s] += u64::from(x.exp_bits[s]);
                sum.mant_bits[s] += u64::from(x.mant_bits[s]);
            }
            if x.deltbaie {
                *sum.params.entry("deltbaie".into()).or_default() += 1;
            }
            if summary_only {
                continue;
            }
            let flags = |v: &[bool]| {
                v[..nf]
                    .iter()
                    .map(|b| if *b { '1' } else { '0' })
                    .collect::<String>()
            };
            write!(out, "  b{k} sw {} dith {}", flags(&x.blksw), flags(&x.dith)).unwrap();
            if let Some((inu, beg, end, strc)) = &x.cplstr {
                write!(out, " cplinu {} {beg}-{end} strc {strc}", u8::from(*inu)).unwrap();
            }
            if st.cplinu {
                write!(
                    out,
                    " cplcoe {} cplexp {}",
                    flags(&x.cplcoe),
                    expstr_char(x.cplexpstr)
                )
                .unwrap();
            }
            let es: String = x.chexpstr[..nf].iter().map(|e| expstr_char(*e)).collect();
            write!(out, " exp {es}").unwrap();
            if lfeon {
                write!(out, " lfe {}", expstr_char(x.lfeexpstr)).unwrap();
            }
            let bw: Vec<String> = x.chbwcod[..nf]
                .iter()
                .map(|b| b.map_or("-".into(), |v| v.to_string()))
                .collect();
            write!(out, " bw {}", bw.join(",")).unwrap();
            if let Some(r) = &x.remat {
                write!(out, " remat {}", flags_any(r)).unwrap();
            }
            if let Some(p) = x.baie {
                write!(out, " baie {p:?}").unwrap();
            }
            if let Some((c, f, g)) = x.snr {
                write!(out, " csnr {c} fsnr {:?} fgain {:?}", &f[..nf], &g[..nf]).unwrap();
                if st.cplinu {
                    write!(out, " cpl {}/{}", f[CPL], g[CPL]).unwrap();
                }
                if lfeon {
                    write!(out, " lfe {}/{}", f[LFE], g[LFE]).unwrap();
                }
            }
            if let Some(l) = x.leak {
                write!(out, " leak {l:?}").unwrap();
            }
            if x.deltbaie {
                write!(out, " deltbaie").unwrap();
            }
            if let Some(n) = x.skipl {
                write!(out, " skipl {n}").unwrap();
            }
            writeln!(out).unwrap();
            for ch in 0..nf {
                if let Some((m, co)) = &x.cplco[ch] {
                    let v: Vec<String> = co.iter().map(|(e, m)| format!("{e}:{m}")).collect();
                    writeln!(out, "     cplco ch{ch} mstr {m} {}", v.join(" ")).unwrap();
                }
            }
            writeln!(
                out,
                "     bits exp {:?} mant {:?} mant@{} end@{}",
                x.exp_bits, x.mant_bits, x.mant_start, x.end
            )
            .unwrap();
        }
        for (ch, p) in pat.iter().enumerate() {
            if p.is_empty() {
                continue;
            }
            if sum.patterns.len() < nf + 2 {
                sum.patterns.resize(nf + 2, BTreeMap::new());
            }
            *sum.patterns[ch].entry(p.clone()).or_default() += 1;
        }
        *sum.cplcoe.entry(coe).or_default() += 1;
        sum.frames += 1;
        pos += len;
    }
    let names: &[&str] = if nf == 2 {
        &["L", "R"]
    } else {
        &["L", "C", "R", "Ls", "Rs"]
    };
    let label = |s: usize| match s {
        CPL => "cpl".to_string(),
        LFE => "lfe".to_string(),
        c => names.get(c).unwrap_or(&"?").to_string(),
    };
    writeln!(
        out,
        "== {path}: {} frames, {nf} fbw, lfe {lfeon}",
        sum.frames
    )
    .unwrap();
    for (ch, p) in sum.patterns.iter().enumerate() {
        if p.is_empty() {
            continue;
        }
        let slot = if ch < nf {
            ch
        } else if ch == nf {
            CPL
        } else {
            LFE
        };
        writeln!(out, "expstr {:>3}: {}", label(slot), histogram(p, 6)).unwrap();
    }
    writeln!(out, "coupling: {}", histogram(&sum.cpl, 4)).unwrap();
    writeln!(out, "cplcoe by block: {}", histogram(&sum.cplcoe, 4)).unwrap();
    for (ch, bw) in sum.bw.iter().enumerate() {
        if !bw.is_empty() {
            writeln!(out, "chbwcod {:>3}: {}", label(ch), histogram(bw, 4)).unwrap();
        }
    }
    writeln!(out, "params: {}", histogram(&sum.params, 6)).unwrap();
    writeln!(out, "csnroffst: {}", histogram(&sum.csnr, 8)).unwrap();
    for (s, f) in sum.fsnr.iter().enumerate() {
        let used = s < nf || (s == CPL && !sum.cpl.is_empty()) || (s == LFE && lfeon);
        if used {
            writeln!(out, "fsnr/fgain {:>3}: {}", label(s), histogram(f, 6)).unwrap();
        }
    }
    writeln!(out, "rematflg: {}", histogram(&sum.remat, 4)).unwrap();
    writeln!(out, "blksw per channel: {:?}", &sum.blksw[..nf]).unwrap();
    let n = sum.frames.max(1) as u64;
    let mut slots: Vec<usize> = (0..nf).collect();
    slots.extend([CPL, LFE]);
    let per: Vec<String> = slots
        .iter()
        .map(|&s| {
            format!(
                "{} {}/{}",
                label(s),
                sum.exp_bits[s] / n,
                sum.mant_bits[s] / n
            )
        })
        .collect();
    writeln!(
        out,
        "bits per frame (exponents/mantissas): {}",
        per.join("  ")
    )
    .unwrap();
    writeln!(
        out,
        "skip fields {}; block 5 mantissas before 5/8: {} frames; blocks 0-1 past 5/8: {}; \
         disagreements with oxideav: {}",
        sum.skips, sum.late5, sum.over01, sum.mismatches
    )
    .unwrap();
    print!("{out}");
}

fn flags_any(v: &[bool]) -> String {
    v.iter().map(|b| if *b { '1' } else { '0' }).collect()
}
