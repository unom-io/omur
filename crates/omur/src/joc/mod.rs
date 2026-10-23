//! Object audio (ETSI TS 103 420) over a 5.1 E-AC-3 core: a bed and dynamic objects, up to
//! 16 elements together, with the metadata that places them. Compiled only with
//! `--features joc`; no tag that contains this module is published before its release gate
//! clears (docs/design.md §8, §12).
//!
//! Per chunk the encoder renders the core and the elements, analyses every element and the
//! core in the QMF domain a timeslot at a time, and at each frame's end fits the matrices,
//! writes OAMD and JOC into one EMDF container and hands it to the core's next frame
//! through [`Encoder::set_extension`]. The core's mantissas get what the container leaves.

mod emdf;
mod matrix;
mod oamd;
mod qmf;
mod render;
mod tables;

use crate::{Blocks, Codec, Config, Encoder, Error, Frame, Layout};
use matrix::{CH, Joc, SLOTS};
use qmf::{Analysis, SUBBANDS};
use render::{CORE, Scene, TS};

pub(crate) use emdf::addbsi;

/// Elements one stream carries, bed channels and objects together.
pub const MAX_ELEMENTS: usize = 16;
/// Objects a host may push. Those beyond the element budget fold into their nearest kept
/// element (docs/design.md §8.3).
pub const MAX_OBJECTS: u8 = 32;

/// Samples per frame: TS 103 420 times its metadata in frames of 1,536 (clause 5.3.2).
const FRAME: usize = 1536;
/// Samples the core encoder holds before a frame completes: its last window's lookahead.
const LOOKAHEAD: usize = 256;

/// A bed channel, as a Windows static object names it. Bottom channels are not bed channels
/// in TS 103 420; a game that wants one gets an object at negative height. The back centre
/// has no bed label either (Tables 12, 13): it goes out as an object at its fixed place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BedChannel {
    FrontLeft,
    FrontRight,
    FrontCenter,
    Lfe,
    SideLeft,
    SideRight,
    BackLeft,
    BackRight,
    TopFrontLeft,
    TopFrontRight,
    TopBackLeft,
    TopBackRight,
    BackCenter,
}

impl BedChannel {
    /// Every channel, in the order `push_bed` interleaves them.
    pub const ALL: [BedChannel; 13] = [
        BedChannel::FrontLeft,
        BedChannel::FrontRight,
        BedChannel::FrontCenter,
        BedChannel::Lfe,
        BedChannel::SideLeft,
        BedChannel::SideRight,
        BedChannel::BackLeft,
        BedChannel::BackRight,
        BedChannel::TopFrontLeft,
        BedChannel::TopFrontRight,
        BedChannel::TopBackLeft,
        BedChannel::TopBackRight,
        BedChannel::BackCenter,
    ];
}

/// The bed's channel set. [`ObjectEncoder::push_bed`] takes samples interleaved in
/// [`BedChannel`] order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bed {
    mask: u16,
}

impl Bed {
    pub fn new(channels: &[BedChannel]) -> Bed {
        let mask = channels.iter().fold(0, |m, c| m | 1 << *c as u16);
        Bed { mask }
    }

    pub fn contains(&self, channel: BedChannel) -> bool {
        self.mask & (1 << channel as u16) != 0
    }

    pub fn channels(&self) -> usize {
        self.mask.count_ones() as usize
    }

    /// Elements the bed takes of the stream's [`MAX_ELEMENTS`]: every channel but the LFE,
    /// which the core carries directly.
    pub fn elements(&self) -> usize {
        self.channels() - usize::from(self.contains(BedChannel::Lfe))
    }
}

/// What an object encoder needs: the bed, how many objects the host may push, and the core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectConfig {
    pub bed: Bed,
    /// Objects the host may push, [`MAX_OBJECTS`] at most.
    pub objects: u8,
    /// The core's rate, from `Config::bitrates(Codec::Eac3 { .. }, Layout::Surround51)`.
    pub bitrate_kbps: u32,
    /// Six: TS 103 420's metadata counts frames of 1,536 samples (clause 5.3.2).
    pub blocks: Blocks,
}

/// Where an object is and how loud, from the sample `at_sample` of the pushed chunk on.
/// `position` is a listener-centred unit cube: +x right, +y front, +z up, each −1..1 with
/// 1 at the speaker wall. From Windows' metres (+z behind the listener): `[x, −z, y] / r`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectUpdate {
    pub position: [f32; 3],
    /// Linear gain, 0 to +15 dB; it scales the object's audio sample by sample.
    pub gain: f32,
    /// 0 is a point; 1 fills the room.
    pub size: f32,
    pub at_sample: u32,
}

/// The program assignment of `bed` with `dynamic` dynamic elements, and where its LFE
/// sits among the program's objects.
fn program(bed: Bed, dynamic: usize) -> (oamd::Program, Option<usize>) {
    use BedChannel::*;
    let has = |c| bed.contains(c);
    let (lfe, bc) = (has(Lfe), usize::from(has(BackCenter)));
    let named = bed.channels() - usize::from(lfe) - bc;
    if named == 0 {
        let p = oamd::Program {
            assignment: oamd::Assignment::Dynamic { lfe },
            fixed: usize::from(lfe),
            dynamic: bc + dynamic,
        };
        return (p, lfe.then_some(0));
    }
    // Table 12 by pair or single, and Table 13 bit by bit, MSB first
    let pairs = [
        (FrontLeft, FrontRight, 9),
        (SideLeft, SideRight, 6),
        (BackLeft, BackRight, 5),
        (TopFrontLeft, TopFrontRight, 4),
        (TopBackLeft, TopBackRight, 2),
    ];
    let standard = pairs.iter().all(|(a, b, _)| has(*a) == has(*b));
    let assignment = if standard {
        let mut mask = u16::from(has(FrontCenter)) << 8 | u16::from(lfe) << 7;
        for (a, _, bit) in pairs {
            mask |= u16::from(has(a)) << bit;
        }
        oamd::Assignment::Standard(mask)
    } else {
        let bits = [
            (FrontLeft, 16),
            (FrontRight, 15),
            (FrontCenter, 14),
            (Lfe, 13),
            (SideLeft, 12),
            (SideRight, 11),
            (BackLeft, 10),
            (BackRight, 9),
            (TopFrontLeft, 8),
            (TopFrontRight, 7),
            (TopBackLeft, 4),
            (TopBackRight, 3),
        ];
        oamd::Assignment::NonStandard(bits.iter().fold(0, |m, (c, b)| m | u32::from(has(*c)) << b))
    };
    let p = oamd::Program {
        assignment,
        fixed: named + usize::from(lfe),
        dynamic: bc + dynamic,
    };
    let lfe_at = lfe.then(|| {
        [FrontLeft, FrontRight, FrontCenter]
            .iter()
            .filter(|c| has(**c))
            .count()
    });
    (p, lfe_at)
}

/// Frames the core encoder has completed once it holds `pushed` samples.
fn completed(pushed: u64) -> u64 {
    pushed.saturating_sub(LOOKAHEAD as u64) / FRAME as u64
}

/// A bed plus objects into one E-AC-3 stream. Same rules as [`crate::Encoder`]: everything
/// allocated in [`ObjectEncoder::new`], sized by the bed and `objects`; nothing after.
pub struct ObjectEncoder {
    config: ObjectConfig,
    core: Encoder,
    scene: Scene,
    program: oamd::Program,
    meta: oamd::Frame,
    joc: Joc,
    /// The elements' analyses, then the core's L R C Ls Rs.
    qmf: Box<[Analysis]>,
    /// The timeslot being rendered: per element, and the core's L R C Ls Rs LFE.
    elem: Box<[[f32; TS]]>,
    core_ts: Box<[[f32; TS]; CORE + 1]>,
    /// The timeslot's QMF, real and imaginary: per element, and the core's five.
    zq: Box<[[[f32; SUBBANDS]; 2]]>,
    xq: Box<[[[f32; SUBBANDS]; 2]; CH]>,
    quiet: Box<[bool]>,
    events: Box<[Option<u8>]>,
    /// The chunk's rendered core, interleaved in WAV order, and how much of it the core
    /// encoder has.
    out: Box<[f32]>,
    out_len: usize,
    sent: usize,
    oamd_buf: Box<[u8]>,
    joc_buf: Box<[u8]>,
    emdf: Box<[u8]>,
    /// The most bytes a frame's container may take.
    room: usize,
    /// The open chunk, its length once known: the bed's, or the first object's.
    open: bool,
    len: Option<usize>,
    /// Samples rendered and handed to the core.
    pos: u64,
    unread: bool,
    /// Every element's signal as the matrices saw it, for the gate tests.
    #[cfg(test)]
    trace: Vec<Vec<f32>>,
    /// The core pushed, in WAV order, for the gate tests.
    #[cfg(test)]
    core_trace: Vec<f32>,
}

impl ObjectEncoder {
    /// Validates the configuration and allocates everything. [`Error::Layout`] for more
    /// than [`MAX_OBJECTS`] objects, a frame of other than six blocks, or nothing to carry;
    /// [`Error::Bitrate`] off the E-AC-3 5.1 ladder or where a frame cannot hold the
    /// metadata.
    pub fn new(config: ObjectConfig) -> Result<ObjectEncoder, Error> {
        if config.objects > MAX_OBJECTS || config.bed.elements() > MAX_ELEMENTS {
            return Err(Error::Layout);
        }
        let blocks = config.blocks;
        let core = Codec::Eac3 { blocks };
        let core_config = Config::new(core, Layout::Surround51, config.bitrate_kbps)?;
        let beds = config.bed.elements();
        let dynamic = usize::from(config.objects).min(MAX_ELEMENTS - beds);
        let elements = beds + dynamic;
        if blocks != Blocks::Six || elements == 0 {
            return Err(Error::Layout);
        }
        let (program, lfe_at) = program(config.bed, dynamic);
        let mut core = Encoder::objects(core_config, elements as u8);
        // the largest extension the stream takes, by bisection
        let zeros = [0u8; emdf::MAX];
        let (mut lo, mut hi) = (0, emdf::MAX);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if core.set_extension(&zeros[..mid]).is_ok() {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let _ = core.set_extension(&[]);
        let scene = Scene::new(config.bed, usize::from(config.objects), FRAME, lfe_at);
        let mut enc = ObjectEncoder {
            config,
            core,
            scene,
            program,
            meta: oamd::Frame::new(),
            joc: Joc::new(elements),
            qmf: (0..elements + CH).map(|_| Analysis::new()).collect(),
            elem: vec![[0.0; TS]; elements].into_boxed_slice(),
            core_ts: Box::new([[0.0; TS]; CORE + 1]),
            zq: vec![[[0.0; SUBBANDS]; 2]; elements].into_boxed_slice(),
            xq: Box::new([[[0.0; SUBBANDS]; 2]; CH]),
            quiet: vec![true; elements].into_boxed_slice(),
            events: vec![None; elements].into_boxed_slice(),
            out: vec![0.0; FRAME * 6].into_boxed_slice(),
            out_len: 0,
            sent: 0,
            oamd_buf: vec![0; 2 * emdf::MAX].into_boxed_slice(),
            joc_buf: vec![0; emdf::MAX].into_boxed_slice(),
            emdf: vec![0; emdf::MAX].into_boxed_slice(),
            room: lo,
            open: false,
            len: None,
            pos: 0,
            unread: false,
            #[cfg(test)]
            trace: vec![Vec::new(); elements],
            #[cfg(test)]
            core_trace: Vec::new(),
        };
        // the busiest first block: every object active, placed and sized
        let mut probe = oamd::Frame::new();
        for o in 0..program.objects() {
            probe.update(o, 0, oamd::Meta::new([0.4, 0.4, 0.4], 0.5));
        }
        probe.begin();
        let n = probe
            .write(&mut enc.oamd_buf, &program)
            .unwrap_or(usize::MAX / 2);
        if emdf::len(n, Joc::min_bytes(elements)) > enc.room {
            return Err(Error::Bitrate {
                kbps: config.bitrate_kbps,
            });
        }
        enc.scene.start(&mut enc.meta);
        enc.meta.begin();
        Ok(enc)
    }

    pub fn config(&self) -> ObjectConfig {
        self.config
    }

    /// Starts a chunk with the bed's samples, interleaved in [`BedChannel`] order, a frame
    /// (1,536 samples) at most; a bed without channels takes an empty slice and the first
    /// object sets the length. Every live object's [`ObjectEncoder::push_object`] for the
    /// chunk carries the same sample count. A chunk left unpulled is encoded first, and
    /// [`Error::Backlog`] refuses this one, consuming nothing, when that would complete a
    /// frame while the last is unread.
    pub fn push_bed(&mut self, pcm: &[f32]) -> Result<(), Error> {
        let channels = self.config.bed.channels();
        if !pcm.len().is_multiple_of(channels) {
            return Err(Error::Length);
        }
        let n = pcm.len().checked_div(channels);
        if n.is_some_and(|n| n > FRAME) {
            return Err(Error::Length);
        }
        if self.open {
            if self.unread && self.completes() {
                return Err(Error::Backlog);
            }
            self.flush();
        }
        self.scene.bed(pcm);
        self.open = true;
        self.len = n;
        Ok(())
    }

    /// One chunk of object `index`, mono, with the update that took effect inside it, if any.
    /// An object not pushed for a chunk is silent for it. [`Error::Length`] before
    /// `push_bed` opens a chunk or for a length other than the chunk's.
    pub fn push_object(
        &mut self,
        index: u8,
        pcm: &[f32],
        update: Option<ObjectUpdate>,
    ) -> Result<(), Error> {
        if index >= self.config.objects {
            return Err(Error::Object { index });
        }
        if !self.open || pcm.len() > FRAME || self.len.is_some_and(|n| n != pcm.len()) {
            return Err(Error::Length);
        }
        self.len = Some(pcm.len());
        self.scene.object(usize::from(index), pcm, update);
        Ok(())
    }

    /// The object stopped; its element is free and it is silent until pushed again.
    pub fn end_object(&mut self, index: u8) {
        self.scene.end(usize::from(index));
    }

    /// The frame the chunk completed, once. A frame still unread when the chunk would
    /// complete the next comes out first, and the chunk with the next pull.
    pub fn pull(&mut self) -> Option<Frame<'_>> {
        if self.open && !(self.unread && self.completes()) {
            self.flush();
        }
        self.unread = false;
        self.core.pull()
    }

    /// The rendered 5.1 core of the last encoded chunk, interleaved in WAV order: the
    /// reference the probe scores the decoded core against.
    #[doc(hidden)]
    pub fn core_pcm(&self) -> &[f32] {
        &self.out[..6 * self.out_len]
    }

    /// Whether the open chunk completes a core frame.
    fn completes(&self) -> bool {
        let n = self.len.unwrap_or(0) as u64;
        completed(self.pos + n) > completed(self.pos)
    }

    /// Encodes the open chunk.
    fn flush(&mut self) {
        self.open = false;
        let Some(n) = self.len else {
            return;
        };
        if self.completes() {
            self.unread = true;
        }
        let offset = (self.pos % FRAME as u64) as usize;
        self.scene.publish(offset, &mut self.meta);
        (self.out_len, self.sent) = (0, 0);
        let mut i = 0;
        while i < n {
            let at = (self.pos % TS as u64) as usize;
            let take = (TS - at).min(n - i);
            let offset = (self.pos % FRAME as u64) as usize;
            self.scene.render(
                i,
                take,
                at,
                offset,
                &mut self.elem,
                &mut self.core_ts,
                &mut self.meta,
            );
            // the core as the decoder sees it: clamped, in WAV order FL FR FC LFE BL BR
            for s in at..at + take {
                for c in self.core_ts.iter_mut() {
                    c[s] = c[s].clamp(-1.0, 1.0);
                }
                let t = &self.core_ts;
                let wav = [t[0][s], t[1][s], t[2][s], t[5][s], t[3][s], t[4][s]];
                if let Some(dst) = self.out.get_mut(6 * self.out_len..6 * self.out_len + 6) {
                    dst.copy_from_slice(&wav);
                }
                self.out_len += 1;
            }
            i += take;
            self.pos += take as u64;
            if self.pos.is_multiple_of(TS as u64) {
                self.timeslot();
            }
        }
        self.push_out();
    }

    /// Hands the core rendered since the last push to the core encoder.
    fn push_out(&mut self) {
        if let Some(pcm) = self.out.get(6 * self.sent..6 * self.out_len) {
            let _ = self.core.push(pcm);
            #[cfg(test)]
            self.core_trace.extend_from_slice(pcm);
        }
        self.sent = self.out_len;
    }

    /// The timeslot is rendered: analyse it, and at a frame's end code the frame.
    fn timeslot(&mut self) {
        let e = self.elem.len();
        #[cfg(test)]
        for (t, s) in self.trace.iter_mut().zip(self.elem.iter()) {
            t.extend_from_slice(s);
        }
        for (k, (a, z)) in self
            .qmf
            .iter_mut()
            .zip(self.zq.iter_mut())
            .take(e)
            .enumerate()
        {
            let [re, im] = z;
            a.process(&self.elem[k], re, im);
            self.quiet[k] = a.silent();
        }
        for (c, (a, x)) in self.qmf[e..].iter_mut().zip(self.xq.iter_mut()).enumerate() {
            let [re, im] = x;
            a.process(&self.core_ts[c], re, im);
        }
        let t = ((self.pos / TS as u64) % SLOTS as u64) as usize;
        let t = (t + SLOTS - 1) % SLOTS;
        self.scene.events(&mut self.events);
        self.joc
            .accumulate(t, &self.xq, &self.zq, &self.quiet, &self.events);
        for v in self.elem.iter_mut() {
            *v = [0.0; TS];
        }
        *self.core_ts = [[0.0; TS]; CORE + 1];
        if t == SLOTS - 1 {
            self.end_frame();
        }
    }

    /// Codes the frame: OAMD, squeezed until the JOC payload has its floor, then the
    /// matrices in what is left, into one container for the core's next frame.
    fn end_frame(&mut self) {
        self.push_out();
        let elements = self.elem.len();
        let oamd = loop {
            let n = self
                .meta
                .write(&mut self.oamd_buf, &self.program)
                .unwrap_or(usize::MAX / 2);
            if emdf::len(n, Joc::min_bytes(elements)) <= self.room || !self.meta.merge() {
                break n;
            }
        };
        let mut budget = self.room.saturating_sub(emdf::len(oamd, 0));
        while budget > 0 && emdf::len(oamd, budget) > self.room {
            budget -= 1;
        }
        let budget = budget.min(self.joc_buf.len());
        let joc = self.joc.encode(&mut self.joc_buf[..budget], budget);
        if let (Some(oamd_bytes), Ok(joc)) = (self.oamd_buf.get(..oamd), joc)
            && let Ok(n) = emdf::write(&mut self.emdf, oamd_bytes, &self.joc_buf[..joc])
        {
            let _ = self.core.set_extension(&self.emdf[..n]);
        }
        self.meta.begin();
        self.scene.begin_frame(&mut self.meta);
    }
}

#[cfg(test)]
mod gate;

#[cfg(test)]
mod tests {
    use super::*;

    fn bed_714() -> Bed {
        use BedChannel::*;
        Bed::new(&[
            FrontLeft,
            FrontRight,
            FrontCenter,
            Lfe,
            SideLeft,
            SideRight,
            BackLeft,
            BackRight,
            TopFrontLeft,
            TopFrontRight,
            TopBackLeft,
            TopBackRight,
        ])
    }

    #[test]
    fn a_714_bed_leaves_five_elements_for_objects() {
        let bed = bed_714();
        assert_eq!(bed.channels(), 12);
        assert_eq!(bed.elements(), 11);
        assert_eq!(MAX_ELEMENTS - bed.elements(), 5);
        let (p, lfe) = program(bed, 5);
        assert_eq!(p.assignment, oamd::Assignment::Standard(0b11_1111_0100));
        assert_eq!((p.fixed, p.dynamic, lfe), (12, 5, Some(3)));
    }

    #[test]
    fn construction_validates_the_core_and_the_object_cap() {
        let ok = ObjectConfig {
            bed: bed_714(),
            objects: 32,
            bitrate_kbps: 768,
            blocks: Blocks::Six,
        };
        assert!(ObjectEncoder::new(ok).is_ok());
        let too_many = ObjectConfig { objects: 33, ..ok };
        assert_eq!(ObjectEncoder::new(too_many).err(), Some(Error::Layout));
        let bad_rate = ObjectConfig {
            bitrate_kbps: 400,
            ..ok
        };
        assert_eq!(
            ObjectEncoder::new(bad_rate).err(),
            Some(Error::Bitrate { kbps: 400 })
        );
        let short = ObjectConfig {
            blocks: Blocks::Two,
            ..ok
        };
        assert_eq!(ObjectEncoder::new(short).err(), Some(Error::Layout));
        let empty = ObjectConfig {
            bed: Bed::new(&[BedChannel::Lfe]),
            objects: 0,
            ..ok
        };
        assert_eq!(ObjectEncoder::new(empty).err(), Some(Error::Layout));
    }

    #[test]
    fn beds_name_their_labels_and_odd_ones_go_non_standard() {
        use BedChannel::*;
        let (p, lfe) = program(Bed::new(&[FrontLeft, FrontCenter, Lfe]), 0);
        assert_eq!(
            p.assignment,
            oamd::Assignment::NonStandard(1 << 16 | 1 << 14 | 1 << 13)
        );
        assert_eq!(lfe, Some(2));
        let (p, lfe) = program(Bed::new(&[Lfe, BackCenter]), 3);
        assert_eq!(p.assignment, oamd::Assignment::Dynamic { lfe: true });
        assert_eq!((p.fixed, p.dynamic, lfe), (1, 4, Some(0)));
        let (p, lfe) = program(Bed::new(&[]), 16);
        assert_eq!((p.objects(), lfe), (16, None));
    }
}
