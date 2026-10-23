//! Object audio metadata (TS 103 420 clause 5.5): the program assignment from the bed, then
//! per update block its timing and per object the active flag, gain, priority, position
//! and size. Bed objects send no render info; their label places them (clause 5.2.1.4).
//! Zone constraints are off, elevation on, screen reference and snap off; the gain is 0 dB
//! and the priority the default, since the host's gain rides in the essence.

use super::emdf::variable_bits;
use crate::bits::{BitWriter, Overflow};

/// Update blocks a frame carries at most (`num_obj_info_blocks_bits`, 3 bits).
pub(crate) const MAX_BLOCKS: usize = 8;
/// Objects a program names at most: 16 elements and the LFE.
pub(crate) const MAX_OBJECTS: usize = 17;
/// Samples an update ramps over: 10 ms, one host period (Table 25, index 5).
pub(crate) const RAMP: u16 = 480;
/// Samples the frame advances `frame_offset` by (clause 5.3.2).
const FRAME: usize = 1536;
/// Ramp durations by `ramp_duration_idx` (Table 25).
const RAMPS: [u16; 16] = [
    32, 64, 128, 256, 320, 480, 1000, 1001, 1024, 1600, 1601, 1602, 1920, 2000, 2002, 2048,
];

/// How the program names its objects (clause 5.5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Assignment {
    /// `b_dyn_object_only_program`: dynamic objects, after the LFE when `lfe`.
    Dynamic { lfe: bool },
    /// One bed by `bed_channel_assignment[]` (Table 12), 10 bits, L/R the first.
    Standard(u16),
    /// One bed by `nonstd_bed_channel_assignment[]` (Table 13), 17 bits, L the first.
    NonStandard(u32),
}

/// The program, fixed for the stream: its assignment, then how many objects come first
/// with a fixed place (the bed's labels, or the dynamic program's LFE) and how many
/// dynamic objects follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Program {
    pub(crate) assignment: Assignment,
    pub(crate) fixed: usize,
    pub(crate) dynamic: usize,
}

impl Program {
    pub(crate) fn objects(&self) -> usize {
        self.fixed + self.dynamic
    }
}

/// One object's coded state in an update block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Meta {
    pub(crate) active: bool,
    /// `pos3D_X_bits` and `pos3D_Y_bits` (0 to 62), then the signed `pos3D_Z` (−15 to 15).
    pub(crate) pos: [i8; 3],
    /// `object_size_bits` (0 to 31); 0 is a point.
    pub(crate) size: u8,
}

/// Rounds half up, for a value the caller has bounded.
fn round(v: f32) -> i8 {
    (v + 0.5).floor() as i8
}

impl Meta {
    /// From the crate's cube (+x right, +y front, +z up, each −1..1) to the room's
    /// (clause 4.2.1: x 0 at the left wall, y 0 at the front, z −1 at the floor), on the
    /// grids of clauses 5.6.1.1.8 to 5.6.1.1.11 and 5.6.1.2.2. Inputs are finite and bounded.
    pub(crate) fn new(position: [f32; 3], size: f32) -> Meta {
        let [x, y, z] = position.map(|v| v.clamp(-1.0, 1.0));
        Meta {
            active: true,
            pos: [
                round((x + 1.0) * 31.0),
                round((1.0 - y) * 31.0),
                round(z * 15.0),
            ],
            size: round(size.clamp(0.0, 1.0) * 31.0) as u8,
        }
    }
}

/// One frame's update blocks: their start in the frame and, per block and object, the
/// state an update set. Block 0 starts the frame and holds every object.
pub(crate) struct Frame {
    start: [u16; MAX_BLOCKS],
    blocks: usize,
    upd: [[Option<Meta>; MAX_OBJECTS]; MAX_BLOCKS],
    /// Each object's state after the last update so far.
    last: [Meta; MAX_OBJECTS],
}

impl Frame {
    pub(crate) fn new() -> Frame {
        let mut f = Frame {
            start: [0; MAX_BLOCKS],
            blocks: 1,
            upd: [[None; MAX_OBJECTS]; MAX_BLOCKS],
            last: [Meta::default(); MAX_OBJECTS],
        };
        f.begin();
        f
    }

    /// Opens the next frame: one block at sample 0 with every object as the last left it.
    pub(crate) fn begin(&mut self) {
        self.blocks = 1;
        self.upd = [[None; MAX_OBJECTS]; MAX_BLOCKS];
        for (u, m) in self.upd[0].iter_mut().zip(self.last) {
            *u = Some(m);
        }
    }

    pub(crate) fn last(&self, object: usize) -> Meta {
        self.last.get(object).copied().unwrap_or_default()
    }

    /// Object `object` takes `m` at `offset` samples into the frame, on the 32-sample grid of
    /// `block_offset_factor`. Past eight blocks, an update joins the block before it.
    pub(crate) fn update(&mut self, object: usize, offset: usize, m: Meta) {
        if object >= MAX_OBJECTS || self.last[object] == m {
            return;
        }
        self.last[object] = m;
        let at = (offset.min(FRAME - 1) / 32 * 32) as u16;
        let n = self.blocks;
        let b = match self.start[..n].iter().position(|s| *s >= at) {
            Some(b) if self.start[b] == at => b,
            Some(b) if n < MAX_BLOCKS => {
                self.start.copy_within(b..n, b + 1);
                self.upd.copy_within(b..n, b + 1);
                (self.start[b], self.upd[b]) = (at, [None; MAX_OBJECTS]);
                self.blocks += 1;
                b
            }
            Some(b) => b - 1,
            None if n < MAX_BLOCKS => {
                (self.start[n], self.upd[n]) = (at, [None; MAX_OBJECTS]);
                self.blocks += 1;
                n
            }
            None => n - 1,
        };
        self.upd[b][object] = Some(m);
        // a later block that set this object would undo the update: it now holds `m` too
        for later in self.upd[b + 1..n.max(b + 1)].iter_mut() {
            if later[object].is_some() {
                later[object] = Some(m);
            }
        }
    }

    /// Merges the last two blocks into one, the later states kept; false at one block.
    pub(crate) fn merge(&mut self) -> bool {
        if self.blocks < 2 {
            return false;
        }
        let n = self.blocks;
        for o in 0..MAX_OBJECTS {
            if let Some(m) = self.upd[n - 1][o] {
                self.upd[n - 2][o] = Some(m);
            }
        }
        self.blocks -= 1;
        true
    }

    /// Writes `object_audio_metadata_payload` (clause 5.5.2) into `out`, its length in bytes.
    pub(crate) fn write(&self, out: &mut [u8], p: &Program) -> Result<usize, Overflow> {
        let mut scratch = [0u8; 2048];
        let mut s = BitWriter::new(&mut scratch);
        self.object_element(&mut s, p)?;
        // b_discard_unknown_element, the element, its padding
        let element = (1 + s.position()).div_ceil(8);

        let mut w = BitWriter::new(out);
        w.put(0, 2)?; // oa_md_version_bits
        let count = p.objects().saturating_sub(1) as u32;
        if count < 31 {
            w.put(count, 5)?;
        } else {
            w.put(31, 5)?;
            w.put(count - 31, 7)?;
        }
        program_assignment(&mut w, p)?;
        w.put(0, 1)?; // b_alternate_object_data_present
        w.put(1, 4)?; // oa_element_count_bits
        w.put(1, 4)?; // oa_element_id_idx: object_element
        variable_bits(&mut w, element as u32 - 1, 4)?;
        let at = w.position();
        w.put(0, 1)?; // b_discard_unknown_element
        self.object_element(&mut w, p)?;
        w.zeros(at + 8 * element - w.position())?;
        Ok(w.finish())
    }

    /// `object_element` (clause 5.5.5): the timing of every block, then every object's blocks.
    fn object_element(&self, w: &mut BitWriter, p: &Program) -> Result<(), Overflow> {
        let n = self.blocks;
        w.put(0, 2)?; // sample_offset_code: 0
        w.put(n as u32 - 1, 3)?;
        for b in 0..n {
            let end = if b + 1 < n {
                self.start[b + 1]
            } else {
                FRAME as u16
            };
            w.put(u32::from(self.start[b] / 32), 6)?;
            ramp(w, RAMP.min(end - self.start[b]))?;
        }
        w.put(1, 1)?; // b_reserved_data_not_present
        for o in 0..p.objects().min(MAX_OBJECTS) {
            let fixed = o < p.fixed;
            let mut cur = Meta::default();
            // the position a differential one counts from: the last coded while active
            let mut from: Option<[i8; 3]> = None;
            for b in 0..n {
                let upd = self.upd[b][o];
                if let Some(m) = upd {
                    cur = m;
                }
                w.put(u32::from(!cur.active), 1)?; // b_object_not_active
                if cur.active {
                    if b > 0 {
                        // object_basic_info_status_idx: full update or reuse
                        w.put(if upd.is_some() { 0b01 } else { 0b10 }, 2)?;
                    }
                    if b == 0 || upd.is_some() {
                        w.put(0b001, 3)?; // object_gain_idx 0 dB, b_default_object_priority
                    }
                    if !fixed {
                        if b > 0 {
                            w.put(if upd.is_some() { 0b01 } else { 0b10 }, 2)?;
                        }
                        if b == 0 || upd.is_some() {
                            render_info(w, &cur, if b > 0 { from } else { None })?;
                        }
                        from = Some(cur.pos);
                    }
                } else {
                    from = None;
                }
                w.put(0, 1)?; // b_additional_table_data_exists
            }
        }
        Ok(())
    }
}

/// `ramp_duration_code` and what follows it (clause 5.5.7, Tables 24 and 25).
fn ramp(w: &mut BitWriter, samples: u16) -> Result<(), Overflow> {
    match samples {
        0 => w.put(0, 2),
        512 => w.put(1, 2),
        1536 => w.put(2, 2),
        _ => match RAMPS.iter().position(|r| *r == samples) {
            Some(i) => w.put(0b111 << 4 | i as u32, 7),
            None => {
                w.put(0b110, 3)?;
                w.put(u32::from(samples.min(2047)), 11)
            }
        },
    }
}

/// `program_assignment` (clause 5.5.3): one bed, no ISF, then the dynamic objects.
fn program_assignment(w: &mut BitWriter, p: &Program) -> Result<(), Overflow> {
    match p.assignment {
        Assignment::Dynamic { lfe } => return w.put(0b10 | u32::from(lfe), 2),
        Assignment::Standard(mask) => {
            w.put(0, 1)?;
            // content_description: bed objects, dynamic objects if any (Table 11a)
            w.put(0b1000 | u32::from(p.dynamic > 0) << 1, 4)?;
            // b_bed_chan_distribute, b_multiple_bed_instances_present, b_lfe_only,
            // b_standard_chan_assign
            w.put(0b0001, 4)?;
            w.put(u32::from(mask), 10)?;
        }
        Assignment::NonStandard(mask) => {
            w.put(0, 1)?;
            w.put(0b1000 | u32::from(p.dynamic > 0) << 1, 4)?;
            w.put(0b0000, 4)?;
            w.put(mask, 17)?;
        }
    }
    if p.dynamic > 0 {
        w.put(p.dynamic as u32 - 1, 5)?; // num_dynamic_objects_bits
    }
    Ok(())
}

/// A full `object_render_info` (clause 5.5.11): the position, differential from `from`
/// when every step fits three bits, no distance, no zones, elevation on, the size, no
/// screen reference, no snap.
fn render_info(w: &mut BitWriter, m: &Meta, from: Option<[i8; 3]>) -> Result<(), Overflow> {
    let diff = from.map(|f| [0, 1, 2].map(|i| m.pos[i] - f[i]));
    match diff {
        Some(d) if d.iter().all(|v| (-4..4).contains(v)) => {
            w.put(1, 1)?;
            for v in d {
                w.put(v as u32 & 7, 3)?;
            }
        }
        _ => {
            if from.is_some() {
                w.put(0, 1)?; // b_differential_position_specified
            }
            w.put(m.pos[0] as u32, 6)?;
            w.put(m.pos[1] as u32, 6)?;
            w.put(u32::from(m.pos[2] >= 0), 1)?;
            w.put(u32::from(m.pos[2].unsigned_abs()), 4)?;
        }
    }
    w.put(0, 1)?; // b_object_distance_specified
    w.put(0b0001, 4)?; // zone_constraints_idx, b_enable_elevation
    if m.size == 0 {
        w.put(0, 2)?;
    } else {
        w.put(0b01 << 5 | u32::from(m.size), 7)?;
    }
    w.put(0, 2) // b_object_use_screen_ref, b_object_snap
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_ac3::eac3::oamd::OamdDecoder;

    /// 7.1.4 by Table 12: L/R, C, LFE, Ls/Rs, Lb/Rb, Tfl/Tfr, Tbl/Tbr.
    const BED_714: u16 = 0b11_1111_0100;

    fn parse(f: &Frame, p: &Program) -> String {
        let mut out = [0u8; 512];
        let n = f.write(&mut out, p).unwrap();
        format!("{:?}", OamdDecoder::default().decode(&out[..n]).unwrap())
    }

    #[test]
    fn quantises_the_cube_onto_the_room() {
        let m = Meta::new([-1.0, 1.0, 0.0], 0.0);
        assert_eq!((m.pos, m.size), ([0, 0, 0], 0)); // front left, at the floor's height
        let m = Meta::new([1.0, -1.0, -1.0], 1.0);
        assert_eq!((m.pos, m.size), ([62, 62, -15], 31));
        assert_eq!(Meta::new([0.0, 0.0, 0.5], 0.5).pos, [31, 31, 8]);
    }

    #[test]
    fn oxideav_reads_a_bed_and_dynamic_objects_back() {
        let p = Program {
            assignment: Assignment::Standard(BED_714),
            fixed: 12,
            dynamic: 5,
        };
        let mut f = Frame::new();
        let bed = Meta {
            active: true,
            ..Meta::default()
        };
        for o in 0..12 {
            f.update(o, 0, bed);
        }
        for o in 12..17 {
            f.update(o, 0, Meta::new([0.1 * o as f32 - 1.5, 0.5, 0.25], 0.0));
        }
        f.begin();
        f.update(13, 100, Meta::new([0.0, 0.0, 0.0], 0.2)); // block at 96
        f.update(14, 700, Meta::new([0.9, -0.9, 1.0], 0.0)); // block at 672
        f.update(13, 130, Meta::new([0.03, 0.0, 0.0], 0.2)); // block at 128, differential
        let dbg = parse(&f, &p);
        // oxideav lists the bed's labels from bit 0 of the mask up: 11 beds and the LFE
        assert_eq!(dbg.matches("Bed").count(), 11, "{dbg}");
        assert_eq!(dbg.matches("Lfe").count(), 1, "{dbg}");
        assert_eq!(dbg.matches("Dynamic").count(), 5, "{dbg}");
        for start in [
            "start_sample: 0,",
            "start_sample: 96,",
            "start_sample: 128,",
        ] {
            assert!(dbg.contains(start), "{start}: {dbg}");
        }
        assert!(
            dbg.contains("start_sample: 672, ramp_duration: 480"),
            "{dbg}"
        );
        assert!(dbg.contains("ramp_duration: 32"), "{dbg}"); // 96 runs into 128
        assert!(dbg.contains("raw_position: [59, 59, 15]"), "{dbg}"); // 0.9, -0.9, 1.0
        assert!(dbg.contains("raw_position: [32, 31, 0]"), "{dbg}");
    }

    #[test]
    fn a_dynamic_program_puts_the_lfe_first_and_marks_ended_objects() {
        let p = Program {
            assignment: Assignment::Dynamic { lfe: true },
            fixed: 1,
            dynamic: 2,
        };
        let mut f = Frame::new();
        f.update(
            0,
            0,
            Meta {
                active: true,
                ..Meta::default()
            },
        );
        f.update(1, 0, Meta::new([0.0, 1.0, 0.0], 0.0));
        f.update(2, 0, Meta::new([0.0, -1.0, 0.0], 0.0));
        f.begin();
        f.update(2, 512, Meta::default());
        let dbg = parse(&f, &p);
        assert!(
            dbg.starts_with("OamdFrame { kinds: [Lfe, Dynamic, Dynamic]"),
            "{dbg}"
        );
        assert!(dbg.contains("active: false"), "{dbg}");
        let nonstd = Program {
            assignment: Assignment::NonStandard(1 << 16 | 1 << 14),
            fixed: 2,
            dynamic: 0,
        };
        assert!(parse(&Frame::new(), &nonstd).contains("kinds: [Bed, Bed]"));
    }

    #[test]
    fn a_ninth_update_joins_the_eighth_block() {
        let mut f = Frame::new();
        for k in 0..12 {
            f.update(0, 100 * k + 50, Meta::new([k as f32 / 12.0, 0.0, 0.0], 0.0));
        }
        assert_eq!(f.blocks, MAX_BLOCKS);
        assert_eq!(f.upd[7][0], Some(f.last(0)));
        while f.merge() {}
        assert_eq!((f.blocks, f.upd[0][0]), (1, Some(f.last(0))));
    }
}
