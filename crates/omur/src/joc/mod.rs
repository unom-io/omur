//! Object audio (ETSI TS 103 420) over a 5.1 E-AC-3 core: a bed and dynamic objects, up to
//! 16 elements together, with the metadata that places them. Compiled only with
//! `--features joc`; no tag that contains this module is published before its release gate
//! clears (docs/design.md §8, §12).

mod emdf;
mod matrix;
mod oamd;
mod qmf;
mod render;
#[allow(dead_code)]
mod tables;

use crate::{Blocks, Codec, Config, Error, Frame, Layout};

/// Elements one stream carries, bed channels and objects together.
pub const MAX_ELEMENTS: usize = 16;
/// Objects a host may push. Those beyond the element budget fold into their nearest kept
/// element (docs/design.md §8.3).
pub const MAX_OBJECTS: u8 = 32;

/// A bed channel, as a Windows static object names it. Bottom channels are not bed channels
/// in TS 103 420; a game that wants one gets an object at negative height.
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
    pub blocks: Blocks,
}

/// Where an object is and how loud, from the sample `at_sample` of the pushed chunk on.
/// `position` is a listener-centred unit cube: +x right, +y front, +z up, each −1..1 with
/// 1 at the speaker wall. From Windows' metres (+z behind the listener): `[x, −z, y] / r`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectUpdate {
    pub position: [f32; 3],
    /// Linear gain.
    pub gain: f32,
    /// 0 is a point; 1 fills the room.
    pub size: f32,
    pub at_sample: u32,
}

/// A bed plus objects into one E-AC-3 stream. Same rules as [`crate::Encoder`]: everything
/// allocated in [`ObjectEncoder::new`], sized by the bed and `objects`; nothing after.
pub struct ObjectEncoder {
    config: ObjectConfig,
}

impl ObjectEncoder {
    pub fn new(config: ObjectConfig) -> Result<ObjectEncoder, Error> {
        if config.objects > MAX_OBJECTS || config.bed.elements() > MAX_ELEMENTS {
            return Err(Error::Layout);
        }
        let core = Codec::Eac3 {
            blocks: config.blocks,
        };
        Config::new(core, Layout::Surround51, config.bitrate_kbps)?;
        Ok(ObjectEncoder { config })
    }

    pub fn config(&self) -> ObjectConfig {
        self.config
    }

    /// One chunk of the bed, interleaved in [`BedChannel`] order. Every live object's
    /// [`ObjectEncoder::push_object`] for the same chunk carries the same sample count.
    pub fn push_bed(&mut self, pcm: &[f32]) -> Result<(), Error> {
        if !pcm.len().is_multiple_of(self.config.bed.channels()) {
            return Err(Error::Length);
        }
        todo!("P3, docs/design.md §8")
    }

    /// One chunk of object `index`, mono, with the update that took effect inside it, if any.
    /// An object not pushed for a chunk is silent for it.
    pub fn push_object(
        &mut self,
        index: u8,
        pcm: &[f32],
        update: Option<ObjectUpdate>,
    ) -> Result<(), Error> {
        let _ = (pcm, update);
        if index >= self.config.objects {
            return Err(Error::Object { index });
        }
        todo!("P3, docs/design.md §8")
    }

    /// The object stopped; its element is free and it is silent until pushed again.
    pub fn end_object(&mut self, index: u8) {
        let _ = index;
        todo!("P3, docs/design.md §8")
    }

    /// The frame the chunk completed, once.
    pub fn pull(&mut self) -> Option<Frame<'_>> {
        todo!("P3, docs/design.md §8")
    }
}

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
    }
}
