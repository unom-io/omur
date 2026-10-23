//! The core renderer and the element budget (docs/design.md §8.3 steps 1 and 2).
//!
//! Every bed channel and every object is panned onto the 5.1 core at its own position:
//! amplitude panning over the floor ring with a square-root law, heights folded at −3 dB,
//! the LFE straight through, gains ramping sample by sample over 480 samples after each
//! update. The elements are what the JOC matrices rebuild: the bed's channels but the LFE
//! first, then objects. Past the budget the loudest objects keep an element and each other
//! folds into the nearest one, the keeper's metadata unchanged; the core never sees the
//! fold, so it cannot click, and an object that changes element crossfades over 512 samples.

use super::oamd::{self, Meta};
use super::{Bed, BedChannel, MAX_ELEMENTS, ObjectUpdate};
use core::f32::consts::FRAC_1_SQRT_2;

/// Core channels in JOC order (Table 53): L, R, C, Ls, Rs; the LFE is the sixth signal.
pub(crate) const CORE: usize = 5;
/// Samples per timeslot buffer.
pub(crate) const TS: usize = 64;
/// Samples a gain change ramps over, as the metadata's ramp.
const RAMP: u32 = oamd::RAMP as u32;
/// Samples a change of element crossfades over, from the frame's start.
const XFADE: u32 = 512;
/// The loudest gain an update sets: +15 dB, OAMD's ceiling (Table 19).
const GAIN_MAX: f32 = 5.623_413;
/// How much louder than the quietest keeper a folded object must be to take its element.
const HYSTERESIS: f32 = 2.0;

/// Core gains, L R C Ls Rs, of a point in the cube (+x right, +y front, +z up): the front
/// row L C R and the surround row Ls Rs each pan by x, the rows by y from the side (y 0)
/// to the front wall, everything behind the side on the surrounds; height lowers the
/// point by up to 3 dB. Each pair's gains are `√(1 − t)` and `√t`: power sums to one.
pub(crate) fn pan(p: [f32; 3]) -> [f32; CORE] {
    let [x, y, z] = p.map(|v| v.clamp(-1.0, 1.0));
    let f = y.max(0.0);
    let (front, rear) = (f.sqrt(), (1.0 - f).sqrt());
    let (l, c, r) = if x <= 0.0 {
        ((-x).sqrt(), (1.0 + x).sqrt(), 0.0)
    } else {
        (0.0, (1.0 - x).sqrt(), x.sqrt())
    };
    let t = (x + 1.0) * 0.5;
    let h = 1.0 - (1.0 - FRAC_1_SQRT_2) * z.max(0.0);
    [
        front * l * h,
        front * r * h,
        front * c * h,
        rear * (1.0 - t).sqrt() * h,
        rear * t.sqrt() * h,
    ]
}

/// Where a bed channel sits in the cube (Table B.10); none for the LFE.
pub(crate) fn position(c: BedChannel) -> Option<[f32; 3]> {
    use BedChannel::*;
    Some(match c {
        FrontLeft => [-1.0, 1.0, 0.0],
        FrontRight => [1.0, 1.0, 0.0],
        FrontCenter => [0.0, 1.0, 0.0],
        Lfe => return None,
        SideLeft => [-1.0, 0.0, 0.0],
        SideRight => [1.0, 0.0, 0.0],
        BackLeft => [-1.0, -1.0, 0.0],
        BackRight => [1.0, -1.0, 0.0],
        TopFrontLeft => [-1.0, 1.0, 1.0],
        TopFrontRight => [1.0, 1.0, 1.0],
        TopBackLeft => [-1.0, -1.0, 1.0],
        TopBackRight => [1.0, -1.0, 1.0],
        BackCenter => [0.0, -1.0, 0.0],
    })
}

/// A sample clamped to full scale; non-finite reads as silence.
fn sample(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// A bounded copy of a host value: non-finite reads as `zero`.
fn bounded(v: f32, lo: f32, hi: f32, zero: f32) -> f32 {
    if v.is_finite() { v.clamp(lo, hi) } else { zero }
}

/// Six gains, the essence's then its five core gains, each ramping linearly to its target.
#[derive(Clone, Copy)]
struct Ramp {
    cur: [f32; 6],
    step: [f32; 6],
    target: [f32; 6],
    left: u32,
}

impl Ramp {
    fn held(g: [f32; 6]) -> Ramp {
        Ramp {
            cur: g,
            step: [0.0; 6],
            target: g,
            left: 0,
        }
    }

    fn retarget(&mut self, g: [f32; 6]) {
        self.target = g;
        for ((s, g), c) in self.step.iter_mut().zip(g).zip(self.cur) {
            *s = (g - c) / RAMP as f32;
        }
        self.left = RAMP;
    }

    fn next(&mut self) -> [f32; 6] {
        if self.left > 0 {
            self.left -= 1;
            if self.left == 0 {
                self.cur = self.target;
            } else {
                for i in 0..6 {
                    self.cur[i] += self.step[i];
                }
            }
        }
        self.cur
    }
}

fn gains(gain: f32, p: [f32; 3]) -> [f32; 6] {
    let c = pan(p);
    [
        gain,
        gain * c[0],
        gain * c[1],
        gain * c[2],
        gain * c[3],
        gain * c[4],
    ]
}

struct Object {
    /// Pushed in the open chunk; `live` until `end_object`; `age` the samples rendered
    /// since it went live: an update within the first ramp takes effect at once.
    pushed: bool,
    live: bool,
    age: u32,
    /// The chunk's update: its offset in the chunk, position, gain, size.
    update: Option<(usize, [f32; 3], f32, f32)>,
    pos: [f32; 3],
    size: f32,
    ramp: Ramp,
    /// The essence's energy this frame, and last frame's.
    energy: f32,
    loud: f32,
    /// The element it feeds, and the one it fades out of over the frame's first `XFADE`.
    dest: Option<u8>,
    from: Option<u8>,
    fade: u32,
}

/// The scene: the bed, the objects and which element each feeds.
pub(crate) struct Scene {
    /// Per bed channel in `BedChannel` order: its element (none for the LFE) and core gains.
    bed: Box<[(Option<u8>, [f32; CORE])]>,
    bed_pcm: Box<[f32]>,
    objects: Box<[Object]>,
    obj_pcm: Box<[f32]>,
    frame: usize,
    /// Elements the bed takes, and all of them; each bed element's place.
    beds: usize,
    elements: usize,
    places: [[f32; 3]; MAX_ELEMENTS],
    /// The bed element a back centre takes: a dynamic object at a fixed place in OAMD.
    back_center: Option<usize>,
    /// Each dynamic element's object, when objects outnumber them.
    keeper: [Option<u8>; MAX_ELEMENTS],
    folding: bool,
    /// The OAMD index the LFE takes, which moves every element at or past it one on.
    lfe_at: Option<usize>,
    /// Per element, the timeslot of its first move this frame and how far, in steps.
    events: [Option<(u8, i8)>; MAX_ELEMENTS],
    /// Keeps the quietest objects instead, a new set every frame: the gate tests' other fold.
    #[cfg(test)]
    pub(super) scramble: bool,
}

impl Scene {
    /// The scene of `bed` and `objects` objects, `frame` samples a frame; `lfe_at` is
    /// where the program puts the LFE among its objects.
    pub(crate) fn new(bed: Bed, objects: usize, frame: usize, lfe_at: Option<usize>) -> Scene {
        let mut places = [[0.0; 3]; MAX_ELEMENTS];
        let mut back_center = None;
        let mut slot = 0u8;
        let channels: Box<[(Option<u8>, [f32; CORE])]> = BedChannel::ALL
            .iter()
            .filter(|c| bed.contains(**c))
            .map(|c| match position(*c) {
                Some(p) => {
                    places[usize::from(slot)] = p;
                    if *c == BedChannel::BackCenter {
                        back_center = Some(usize::from(slot));
                    }
                    slot += 1;
                    (Some(slot - 1), pan(p))
                }
                None => (None, [0.0; CORE]),
            })
            .collect();
        let beds = usize::from(slot);
        let dynamic = (MAX_ELEMENTS - beds).min(objects);
        let folding = objects > dynamic;
        let objects: Box<[Object]> = (0..objects)
            .map(|o| Object {
                pushed: false,
                live: false,
                age: 0,
                update: None,
                pos: [0.0; 3],
                size: 0.0,
                ramp: Ramp::held(gains(1.0, [0.0; 3])),
                energy: 0.0,
                loud: 0.0,
                dest: (!folding).then_some((beds + o) as u8),
                from: None,
                fade: XFADE,
            })
            .collect();
        Scene {
            bed_pcm: vec![0.0; channels.len() * frame].into_boxed_slice(),
            bed: channels,
            obj_pcm: vec![0.0; objects.len() * frame].into_boxed_slice(),
            objects,
            frame,
            beds,
            elements: beds + dynamic,
            places,
            back_center,
            keeper: [None; MAX_ELEMENTS],
            folding,
            lfe_at,
            events: [None; MAX_ELEMENTS],
            #[cfg(test)]
            scramble: false,
        }
    }

    /// Element `e`'s index among the program's objects.
    fn oamd(&self, e: usize) -> usize {
        e + usize::from(self.lfe_at.is_some_and(|l| e >= l))
    }

    /// The bed's state for the first frame: its elements and the LFE active, a back
    /// centre at its place.
    pub(crate) fn start(&self, meta: &mut oamd::Frame) {
        let active = Meta {
            active: true,
            ..Meta::default()
        };
        if let Some(l) = self.lfe_at {
            meta.update(l, 0, active);
        }
        for e in 0..self.beds {
            let m = if Some(e) == self.back_center {
                Meta::new(self.places[e], 0.0)
            } else {
                active
            };
            meta.update(self.oamd(e), 0, m);
        }
    }

    /// Takes a chunk of the bed, interleaved; every object is silent in it until pushed.
    pub(crate) fn bed(&mut self, pcm: &[f32]) {
        if let Some(dst) = self.bed_pcm.get_mut(..pcm.len()) {
            for (d, s) in dst.iter_mut().zip(pcm) {
                *d = sample(*s);
            }
        }
        for o in self.objects.iter_mut() {
            (o.pushed, o.update) = (false, None);
        }
    }

    /// Takes object `index`'s chunk, `n` samples, and the update inside it.
    pub(crate) fn object(&mut self, index: usize, pcm: &[f32], update: Option<ObjectUpdate>) {
        let frame = self.frame;
        let (Some(o), Some(dst)) = (
            self.objects.get_mut(index),
            self.obj_pcm
                .get_mut(index * frame..index * frame + pcm.len()),
        ) else {
            return;
        };
        for (d, s) in dst.iter_mut().zip(pcm) {
            *d = sample(*s);
        }
        if !o.live {
            o.age = 0;
        }
        o.pushed = true;
        o.live = true;
        o.update = update.map(|u| {
            let at = (u.at_sample as usize).min(pcm.len().saturating_sub(1));
            let pos = u.position.map(|v| bounded(v, -1.0, 1.0, 0.0));
            let gain = bounded(u.gain, 0.0, GAIN_MAX, 0.0);
            (at, pos, gain, bounded(u.size, 0.0, 1.0, 0.0))
        });
    }

    pub(crate) fn end(&mut self, index: usize) {
        if let Some(o) = self.objects.get_mut(index) {
            o.live = false;
            o.pushed = false;
            o.update = None;
        }
    }

    /// Dynamic element `e`'s object.
    fn keeper(&self, e: usize) -> Option<usize> {
        if self.folding {
            self.keeper.get(e).copied().flatten().map(usize::from)
        } else {
            (e >= self.beds && e - self.beds < self.objects.len()).then(|| e - self.beds)
        }
    }

    /// Hands the metadata every dynamic element's state at `offset` into the frame.
    pub(crate) fn publish(&self, offset: usize, meta: &mut oamd::Frame) {
        for e in self.beds..self.elements {
            let m = match self.keeper(e).and_then(|o| self.objects.get(o)) {
                Some(o) if o.live => Meta::new(o.pos, o.size),
                _ => Meta::default(),
            };
            meta.update(self.oamd(e), offset, m);
        }
    }

    /// Each element's first move this frame: the timeslot its effect lands in.
    pub(crate) fn events(&self, out: &mut [Option<u8>]) {
        for (o, e) in out.iter_mut().zip(self.events) {
            *o = e.map(|(t, _)| t);
        }
    }

    /// Opens a frame: last frame's energies decide, when objects outnumber the dynamic
    /// elements, which keep one; the rest go to the nearest element.
    pub(crate) fn begin_frame(&mut self, meta: &mut oamd::Frame) {
        self.events = [None; MAX_ELEMENTS];
        for o in self.objects.iter_mut() {
            o.loud = o.energy;
            o.energy = 0.0;
        }
        if !self.folding {
            return;
        }
        let dynamic = self.elements - self.beds;
        for k in self.keeper.iter_mut().take(dynamic) {
            if k.is_some_and(|o| !self.objects[usize::from(o)].live) {
                *k = None;
            }
        }
        // challengers, loudest first
        let mut order = [0u8; super::MAX_OBJECTS as usize];
        let mut n = 0;
        for (i, o) in self.objects.iter().enumerate() {
            let kept = self.keeper[..dynamic].contains(&Some(i as u8));
            if o.live && !kept {
                let mut j = n;
                while j > 0 && self.objects[usize::from(order[j - 1])].loud < o.loud {
                    order[j] = order[j - 1];
                    j -= 1;
                }
                order[j] = i as u8;
                n += 1;
            }
        }
        #[cfg(test)]
        if self.scramble {
            self.keeper = [None; MAX_ELEMENTS];
            order[..n].reverse();
        }
        for &c in &order[..n] {
            let loud = self.objects[usize::from(c)].loud;
            if let Some(free) = self.keeper[..dynamic].iter().position(Option::is_none) {
                self.keeper[free] = Some(c);
                continue;
            }
            let weakest = (0..dynamic).min_by(|a, b| {
                let l =
                    |e: usize| self.keeper[e].map_or(0.0, |o| self.objects[usize::from(o)].loud);
                l(*a).total_cmp(&l(*b))
            });
            match weakest {
                Some(w)
                    if loud
                        > HYSTERESIS
                            * self.keeper[w].map_or(0.0, |o| self.objects[usize::from(o)].loud)
                            + 1e-9 =>
                {
                    self.keeper[w] = Some(c);
                }
                _ => break,
            }
        }
        for i in 0..self.objects.len() {
            let own = self.keeper[..dynamic]
                .iter()
                .position(|k| *k == Some(i as u8))
                .map(|e| (self.beds + e) as u8);
            let dest = own.or_else(|| self.nearest(self.objects[i].pos));
            let o = &mut self.objects[i];
            if dest != o.dest {
                if o.dest.is_some() && o.loud > 0.0 {
                    (o.from, o.fade) = (o.dest, 0);
                } else {
                    (o.from, o.fade) = (None, XFADE);
                }
                if let Some(d) = own {
                    self.events[usize::from(d)] = Some((((XFADE + 320) / 64) as u8, i8::MAX));
                }
                o.dest = dest;
            }
        }
        self.publish(0, meta);
    }

    /// The element nearest `p`: a bed channel's place or a keeper's position.
    fn nearest(&self, p: [f32; 3]) -> Option<u8> {
        let mut best: Option<(u8, f32)> = None;
        for e in 0..self.elements {
            let at = if e < self.beds {
                Some(self.places[e])
            } else {
                self.keeper(e)
                    .and_then(|o| self.objects.get(o))
                    .map(|o| o.pos)
            };
            if let Some(q) = at {
                let d: f32 = (0..3).map(|i| (p[i] - q[i]) * (p[i] - q[i])).sum();
                if best.is_none_or(|b| d < b.1) {
                    best = Some((e as u8, d));
                }
            }
        }
        best.map(|b| b.0)
    }

    /// Renders `len` samples from `start` in the chunk into the timeslot buffers at `at`:
    /// `elem` per element, `core` L R C Ls Rs LFE. `offset` is the first sample's place in
    /// the frame; a keeper's update there reaches `meta`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render(
        &mut self,
        start: usize,
        len: usize,
        at: usize,
        offset: usize,
        elem: &mut [[f32; TS]],
        core: &mut [[f32; TS]; CORE + 1],
        meta: &mut oamd::Frame,
    ) {
        let len = len.min(TS - at.min(TS));
        let nb = self.bed.len();
        for (j, (e, g)) in self.bed.iter().enumerate() {
            let mut x = [0f32; TS];
            for (i, v) in x.iter_mut().enumerate().take(len) {
                *v = self
                    .bed_pcm
                    .get((start + i) * nb + j)
                    .copied()
                    .unwrap_or(0.0);
            }
            let x = &x[..len];
            match e.and_then(|e| elem.get_mut(usize::from(e))) {
                Some(dst) => {
                    dst[at..at + len].copy_from_slice(x);
                    for (c, g) in g.iter().enumerate().filter(|(_, g)| **g != 0.0) {
                        for (d, v) in core[c][at..at + len].iter_mut().zip(x) {
                            *d += v * g;
                        }
                    }
                }
                None => {
                    for (d, v) in core[CORE][at..at + len].iter_mut().zip(x) {
                        *d += v;
                    }
                }
            }
        }
        for k in 0..self.objects.len() {
            if !self.objects[k].pushed {
                continue;
            }
            let mut from = start;
            let end = start + len;
            if let Some((u, pos, gain, size)) = self.objects[k].update
                && (start..end).contains(&u)
            {
                self.object_span(k, from, u, start, at, elem, core);
                from = u;
                let o = &mut self.objects[k];
                (o.pos, o.size) = (pos, size);
                if o.age < RAMP {
                    o.ramp = Ramp::held(gains(gain, pos));
                } else {
                    o.ramp.retarget(gains(gain, pos));
                }
                o.update = None;
                let own = (0..self.elements).find(|e| self.keeper(*e) == Some(k));
                if let Some(e) = own {
                    let place = offset + u - start;
                    let before = meta.last(self.oamd(e));
                    let m = Meta::new(pos, size);
                    let step = (0..3)
                        .map(|i| (m.pos[i] - before.pos[i]).abs())
                        .max()
                        .unwrap_or(0);
                    let slot = ((place + 320 + RAMP as usize / 2) / TS).min(255) as u8;
                    if step >= 2 && self.events[e].is_none_or(|(_, s)| step > s) {
                        self.events[e] = Some((slot, step));
                    }
                    meta.update(self.oamd(e), place, m);
                }
            }
            self.object_span(k, from, end, start, at, elem, core);
        }
    }

    /// Object `k`'s samples `from..to` of the chunk; `start` sits at `at` in the buffers.
    #[allow(clippy::too_many_arguments)]
    fn object_span(
        &mut self,
        k: usize,
        from: usize,
        to: usize,
        start: usize,
        at: usize,
        elem: &mut [[f32; TS]],
        core: &mut [[f32; TS]; CORE + 1],
    ) {
        let frame = self.frame;
        let Some(o) = self.objects.get_mut(k) else {
            return;
        };
        o.age = o.age.saturating_add((to - from) as u32);
        let pcm = self.obj_pcm.get(k * frame..(k + 1) * frame).unwrap_or(&[]);
        let folding = self.folding;
        // held gains and no crossfade: the same sums, loop by loop
        if o.ramp.left == 0 && o.fade >= XFADE {
            let Some(x) = pcm.get(from..to) else {
                return;
            };
            let (b, n) = (at + from - start, to - from);
            let g = o.ramp.cur;
            for c in 0..CORE {
                for (d, v) in core[c][b..b + n].iter_mut().zip(x) {
                    *d += v * g[1 + c];
                }
            }
            if let Some(d) = o.dest.and_then(|d| elem.get_mut(usize::from(d))) {
                for (d, v) in d[b..b + n].iter_mut().zip(x) {
                    *d += v * g[0];
                }
            }
            if folding {
                for v in x {
                    o.energy += (v * g[0]) * (v * g[0]);
                }
            }
            return;
        }
        for i in from..to {
            let x = pcm.get(i).copied().unwrap_or(0.0);
            let g = o.ramp.next();
            let e = x * g[0];
            if folding {
                o.energy += e * e;
            }
            let b = at + i - start;
            for c in 0..CORE {
                core[c][b] += x * g[1 + c];
            }
            let (mut keep, mut leave) = (1.0, 0.0);
            if o.fade < XFADE {
                let r = o.fade as f32 / XFADE as f32;
                (keep, leave) = (r.sqrt(), (1.0 - r).sqrt());
                o.fade += 1;
            }
            if let Some(d) = o.dest.and_then(|d| elem.get_mut(usize::from(d))) {
                d[b] += keep * e;
            }
            if leave > 0.0
                && let Some(d) = o.from.and_then(|d| elem.get_mut(usize::from(d)))
            {
                d[b] += leave * e;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panner_keeps_power_and_puts_the_bed_on_its_speakers() {
        let close =
            |a: [f32; CORE], b: [f32; CORE]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
        let h = FRAC_1_SQRT_2;
        for (c, want) in [
            (BedChannel::FrontLeft, [1.0, 0.0, 0.0, 0.0, 0.0]),
            (BedChannel::FrontCenter, [0.0, 0.0, 1.0, 0.0, 0.0]),
            (BedChannel::SideRight, [0.0, 0.0, 0.0, 0.0, 1.0]),
            (BedChannel::BackLeft, [0.0, 0.0, 0.0, 1.0, 0.0]),
            (BedChannel::TopFrontRight, [0.0, h, 0.0, 0.0, 0.0]),
            (BedChannel::TopBackLeft, [0.0, 0.0, 0.0, h, 0.0]),
            (BedChannel::BackCenter, [0.0, 0.0, 0.0, h, h]),
        ] {
            let got = pan(position(c).unwrap());
            assert!(close(got, want), "{c:?}: {got:?}");
        }
        for i in 0..=40 {
            let t = i as f32 / 20.0 - 1.0;
            for p in [[t, 0.3, 0.0], [0.6, t, 0.0], [t, -t, 0.0], [t, 7.0, -3.0]] {
                let power: f32 = pan(p).iter().map(|g| g * g).sum();
                assert!((power - 1.0).abs() < 1e-5, "{p:?}: {power}");
            }
        }
    }

    #[test]
    fn a_ramp_lands_on_its_target_after_480_samples() {
        let mut r = Ramp::held([0.0; 6]);
        let target = gains(0.5, [0.3, -0.2, 0.1]);
        r.retarget(target);
        let mut last = [0.0; 6];
        for _ in 0..RAMP {
            last = r.next();
        }
        assert_eq!(last, target);
        assert_eq!(r.next(), target);
    }
}
