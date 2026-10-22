//! The bit writer over one frame buffer. Fixed capacity: a write past the end is a tuner
//! bug, surfaced as an error the frame drops on, never a panic.
