//! Design D4: the same input under two push schedules gives byte-identical frames at the
//! same pts.

#![no_main]

mod common;

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = run(&mut Unstructured::new(data));
});

fn run(u: &mut Unstructured) -> arbitrary::Result<()> {
    let Some(config) = common::config(u)? else {
        return Ok(());
    };
    let (a, b) = (common::schedule(u)?, common::schedule(u)?);
    let pcm = common::signal(u, config)?;
    let one = common::encode(config, &pcm, &a);
    let two = common::encode(config, &pcm, &b);
    if let Some(k) = one.iter().zip(&two).position(|(x, y)| x != y) {
        panic!("frame {k} differs between schedules {a:?} and {b:?}");
    }
    Ok(())
}
