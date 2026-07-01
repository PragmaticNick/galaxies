//! Direct O(n^2) all-pairs gravity. Exact, the baseline for accuracy/perf.

use rayon::prelude::*;

use super::G;
use crate::star::Star;

/// O(n^2) accel, single-threaded.
pub fn accels_plain(stars: &[Star]) -> Vec<[f32; 2]> {
    (0..stars.len()).map(|i| body_accel(stars, i)).collect()
}

/// O(n^2) accel, parallel over bodies (immutable borrow of `stars`).
pub fn accels_rayon(stars: &[Star]) -> Vec<[f32; 2]> {
    (0..stars.len())
        .into_par_iter()
        .map(|i| body_accel(stars, i))
        .collect()
}

/// Accel on body `i` from all others. Body 0 (core) is fixed.
fn body_accel(stars: &[Star], i: usize) -> [f32; 2] {
    if i == 0 {
        return [0.0, 0.0];
    }
    let si = &stars[i];
    let mut ax = 0.0;
    let mut ay = 0.0;
    for j in 0..stars.len() {
        if i == j {
            continue;
        }
        let [fx, fy] = gravity(si, &stars[j]);
        ax += fx / si.mass;
        ay += fy / si.mass;
    }
    [ax, ay]
}

fn gravity(s: &Star, t: &Star) -> [f32; 2] {
    const SOFTENING2: f32 = 400.0;
    let dx = t.pos[0] - s.pos[0];
    let dy = t.pos[1] - s.pos[1];
    let r2 = dx * dx + dy * dy + SOFTENING2;
    let r = r2.sqrt();
    let f = G * s.mass * t.mass / r2;
    [f * dx / r, f * dy / r]
}
