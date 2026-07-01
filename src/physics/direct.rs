//! Direct O(n^2) all-pairs gravity. Exact, the baseline for accuracy/perf.

use rayon::prelude::*;

use super::gravity;
use crate::star::Star;

pub fn accels_plain(stars: &[Star]) -> Vec<[f32; 2]> {
    (0..stars.len()).map(|i| body_accel(stars, i)).collect()
}

pub fn accels_rayon(stars: &[Star]) -> Vec<[f32; 2]> {
    (0..stars.len())
        .into_par_iter()
        .map(|i| body_accel(stars, i))
        .collect()
}

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
