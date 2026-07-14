use rayon::prelude::*;

use super::gravity;
use crate::star::Star;

pub fn direct(stars: &[Star]) -> Vec<[f32; 2]> {
    (0..stars.len()).map(|i| acceleration(stars, i)).collect()
}

pub fn direct_rayon(stars: &[Star]) -> Vec<[f32; 2]> {
    (0..stars.len())
        .into_par_iter()
        .map(|i| acceleration(stars, i))
        .collect()
}

fn acceleration(stars: &[Star], i: usize) -> [f32; 2] {
    if i == 0 {
        return [0.0, 0.0];
    }
    let s = &stars[i];
    let mut ax = 0.0;
    let mut ay = 0.0;
    for j in 0..stars.len() {
        if i == j {
            continue;
        }
        let [fx, fy] = gravity(s, &stars[j]);
        ax += fx / s.mass;
        ay += fy / s.mass;
    }
    [ax, ay]
}
