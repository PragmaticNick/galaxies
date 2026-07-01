mod multipole;
mod tree;

use rayon::prelude::*;

use super::gravity;
use crate::star::Star;
use tree::{admissible, Tree};

pub const THETA: f32 = 0.5;

/// Single-threaded: walk each target leaf in sequence.
pub fn accels_serial(stars: &[Star]) -> Vec<[f32; 2]> {
    if stars.is_empty() {
        return Vec::new();
    }
    let (tree, leaves) = prepare(stars);
    let contributions: Vec<(usize, [f32; 2])> = leaves
        .iter()
        .flat_map(|&t| leaf_forces(&tree, t, stars))
        .collect();
    finalize(contributions, stars)
}

/// Parallel: each target leaf owns a disjoint particle range, so leaves are
/// processed across cores without contention.
pub fn accels_rayon(stars: &[Star]) -> Vec<[f32; 2]> {
    if stars.is_empty() {
        return Vec::new();
    }
    let (tree, leaves) = prepare(stars);
    let contributions: Vec<(usize, [f32; 2])> = leaves
        .par_iter()
        .flat_map_iter(|&t| leaf_forces(&tree, t, stars))
        .collect();
    finalize(contributions, stars)
}

fn prepare(stars: &[Star]) -> (Tree, Vec<usize>) {
    let mut tree = Tree::build(stars);
    assemble_multipoles(&mut tree, stars);
    let leaves = (0..tree.nodes.len())
        .filter(|&i| tree.nodes[i].is_leaf())
        .collect();
    (tree, leaves)
}

/// Forces on target leaf `t`'s particles from the whole tree. A leaf's targets
/// are fixed, so only the source tree is descended.
fn leaf_forces<'a>(
    tree: &'a Tree,
    t: usize,
    stars: &'a [Star],
) -> impl Iterator<Item = (usize, [f32; 2])> + 'a {
    let (ts, te) = (tree.nodes[t].start, tree.nodes[t].end);
    let targets = &tree.order[ts..te];
    let mut local = vec![[0.0f32; 2]; targets.len()];
    accumulate(tree, tree.root, t, targets, stars, &mut local);
    targets.iter().copied().zip(local)
}

fn finalize(contributions: Vec<(usize, [f32; 2])>, stars: &[Star]) -> Vec<[f32; 2]> {
    let mut forces = vec![[0.0f32; 2]; stars.len()];
    for (i, f) in contributions {
        forces[i] = f;
    }
    forces
        .iter()
        .zip(stars)
        .map(|(f, s)| [f[0] / s.mass, f[1] / s.mass])
        .collect()
}

fn assemble_multipoles(tree: &mut Tree, stars: &[Star]) {
    for i in 0..tree.nodes.len() {
        if tree.nodes[i].is_leaf() {
            let (start, end) = (tree.nodes[i].start, tree.nodes[i].end);
            for &pi in &tree.order[start..end] {
                tree.nodes[i].m.add_point(stars[pi].pos, stars[pi].mass);
            }
        } else {
            let lm = tree.nodes[tree.nodes[i].left.unwrap()].m;
            let rm = tree.nodes[tree.nodes[i].right.unwrap()].m;
            tree.nodes[i].m.add_multipole(&lm);
            tree.nodes[i].m.add_multipole(&rm);
        }
    }
}

/// Accumulate force on each particle of target leaf `t` (its bodies are
/// `targets`, with `local` aligned to them) from source subtree `source`.
fn accumulate(
    tree: &Tree,
    source: usize,
    t: usize,
    targets: &[usize],
    stars: &[Star],
    local: &mut [[f32; 2]],
) {
    if admissible(&tree.nodes[source], &tree.nodes[t]) {
        let m = tree.nodes[source].m;
        for (k, &ti) in targets.iter().enumerate() {
            let [fx, fy] = m.eval(stars[ti].pos, stars[ti].mass);
            local[k][0] += fx;
            local[k][1] += fy;
        }
        return;
    }

    if tree.nodes[source].is_leaf() {
        let (ss, se) = (tree.nodes[source].start, tree.nodes[source].end);
        for (k, &ti) in targets.iter().enumerate() {
            let mut fx = 0.0;
            let mut fy = 0.0;
            for &si in &tree.order[ss..se] {
                let [dx, dy] = gravity(&stars[ti], &stars[si]);
                fx += dx;
                fy += dy;
            }
            local[k][0] += fx;
            local[k][1] += fy;
        }
        return;
    }

    accumulate(tree, tree.nodes[source].left.unwrap(), t, targets, stars, local);
    accumulate(tree, tree.nodes[source].right.unwrap(), t, targets, stars, local);
}

#[cfg(test)]
mod tests {
    use super::super::direct;
    use super::*;

    fn make_stars(n: usize) -> Vec<Star> {
        let mut s = 12345u32;
        let mut rand = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1 << 24) as f32
        };
        (0..n)
            .map(|i| Star {
                pos: [(rand() - 0.5) * 1000.0, (rand() - 0.5) * 1000.0],
                vel: [0.0, 0.0],
                mass: if i == 0 { 50000.0 } else { 0.1 },
                radius: 1.0,
                color: [1.0, 1.0, 1.0],
            })
            .collect()
    }

    #[test]
    fn serial_matches_rayon() {
        let stars = make_stars(1500);
        assert_eq!(accels_serial(&stars), accels_rayon(&stars));
    }

    #[test]
    fn fmm_matches_direct() {
        let stars = make_stars(1500);
        let exact = direct::accels_plain(&stars);
        let approx = accels_serial(&stars);

        let mut max_rel = 0.0f32;
        let mut sum_rel = 0.0f32;
        let mut count = 0;
        for i in 1..stars.len() {
            let dx = approx[i][0] - exact[i][0];
            let dy = approx[i][1] - exact[i][1];
            let err = (dx * dx + dy * dy).sqrt();
            let mag = (exact[i][0].powi(2) + exact[i][1].powi(2)).sqrt();
            if mag > 1e-3 {
                let rel = err / mag;
                max_rel = max_rel.max(rel);
                sum_rel += rel;
                count += 1;
            }
        }
        let mean_rel = sum_rel / count as f32;
        assert!(mean_rel < 0.04, "mean relative error too high: {mean_rel}");
        assert!(max_rel < 0.2, "max relative error too high: {max_rel}");
    }
}
