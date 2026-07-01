mod multipole;
mod tree;

use super::gravity;
use crate::star::Star;
use tree::{admissible, Tree};

pub const THETA: f32 = 0.5;

pub fn accels(stars: &[Star]) -> Vec<[f32; 2]> {
    let n = stars.len();
    if n == 0 {
        return Vec::new();
    }

    let mut tree = Tree::build(stars);
    assemble_multipoles(&mut tree, stars);

    let mut forces = vec![[0.0f32; 2]; n];
    traverse(&tree, tree.root, tree.root, stars, &mut forces);

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

fn traverse(tree: &Tree, s: usize, t: usize, stars: &[Star], forces: &mut [[f32; 2]]) {
    let nodes = &tree.nodes;
    let order = &tree.order;
    let s_leaf = nodes[s].is_leaf();
    let t_leaf = nodes[t].is_leaf();

    if s_leaf && t_leaf {
        let (ss, se) = (nodes[s].start, nodes[s].end);
        let (ts, te) = (nodes[t].start, nodes[t].end);
        for &ti in &order[ts..te] {
            let mut fx = 0.0;
            let mut fy = 0.0;
            for &si in &order[ss..se] {
                let [dx, dy] = gravity(&stars[ti], &stars[si]);
                fx += dx;
                fy += dy;
            }
            forces[ti][0] += fx;
            forces[ti][1] += fy;
        }
        return;
    }

    if admissible(&nodes[s], &nodes[t]) {
        let (ts, te) = (nodes[t].start, nodes[t].end);
        let m = nodes[s].m;
        for &ti in &order[ts..te] {
            let [fx, fy] = m.eval(stars[ti].pos, stars[ti].mass);
            forces[ti][0] += fx;
            forces[ti][1] += fy;
        }
        return;
    }

    let split_source = if s_leaf {
        false
    } else if t_leaf {
        true
    } else {
        nodes[s].radius >= nodes[t].radius
    };

    if split_source {
        traverse(tree, nodes[s].left.unwrap(), t, stars, forces);
        traverse(tree, nodes[s].right.unwrap(), t, stars, forces);
    } else {
        traverse(tree, s, nodes[t].left.unwrap(), stars, forces);
        traverse(tree, s, nodes[t].right.unwrap(), stars, forces);
    }
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
    fn fmm_matches_direct() {
        let stars = make_stars(1500);
        let exact = direct::accels_plain(&stars);
        let approx = accels(&stars);

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
