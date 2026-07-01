use super::multipole::Multipole;
use super::THETA;
use crate::star::Star;

const LEAF_CAPACITY: usize = 64;

pub struct Node {
    pub center: [f32; 2],
    pub radius: f32,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub start: usize,
    pub end: usize,
    pub m: Multipole,
}

impl Node {
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub order: Vec<usize>,
    pub root: usize,
}

impl Tree {
    pub fn build(stars: &[Star]) -> Tree {
        let n = stars.len();
        let mut order: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::new();
        let root = fill(&mut nodes, &mut order, 0, n, 0, stars);
        Tree { nodes, order, root }
    }
}

fn fill(
    nodes: &mut Vec<Node>,
    order: &mut [usize],
    lo: usize,
    hi: usize,
    depth: usize,
    stars: &[Star],
) -> usize {
    let (center, radius) = bounding(&order[lo..hi], stars);

    if hi - lo <= LEAF_CAPACITY {
        nodes.push(Node {
            center,
            radius,
            left: None,
            right: None,
            start: lo,
            end: hi,
            m: Multipole::new(center),
        });
        return nodes.len() - 1;
    }

    let axis = depth % 2;
    let mid = lo + (hi - lo) / 2;
    order[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
        stars[a].pos[axis]
            .partial_cmp(&stars[b].pos[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let left = fill(nodes, order, lo, mid, depth + 1, stars);
    let right = fill(nodes, order, mid, hi, depth + 1, stars);

    nodes.push(Node {
        center,
        radius,
        left: Some(left),
        right: Some(right),
        start: lo,
        end: hi,
        m: Multipole::new(center),
    });
    nodes.len() - 1
}

fn bounding(idx: &[usize], stars: &[Star]) -> ([f32; 2], f32) {
    let mut min = stars[idx[0]].pos;
    let mut max = min;
    for &i in idx {
        let p = stars[i].pos;
        min[0] = min[0].min(p[0]);
        min[1] = min[1].min(p[1]);
        max[0] = max[0].max(p[0]);
        max[1] = max[1].max(p[1]);
    }
    let center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5];
    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    (center, (dx * dx + dy * dy).sqrt() * 0.5)
}

/// Two cells are well-separated if 2*max(radius) / dist(centers) < THETA.
pub fn admissible(a: &Node, b: &Node) -> bool {
    let dx = a.center[0] - b.center[0];
    let dy = a.center[1] - b.center[1];
    let d = (dx * dx + dy * dy).sqrt();
    if d == 0.0 {
        return false;
    }
    2.0 * a.radius.max(b.radius) / d < THETA
}
