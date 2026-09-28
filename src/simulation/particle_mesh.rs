use std::sync::Arc;

use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::gpu::GpuContext;
use crate::simulation::{EPS, G, Simulation};
use crate::star::Star;

#[derive(Debug)]
pub struct Grid {
    data: Vec<f32>,
    size: i32,
    h: f32,
    x0: f32,
    y0: f32,
}

impl Grid {
    fn new (size: i32, h: f32, x0: f32, y0: f32) -> Self {
        Self {
            size,
            h,
            x0,
            y0,
            data: vec![0.0; ((size + 1) * (size + 1)) as usize]
        }
    }
    fn cell(&self, x: f32, y: f32) -> (i32, i32, f32, f32, bool) {
        let gx = (x - self.x0) / self.h;
        let gy = (y - self.y0) / self.h;

        let i = gx.floor() as i32;
        let j = gy.floor() as i32;

        let dx = gx - i as f32;
        let dy = gy - j as f32;

        let ok = i >= 0 && j >= 0 && i < self.size && j < self.size;

        (i, j, dx, dy, ok)
    }

    fn nodes(&self) -> impl Iterator<Item = (i32, i32)> + use<> {
        let n = self.size + 1;
        (0..n).flat_map(move |i| (0..n).map(move |j| (i, j)))
    }

    fn node(&self, index: usize) -> (i32, i32) {
        let n = (self.size + 1) as usize;
        ((index / n) as i32, (index % n) as i32)
    }

    fn index(&self, i: i32, j: i32) -> usize {
        (i * (self.size + 1) + j) as usize
    }

    fn add(&mut self, i: i32, j: i32, value: f32) {
        let index = self.index(i, j);
        self.data[index] += value;
    }
}

/// Node accelerations as a convolution of the node masses with the softened
/// gravity kernel, done through a 2D FFT. Gives the same result as summing
/// over all node pairs, in O(M log M) instead of O(M²).
struct FftSolver {
    /// padded side length, at least 2n - 1 so the circular convolution of the
    /// FFT doesn't wrap one edge of the grid onto the other
    p: usize,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    /// FFT of kx + i·ky: both kernels packed into one complex array, so one
    /// inverse FFT gives ax in the real part and ay in the imaginary part
    kernel: Vec<Complex32>,
    buffer: Vec<Complex32>,
    transposed: Vec<Complex32>,
}

impl FftSolver {
    fn new(grid: &Grid) -> Self {
        let n = (grid.size + 1) as usize;
        let p = (2 * n - 1).next_power_of_two();

        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(p);
        let inverse = planner.plan_fft_inverse(p);

        let mut kernel = vec![Complex32::ZERO; p * p];
        let mut transposed = vec![Complex32::ZERO; p * p];

        // acc[t] = Σ m[s] · F(s - t), F = pull towards s. As a convolution
        // over u = t - s the kernel is F(-u) = -F(u).
        let reach = n as i32 - 1;
        for ui in -reach..=reach {
            for uj in -reach..=reach {
                let dx = ui as f32 * grid.h;
                let dy = uj as f32 * grid.h;
                let r2 = dx * dx + dy * dy + EPS * EPS;
                let k = -G / (r2 * r2.sqrt());

                let wi = ui.rem_euclid(p as i32) as usize;
                let wj = uj.rem_euclid(p as i32) as usize;
                kernel[wi * p + wj] = Complex32::new(k * dx, k * dy);
            }
        }
        fft_2d(&forward, &mut kernel, &mut transposed, p);

        Self {
            p,
            forward,
            inverse,
            kernel,
            buffer: vec![Complex32::ZERO; p * p],
            transposed,
        }
    }

    fn solve(&mut self, grid: &Grid, acc: &mut [(f32, f32)]) {
        let n = (grid.size + 1) as usize;
        let p = self.p;

        self.buffer.fill(Complex32::ZERO);
        self.buffer
            .par_chunks_mut(p)
            .zip(grid.data.par_chunks(n))
            .for_each(|(row, masses)| {
                for (b, &m) in row.iter_mut().zip(masses) {
                    *b = Complex32::new(m, 0.0);
                }
            });

        fft_2d(&self.forward, &mut self.buffer, &mut self.transposed, p);
        self.buffer
            .par_iter_mut()
            .zip(&self.kernel)
            .for_each(|(b, k)| *b *= k);
        fft_2d(&self.inverse, &mut self.buffer, &mut self.transposed, p);

        // rustfft doesn't normalize, forward + inverse scales by p² in total
        let scale = 1.0 / (p * p) as f32;
        acc.par_chunks_mut(n)
            .zip(self.buffer.par_chunks(p))
            .for_each(|(acc_row, row)| {
                for (a, b) in acc_row.iter_mut().zip(row) {
                    *a = (b.re * scale, b.im * scale);
                }
            });
    }
}

/// 2D FFT in place: rows, transpose, rows again (former columns), transpose back.
fn fft_2d(fft: &Arc<dyn Fft<f32>>, data: &mut [Complex32], transposed: &mut [Complex32], p: usize) {
    let rows = |data: &mut [Complex32]| {
        data.par_chunks_mut(p).for_each_init(
            || vec![Complex32::ZERO; fft.get_inplace_scratch_len()],
            |scratch, row| fft.process_with_scratch(row, scratch),
        );
    };
    let transpose = |from: &[Complex32], to: &mut [Complex32]| {
        to.par_chunks_mut(p).enumerate().for_each(|(r, row)| {
            for (c, value) in row.iter_mut().enumerate() {
                *value = from[c * p + r];
            }
        });
    };

    rows(data);
    transpose(data, transposed);
    rows(transposed);
    transpose(transposed, data);
}

/// Node accelerations by summing over all node pairs, O(M²).
fn node_acc_direct(grid: &Grid, acc: &mut [(f32, f32)]) {
    let sources: Vec<(i32, i32, f32)> = grid
        .nodes()
        .zip(&grid.data)
        .filter(|&(_, &mass)| mass != 0.0)
        .map(|((i, j), &mass)| (i, j, mass))
        .collect();

    acc.par_iter_mut().enumerate().for_each(|(target, acc)| {
        let (ti, tj) = grid.node(target);
        *acc = (0.0, 0.0);

        for &(si, sj, mass) in &sources {
            if si == ti && sj == tj {
                continue;
            }

            let dx = (si - ti) as f32 * grid.h;
            let dy = (sj - tj) as f32 * grid.h;

            let r2 = dx * dx + dy * dy + EPS * EPS;
            let k = G * mass / (r2 * r2.sqrt());
            acc.0 += k * dx;
            acc.1 += k * dy;
        }
    });
}

/// How node accelerations are computed from node masses.
enum NodeSolver {
    Direct,
    Fft(FftSolver),
}

pub struct ParticleMesh {
    stars: Vec<Star>,
    grid: Grid,
    acc: Vec<(f32, f32)>,
    solver: NodeSolver,
}

impl ParticleMesh {
    pub fn new(stars: Vec<Star>, size: i32, h: f32, x0: f32, y0: f32) -> Self {
        Self::with_solver(stars, Grid::new(size, h, x0, y0), |_| NodeSolver::Direct)
    }

    pub fn new_fft(stars: Vec<Star>, size: i32, h: f32, x0: f32, y0: f32) -> Self {
        Self::with_solver(stars, Grid::new(size, h, x0, y0), |grid| {
            NodeSolver::Fft(FftSolver::new(grid))
        })
    }

    fn with_solver(stars: Vec<Star>, grid: Grid, solver: impl FnOnce(&Grid) -> NodeSolver) -> Self {
        let acc = vec![(0.0, 0.0); grid.data.len()];
        let solver = solver(&grid);
        Self {
            stars,
            grid,
            acc,
            solver,
        }
    }

    pub fn stars(&self) -> &[Star] {
        &self.stars
    }

    pub fn update(&mut self, dt: f32) {
        self.stars.par_iter_mut().skip(1).for_each(|s| {
            s.pos[0] += 0.5 * s.vel[0] * dt;
            s.pos[1] += 0.5 * s.vel[1] * dt;
        });

        self.grid.data.fill(0.0);

        for star in self.stars.iter() {
            let (i, j, dx, dy, ok) = self.grid.cell(star.pos[0], star.pos[1]);

            if !ok {
                continue;
            }

            self.grid.add(i, j, star.mass * (1.0 - dx) * (1.0 - dy));
            self.grid.add(i, j + 1, star.mass * (1.0 - dx) * dy);
            self.grid.add(i + 1, j, star.mass * dx * (1.0 - dy));
            self.grid.add(i + 1, j + 1, star.mass * dx * dy);
        }

        match &mut self.solver {
            NodeSolver::Direct => node_acc_direct(&self.grid, &mut self.acc),
            NodeSolver::Fft(solver) => solver.solve(&self.grid, &mut self.acc),
        }

        let grid = &self.grid;
        let acc = &self.acc;
        self.stars.par_iter_mut().for_each(|star| {
            let (i, j, dx, dy, ok) = grid.cell(star.pos[0], star.pos[1]);

            if !ok {
                return;
            }

            let weights = [
                (i, j, (1.0 - dx) * (1.0 - dy)),
                (i, j + 1, (1.0 - dx) * dy),
                (i + 1, j, dx * (1.0 - dy)),
                (i + 1, j + 1, dx * dy),
            ];

            for (ni, nj, w) in weights {
                let node = acc[grid.index(ni, nj)];
                star.vel[0] += w * node.0 * dt;
                star.vel[1] += w * node.1 * dt;
            }
        });

        self.stars.par_iter_mut().skip(1).for_each(|s| {
            s.pos[0] += 0.5 * s.vel[0] * dt;
            s.pos[1] += 0.5 * s.vel[1] * dt;
        });
    }
}

impl Simulation for ParticleMesh {
    fn step(
        &mut self,
        gpu: &GpuContext,
        _encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        dt: f32,
    ) {
        self.update(dt);
        gpu.write(buffer, &self.stars);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn equal(a: f32, b: f32) -> bool {
        (a - b).abs() < 1.0e-3
    }

    #[test]
    fn test_fft_matches_direct() {
        let mut grid = Grid::new(20, 5.0, 0.0, 0.0);
        let nodes: Vec<_> = grid.nodes().collect();
        for (i, j) in nodes {
            // uneven masses, some nodes left empty
            let index = grid.index(i, j);
            grid.data[index] = ((i * 7 + j * 13) % 5) as f32 * 10.0;
        }

        let mut direct = vec![(0.0, 0.0); grid.data.len()];
        node_acc_direct(&grid, &mut direct);

        let mut fft = vec![(0.0, 0.0); grid.data.len()];
        FftSolver::new(&grid).solve(&grid, &mut fft);

        let max = direct.iter().map(|a| a.0.abs().max(a.1.abs())).fold(0.0, f32::max);
        for (d, f) in direct.iter().zip(&fft) {
            assert!((d.0 - f.0).abs() < max * 1.0e-4, "direct {d:?}, fft {f:?}");
            assert!((d.1 - f.1).abs() < max * 1.0e-4, "direct {d:?}, fft {f:?}");
        }
    }

    #[test]
    fn test_pm_basic() {
        let mass = 1000.0;
        let star =  Star {
            pos: [10.0, 10.0],
            mass,
            radius: 3.0,
            ..bytemuck::Zeroable::zeroed()
        };

        let mut pm = ParticleMesh::new(vec![star], 2, 100.0, 0.0, 0.0);
        pm.update(0.1);
        pm.update(0.1);

        assert!(equal(pm.grid.data.iter().sum::<f32>(), mass));
        assert!(equal(pm.grid.data[0], 810.0));
        assert!(equal(pm.grid.data[1], 90.0));
        assert!(equal(pm.grid.data[3], 90.0));
        assert!(equal(pm.grid.data[4], 10.0));
    }

    #[test]
    fn test_pm_star_outside() {
        let star =  Star {
            pos: [-10.0, -10.0],
            mass: 1000.0,
            radius: 3.0,
            ..bytemuck::Zeroable::zeroed()
        };

        let mut pm = ParticleMesh::new(vec![star], 2, 100.0, 0.0, 0.0);
        pm.update(0.1);

        assert!(equal(pm.grid.data.iter().sum::<f32>(), 0.0));
    }

    #[test]
    fn test_pm_non_zero_origin() {
        let mass = 1000.0;
        let star =  Star {
            pos: [120.0, 80.0],
            mass: mass,
            radius: 3.0,
            ..bytemuck::Zeroable::zeroed()
        };

        let mut pm = ParticleMesh::new(vec![star], 2, 100.0, 100.0, 50.0);
        pm.update(0.1);

        assert!(equal(pm.grid.data.iter().sum::<f32>(), mass));
        assert!(equal(pm.grid.data[0], 560.0));
        assert!(equal(pm.grid.data[1], 240.0));
        assert!(equal(pm.grid.data[3], 140.0));
        assert!(equal(pm.grid.data[4], 60.0));
    }

        #[test]
    fn test_pm_star_on_node() {
        let mass = 1000.0;
        let star =  Star {
            pos: [200.0, 150.0],
            mass: mass,
            radius: 3.0,
            ..bytemuck::Zeroable::zeroed()
        };

        let mut pm = ParticleMesh::new(vec![star], 2, 100.0, 100.0, 50.0);
        pm.update(0.1);

        assert!(equal(pm.grid.data.iter().sum::<f32>(), mass));
        assert!(equal(pm.grid.data[4], mass));
    }

    #[test]
    fn test_pm_star_on_far_edge() {
        let star =  Star {
            pos: [200.0, 250.0],
            mass: 1000.0,
            radius: 3.0,
            ..bytemuck::Zeroable::zeroed()
        };

        let mut pm = ParticleMesh::new(vec![star], 2, 100.0, 100.0, 50.0);
        pm.update(0.1);

        assert!(equal(pm.grid.data.iter().sum::<f32>(), 0.0));
    }

    #[test]
    fn test_pm_multiple_stars() {
        let star = |pos: [f32; 2], mass: f32| Star {
            pos,
            mass,
            radius: 3.0,
            ..bytemuck::Zeroable::zeroed()
        };

        let stars = vec![
            star([10.0, 10.0], 1000.0),
            star([50.0, 20.0], 200.0),
            star([130.0, 160.0], 500.0),
        ];

        let mut pm = ParticleMesh::new(stars, 2, 100.0, 0.0, 0.0);
        pm.update(0.1);

        let expected = [
            890.0, 110.0, 0.0,
            170.0, 170.0, 210.0,
            0.0, 60.0, 90.0,
        ];

        assert!(equal(pm.grid.data.iter().sum::<f32>(), 1700.0));
        for (actual, expected) in pm.grid.data.iter().zip(expected) {
            assert!(equal(*actual, expected));
        }
    }
}