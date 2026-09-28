use std::sync::Arc;

use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::gpu::GpuContext;
use crate::simulation::{EPS, G, Simulation, drift_half};
use crate::star::Star;

struct Grid {
    data: Vec<f32>,
    size: i32,
    h: f32,
}

impl Grid {
    fn new(size: i32, h: f32) -> Self {
        let n = (size + 1) as usize;
        Self { data: vec![0.0; n * n], size, h }
    }

    fn weights(&self, pos: [f32; 2]) -> Option<[(usize, f32); 4]> {
        let half = self.size as f32 * self.h / 2.0;
        let gx = (pos[0] + half) / self.h;
        let gy = (pos[1] + half) / self.h;
        let (i, j) = (gx.floor() as i32, gy.floor() as i32);
        if i < 0 || j < 0 || i >= self.size || j >= self.size {
            return None;
        }

        let (dx, dy) = (gx - i as f32, gy - j as f32);
        let n = self.size + 1;
        let index = |i: i32, j: i32| (i * n + j) as usize;
        Some([
            (index(i, j), (1.0 - dx) * (1.0 - dy)),
            (index(i, j + 1), (1.0 - dx) * dy),
            (index(i + 1, j), dx * (1.0 - dy)),
            (index(i + 1, j + 1), dx * dy),
        ])
    }
}

pub(crate) fn fft_kernel(size: i32, h: f32) -> (usize, Vec<Complex32>) {
    let n = (size + 1) as usize;
    let p = (2 * n - 1).next_power_of_two();

    let mut kernel = vec![Complex32::ZERO; p * p];
    let reach = n as i32 - 1;
    for ui in -reach..=reach {
        for uj in -reach..=reach {
            let dx = ui as f32 * h;
            let dy = uj as f32 * h;
            let r2 = dx * dx + dy * dy + EPS * EPS;
            let k = -G / (r2 * r2.sqrt());

            let wi = ui.rem_euclid(p as i32) as usize;
            let wj = uj.rem_euclid(p as i32) as usize;
            kernel[wi * p + wj] = Complex32::new(k * dx, k * dy);
        }
    }

    let forward = FftPlanner::new().plan_fft_forward(p);
    let mut transposed = vec![Complex32::ZERO; p * p];
    fft_2d(&forward, &mut kernel, &mut transposed, p);

    (p, kernel)
}

struct FftSolver {
    p: usize,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    kernel: Vec<Complex32>,
    buffer: Vec<Complex32>,
    transposed: Vec<Complex32>,
}

impl FftSolver {
    fn new(grid: &Grid) -> Self {
        let (p, kernel) = fft_kernel(grid.size, grid.h);
        let mut planner = FftPlanner::new();

        Self {
            p,
            forward: planner.plan_fft_forward(p),
            inverse: planner.plan_fft_inverse(p),
            kernel,
            buffer: vec![Complex32::ZERO; p * p],
            transposed: vec![Complex32::ZERO; p * p],
        }
    }

    fn solve(&mut self, grid: &Grid, acc: &mut [(f32, f32)]) {
        let n = (grid.size + 1) as usize;
        let p = self.p;

        self.buffer.fill(Complex32::ZERO);
        for (row, masses) in self.buffer.chunks_mut(p).zip(grid.data.chunks(n)) {
            for (b, &m) in row.iter_mut().zip(masses) {
                *b = Complex32::new(m, 0.0);
            }
        }

        fft_2d(&self.forward, &mut self.buffer, &mut self.transposed, p);
        self.buffer
            .par_iter_mut()
            .zip(&self.kernel)
            .for_each(|(b, k)| *b *= k);
        fft_2d(&self.inverse, &mut self.buffer, &mut self.transposed, p);

        let scale = 1.0 / (p * p) as f32;
        for (acc_row, row) in acc.chunks_mut(n).zip(self.buffer.chunks(p)) {
            for (a, b) in acc_row.iter_mut().zip(row) {
                *a = (b.re * scale, b.im * scale);
            }
        }
    }
}

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

fn node_acc_direct(grid: &Grid, acc: &mut [(f32, f32)]) {
    let n = grid.size + 1;
    let node = |index: usize| (index as i32 / n, index as i32 % n);

    acc.par_iter_mut().enumerate().for_each(|(target, acc)| {
        let (ti, tj) = node(target);
        *acc = (0.0, 0.0);

        for (source, &mass) in grid.data.iter().enumerate() {
            let (si, sj) = node(source);
            let dx = (si - ti) as f32 * grid.h;
            let dy = (sj - tj) as f32 * grid.h;
            let r2 = dx * dx + dy * dy + EPS * EPS;
            let k = G * mass / (r2 * r2.sqrt());
            acc.0 += k * dx;
            acc.1 += k * dy;
        }
    });
}

enum NodeSolver {
    Direct,
    Fft(FftSolver),
}

pub struct ParticleMesh {
    stars: Vec<Star>,
    total_mass: f32,
    grid: Grid,
    acc: Vec<(f32, f32)>,
    solver: NodeSolver,
}

impl ParticleMesh {
    pub fn new(stars: Vec<Star>, size: i32, h: f32) -> Self {
        Self::with_solver(stars, Grid::new(size, h), NodeSolver::Direct)
    }

    pub fn new_fft(stars: Vec<Star>, size: i32, h: f32) -> Self {
        let grid = Grid::new(size, h);
        let solver = NodeSolver::Fft(FftSolver::new(&grid));
        Self::with_solver(stars, grid, solver)
    }

    fn with_solver(stars: Vec<Star>, grid: Grid, solver: NodeSolver) -> Self {
        let acc = vec![(0.0, 0.0); grid.data.len()];
        let total_mass = stars.iter().map(|s| s.mass).sum();
        Self { stars, total_mass, grid, acc, solver }
    }

    #[cfg(test)]
    pub fn stars(&self) -> &[Star] {
        &self.stars
    }

    pub fn update(&mut self, dt: f32) {
        drift_half(&mut self.stars, dt);

        self.grid.data.fill(0.0);
        for star in &self.stars {
            for (index, w) in self.grid.weights(star.pos).into_iter().flatten() {
                self.grid.data[index] += star.mass * w;
            }
        }

        match &mut self.solver {
            NodeSolver::Direct => node_acc_direct(&self.grid, &mut self.acc),
            NodeSolver::Fft(solver) => solver.solve(&self.grid, &mut self.acc),
        }

        let (grid, acc, gm) = (&self.grid, &self.acc, G * self.total_mass);
        self.stars.par_iter_mut().for_each(|star| {
            let a = match grid.weights(star.pos) {
                Some(weights) => weights.iter().fold((0.0, 0.0), |a, &(index, w)| {
                    (a.0 + w * acc[index].0, a.1 + w * acc[index].1)
                }),
                None => {
                    let [x, y] = star.pos;
                    let r2 = x * x + y * y + EPS * EPS;
                    let k = -gm / (r2 * r2.sqrt());
                    (k * x, k * y)
                }
            };
            star.vel[0] += a.0 * dt;
            star.vel[1] += a.1 * dt;
        });

        drift_half(&mut self.stars, dt);
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
    use crate::galaxy::{GalaxyConfig, generate_galaxy};
    use crate::simulation::CpuDirect;

    fn star(x: f32, y: f32, mass: f32) -> Star {
        Star { pos: [x, y], mass, ..bytemuck::Zeroable::zeroed() }
    }

    #[test]
    fn deposit() {
        let stars = vec![star(-90.0, -90.0, 1000.0), star(30.0, 60.0, 500.0), star(-110.0, -50.0, 1000.0)];
        let mut pm = ParticleMesh::new(stars, 2, 100.0);
        pm.update(0.1);


        let expected = [810.0, 90.0, 0.0, 90.0, 150.0, 210.0, 0.0, 60.0, 90.0];
        for (actual, expected) in pm.grid.data.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-3, "{:?}", pm.grid.data);
        }
    }

    #[test]
    fn fft_matches_pair_sum() {
        let mut grid = Grid::new(20, 5.0);
        for (index, mass) in grid.data.iter_mut().enumerate() {
            *mass = (index % 7) as f32 * 10.0;
        }

        let mut pairs = vec![(0.0, 0.0); grid.data.len()];
        let mut fft = pairs.clone();
        node_acc_direct(&grid, &mut pairs);
        FftSolver::new(&grid).solve(&grid, &mut fft);

        let max = pairs.iter().map(|a| a.0.abs().max(a.1.abs())).fold(0.0, f32::max);
        for (p, f) in pairs.iter().zip(&fft) {
            assert!((p.0 - f.0).abs() < max * 1e-4 && (p.1 - f.1).abs() < max * 1e-4);
        }
    }

    #[test]
    fn pm_matches_direct() {
        let stars = generate_galaxy(&GalaxyConfig {
            center: [0.0, 0.0],
            radius: 200.0,
            star_count: 2000,
            core_mass: 500000.0,
            core_radius: 2.0,
            star_mass: 250.0,
            star_radius: 1.5,
            gap: 20.0,
            arms: 2,
        });
        let mut pm = ParticleMesh::new_fft(stars.clone(), 127, 5.0);
        let mut direct = CpuDirect::new(stars.clone());
        for _ in 0..10 {
            pm.update(0.004);
            direct.update(0.004);
        }

        let distance = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
        let (mut error, mut travel) = (0.0, 0.0);
        for ((p, d), s) in pm.stars().iter().zip(direct.stars()).zip(&stars) {
            error += distance(p.pos, d.pos);
            travel += distance(d.pos, s.pos);
        }
        println!("error {:.3}% of travel", error / travel * 100.0);
        assert!(error < travel * 0.01);
    }
}
