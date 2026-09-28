use rayon::prelude::*;

use crate::gpu::GpuContext;
use crate::simulation::{EPS, G, Simulation, drift_half};
use crate::star::Star;

pub struct CpuDirect {
    stars: Vec<Star>,
}

impl CpuDirect {
    pub fn new(stars: Vec<Star>) -> Self {
        Self { stars }
    }

    #[cfg(test)]
    pub fn stars(&self) -> &[Star] {
        &self.stars
    }

    pub fn update(&mut self, dt: f32) {
        let stars = &mut self.stars;

        drift_half(stars, dt);

        let acc: Vec<[f32; 2]> = stars
            .par_iter()
            .map(|star| {
                let pos = star.pos;
                let mut a = [0.0, 0.0];
                for other in stars.iter() {
                    let d = [other.pos[0] - pos[0], other.pos[1] - pos[1]];
                    let inv_dist = 1.0 / (d[0] * d[0] + d[1] * d[1] + EPS * EPS).sqrt();
                    let k = G * other.mass * inv_dist * inv_dist * inv_dist;
                    a[0] += d[0] * k;
                    a[1] += d[1] * k;
                }
                a
            })
            .collect();

        stars.par_iter_mut().zip(acc).for_each(|(s, a)| {
            s.vel[0] += a[0] * dt;
            s.vel[1] += a[1] * dt;
        });

        drift_half(stars, dt);
    }
}

impl Simulation for CpuDirect {
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
