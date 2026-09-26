use crate::gpu::GpuContext;
use crate::simulation::{EPS, G, Simulation};
use crate::star::Star;

pub struct CpuDirect {
    stars: Vec<Star>,
}

impl Simulation for CpuDirect {
    fn init(_gpu: &GpuContext, _buffer: &wgpu::Buffer, stars: Vec<Star>) -> Self {
        Self { stars }
    }

    fn step(
        &mut self,
        gpu: &GpuContext,
        _encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        dt: f32,
    ) {
        let stars = &mut self.stars;

        for s in stars.iter_mut().skip(1) {
            s.pos[0] += 0.5 * s.vel[0] * dt;
            s.pos[1] += 0.5 * s.vel[1] * dt;
        }

        for i in 0..stars.len() {
            let pos = stars[i].pos;
            let mut a = [0.0, 0.0];
            for other in stars.iter() {
                let d = [other.pos[0] - pos[0], other.pos[1] - pos[1]];
                let inv_dist = 1.0 / (d[0] * d[0] + d[1] * d[1] + EPS * EPS).sqrt();
                let k = G * other.mass * inv_dist * inv_dist * inv_dist;
                a[0] += d[0] * k;
                a[1] += d[1] * k;
            }
            stars[i].vel[0] += a[0] * dt;
            stars[i].vel[1] += a[1] * dt;
        }

        for s in stars.iter_mut().skip(1) {
            s.pos[0] += 0.5 * s.vel[0] * dt;
            s.pos[1] += 0.5 * s.vel[1] * dt;
        }

        gpu.write(buffer, stars);
    }
}
