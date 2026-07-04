mod direct;
mod fmm;

use crate::star::Star;

pub const G: f32 = 100.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PhysicsStrategy {
    PlainLoop,
    Rayon,
    FmmSerial,
    FmmRayon,
}

impl PhysicsStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            PhysicsStrategy::PlainLoop => "plain loop",
            PhysicsStrategy::Rayon => "rayon",
            PhysicsStrategy::FmmSerial => "fmm (serial)",
            PhysicsStrategy::FmmRayon => "fmm (rayon)",
        }
    }
}

pub fn update_physics(stars: &mut [Star], dt: f32, strategy: PhysicsStrategy) {
    let accels = match strategy {
        PhysicsStrategy::PlainLoop => direct::accels_plain(stars),
        PhysicsStrategy::Rayon => direct::accels_rayon(stars),
        PhysicsStrategy::FmmSerial => fmm::accels_serial(stars),
        PhysicsStrategy::FmmRayon => fmm::accels_rayon(stars),
    };

    for i in 1..stars.len() {
        stars[i].vel[0] += accels[i][0] * dt;
        stars[i].vel[1] += accels[i][1] * dt;
        stars[i].pos[0] += stars[i].vel[0] * dt;
        stars[i].pos[1] += stars[i].vel[1] * dt;
    }
}

pub(crate) fn gravity(target: &Star, source: &Star) -> [f32; 2] {
    const SOFTENING2: f32 = 400.0;
    let dx = source.pos[0] - target.pos[0];
    let dy = source.pos[1] - target.pos[1];
    let r2 = dx * dx + dy * dy + SOFTENING2;
    let r = r2.sqrt();
    let f = G * target.mass * source.mass / r2;
    [f * dx / r, f * dy / r]
}
