mod direct;
mod fmm;

use crate::star::Star;

/// Gravitational constant (tuned for the sim, not physical).
pub const G: f32 = 100.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PhysicsStrategy {
    Direct,
    Rayon,
    FmmSerial,
    FmmRayon,
}

impl PhysicsStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            PhysicsStrategy::Direct => "direct",
            PhysicsStrategy::Rayon => "rayon",
            PhysicsStrategy::FmmSerial => "fmm",
            PhysicsStrategy::FmmRayon => "fmm (rayon)",
        }
    }
}

/// Advance the sim one step: compute accelerations via `strategy`, then integrate.
/// Body 0 (galactic core) is held fixed.
pub fn update_physics(stars: &mut [Star], dt: f32, strategy: PhysicsStrategy) {
    let accels = match strategy {
        PhysicsStrategy::Direct => direct::direct(stars),
        PhysicsStrategy::Rayon => direct::direct_rayon(stars),
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

/// Softened pairwise gravity: force on `target` due to `source`.
pub(crate) fn gravity(target: &Star, source: &Star) -> [f32; 2] {
    const SOFTENING2: f32 = 400.0;
    let dx = source.pos[0] - target.pos[0];
    let dy = source.pos[1] - target.pos[1];
    let r2 = dx * dx + dy * dy + SOFTENING2;
    let r = r2.sqrt();
    let f = G * target.mass * source.mass / r2;
    [f * dx / r, f * dy / r]
}
