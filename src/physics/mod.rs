mod direct;

use crate::star::Star;

/// Gravitational constant (tuned for the sim, not physical).
pub const G: f32 = 100.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PhysicsStrategy {
    PlainLoop,
    Rayon,
}

impl PhysicsStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            PhysicsStrategy::PlainLoop => "plain loop",
            PhysicsStrategy::Rayon => "rayon",
        }
    }
}

/// Advance the sim one step: compute accelerations via `strategy`, then integrate.
/// Body 0 (galactic core) is held fixed.
pub fn update_physics(stars: &mut [Star], dt: f32, strategy: PhysicsStrategy) {
    let accels = match strategy {
        PhysicsStrategy::PlainLoop => direct::accels_plain(stars),
        PhysicsStrategy::Rayon => direct::accels_rayon(stars),
    };

    for i in 1..stars.len() {
        stars[i].vel[0] += accels[i][0] * dt;
        stars[i].vel[1] += accels[i][1] * dt;
        stars[i].pos[0] += stars[i].vel[0] * dt;
        stars[i].pos[1] += stars[i].vel[1] * dt;
    }
}
