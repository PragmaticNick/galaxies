use std::f32::consts::PI;

use rand::RngExt;

use crate::star::Star;

pub const G: f32 = 100.0;

pub struct GalaxyConfig {
    pub center: [f32; 2],
    pub radius: f32,
    pub star_count: u32,
    pub core_mass: f32,
    pub star_mass: f32,
    pub star_radius: f32,
}

pub fn generate_galaxy(config: &GalaxyConfig) -> Vec<Star> {
    let mut rng = rand::rng();

    let mut stars = vec![Star {
        pos: config.center,
        vel: [0.0, 0.0],
        mass: config.core_mass,
        radius: config.star_radius * 3.0,
        color: [1.0, 1.0, 1.0],
    }];

    for _ in 0..config.star_count {
        let r = config.radius * rng.random::<f32>().sqrt();
        let theta = 2.0 * PI * rng.random::<f32>();

        let v = (G * config.core_mass / r.max(1.0)).sqrt();

        // Warm golden-white core → cool blue-white edge.
        let t = r / config.radius;
        let brightness = 0.30 + 0.20 * (1.0 - t);
        let color = [
            (1.0 - t * 0.35) * brightness,
            (0.90 - t * 0.05) * brightness,
            (0.55 + t * 0.45) * brightness,
        ];

        stars.push(Star {
            pos: [config.center[0] + r * theta.cos(), config.center[1] + r * theta.sin()],
            vel: [-theta.sin() * v, theta.cos() * v],
            mass: config.star_mass,
            radius: config.star_radius,
            color,
        });
    }

    stars
}
