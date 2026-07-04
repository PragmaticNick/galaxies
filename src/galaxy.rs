use std::f32::consts::PI;

use rand::RngExt;

use crate::physics::G;
use crate::star::Star;

pub struct GalaxyConfig {
    pub center: [f32; 2],
    pub radius: f32,
    pub star_count: u32,
    pub core_mass: f32,
    pub arm_count: u32,
    pub arm_rotation_factor: f32,
    pub arm_max_offset: f32,
    pub star_mass: f32,
    pub star_radius: f32,
    pub gap: f32,
}

pub fn generate_galaxy(config: &GalaxyConfig) -> Vec<Star> {
    let mut rng = rand::rng();
    let arm_angle = 2.0 * PI / config.arm_count as f32;

    let mut stars = vec![Star {
        pos: config.center,
        vel: [0.0, 0.0],
        mass: config.core_mass,
        radius: config.star_radius * 3.0,
        color: [1.0, 1.0, 1.0],
        ..bytemuck::Zeroable::zeroed()
    }];

    for _ in 0..config.star_count {
        let r_unit: f32 = rng.random();
        let phi_base: f32 = 2.0 * PI * rng.random::<f32>();

        let offset = config.arm_max_offset * (rng.random::<f32>() - 0.5) / r_unit;
        let phi = (phi_base / arm_angle).floor() * arm_angle
            + r_unit * config.arm_rotation_factor
            + offset * offset.abs();
        let r = config.gap + r_unit * (config.radius - config.gap);

        let v = -(G * config.core_mass / r.max(1.0)).sqrt();
        let t = (r / config.radius).clamp(0.0, 1.0);

        let [cr, cg, cb] = arm_color(t);

        stars.push(Star {
            pos: [config.center[0] + r * phi.cos(), config.center[1] + r * phi.sin()],
            vel: [-phi.sin() * v, phi.cos() * v],
            mass: config.star_mass,
            radius: config.star_radius,
            color: [cr * 0.25, cg * 0.25, cb * 0.25],
            ..bytemuck::Zeroable::zeroed()
        });
    }

    stars
}

fn arm_color(t: f32) -> [f32; 3] {
    const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
    const YELLOW: [f32; 3] = [1.0, 0.85, 0.3];
    const PURPLE: [f32; 3] = [0.6, 0.2, 0.9];

    if t < 0.5 {
        lerp(WHITE, YELLOW, t / 0.5)
    } else {
        lerp(YELLOW, PURPLE, (t - 0.5) / 0.5)
    }
}

fn lerp(a: [f32; 3], b: [f32; 3], s: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * s,
        a[1] + (b[1] - a[1]) * s,
        a[2] + (b[2] - a[2]) * s,
    ]
}
