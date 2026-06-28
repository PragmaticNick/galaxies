use std::f32::consts::PI;

use rand::RngExt;

use crate::square::Square;

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

pub fn generate_galaxy(config: &GalaxyConfig) -> Vec<Square> {
    let mut result = vec![];
    result.push(Square {
        center: config.center,
        radius: config.star_radius * 3.0,
        color: [1.0, 0.2, 0.2],
    });

    let mut rng = rand::rng();
    let arm_angle = 2.0 * PI / config.arm_count as f32;
    println!("{}", arm_angle);

    for _ in 0..config.star_count {
        let mut r: f32 = rng.random();
        let mut phi: f32 = 2.0 * PI * rng.random::<f32>();
        let arm_offset: f32 = config.arm_max_offset * (rng.random::<f32>() - 0.5) / r;

        let mut squared_arm_offset = arm_offset * arm_offset;
        if arm_offset < 0.0 {
            squared_arm_offset *= -1.0;
        }

        phi = (phi / arm_angle).floor() * arm_angle
            + r * config.arm_rotation_factor
            + squared_arm_offset;
        r = config.gap + r * (config.radius - config.gap);
        let x = config.center[0] + r * phi.cos();
        let y = config.center[1] + r * phi.sin();

        // let M = config.core_mass + config.star_mass;

        result.push(Square {
            center: [x, y],
            radius: config.star_radius,
            color: [1.0, 0.2, 0.2],
        });
    }

    result
}
