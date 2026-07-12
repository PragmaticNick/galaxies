use std::f32::consts::PI;

use rand::RngExt;

use crate::star::Star;

pub struct GalaxyConfig {
    pub center: [f32; 2],
    pub radius: f32,
    pub star_count: u32,
    pub core_mass: f32,
    pub core_radius: f32,
    pub star_mass: f32,
    pub star_radius: f32,
    pub gap: f32,
}

pub fn generate_galaxy(config: &GalaxyConfig, eps: f32, g: f32) -> Vec<Star> {
    let mut rng = rand::rng();

    let disk_mass_total = config.star_mass * config.star_count as f32;
    let disk_span = (config.radius - config.gap).max(1.0);

    let mut stars = vec![Star {
        pos: config.center,
        vel: [0.0, 0.0],
        mass: config.core_mass,
        radius: config.core_radius,
        ..bytemuck::Zeroable::zeroed()
    }];

    for _ in 0..config.star_count {
        let r_unit: f32 = rng.random();
        let phi: f32 = 2.0 * PI * rng.random::<f32>();

        let r = config.gap + r_unit * disk_span;

        let r_norm = ((r - config.gap) / disk_span).clamp(0.0, 1.0);
        let m_enclosed = config.core_mass + disk_mass_total * r_norm;
        let softened = (r * r + eps * eps).powf(1.5);
        let v = -(g * m_enclosed * r * r / softened).sqrt();

        stars.push(Star {
            pos: [
                config.center[0] + r * phi.cos(),
                config.center[1] + r * phi.sin(),
            ],
            vel: [-phi.sin() * v, phi.cos() * v],
            mass: config.star_mass,
            radius: config.star_radius,
            ..bytemuck::Zeroable::zeroed()
        });
    }

    stars
}
