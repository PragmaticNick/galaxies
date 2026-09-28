use std::f32::consts::PI;

use rand::RngExt;

use crate::simulation::{EPS, G};
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
    pub arms: u32,
}

const GAP_RAMP_WIDTH: f32 = 0.7;
const EDGE_RAMP_WIDTH: f32 = 0.15;

const WOLF_RAYET_FRACTION: f32 = 0.04;
const WOLF_RAYET_COLOR: [f32; 3] = [0.36, 0.12, 0.6];

const ARM_TWIST: f32 = 2.5;
const ARM_SPREAD: f32 = 0.55;
const ARM_FRACTION: f32 = 0.45;

const SPECTRUM: [[f32; 3]; 7] = [
    [0.55, 0.75, 1.0],
    [0.65, 0.8, 1.0],
    [0.85, 0.9, 1.0],
    [1.0, 0.95, 0.8],
    [1.0, 0.75, 0.25],
    [1.0, 0.55, 0.2],
    [1.0, 0.35, 0.2],
];
const SPECTRUM_BRIGHTNESS: f32 = 0.25;

fn gaussian(rng: &mut impl RngExt) -> f32 {
    let u1: f32 = rng.random_range(1e-6f32..1.0);
    let u2: f32 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn radial_weight(r: f32, gap: f32, radius: f32) -> f32 {
    let gap_ramp = gap * GAP_RAMP_WIDTH;
    let edge_ramp = radius * EDGE_RAMP_WIDTH;
    let inner = smoothstep(gap - gap_ramp, gap + gap_ramp, r);
    let outer = 1.0 - smoothstep(radius - edge_ramp, radius + edge_ramp, r);
    inner * outer
}

fn spectrum_color(t: f32) -> [f32; 3] {
    let x = t * (SPECTRUM.len() - 1) as f32;
    let i = (x as usize).min(SPECTRUM.len() - 2);
    let f = x - i as f32;
    let (a, b) = (SPECTRUM[i], SPECTRUM[i + 1]);
    [0, 1, 2].map(|c| (a[c] + (b[c] - a[c]) * f) * SPECTRUM_BRIGHTNESS)
}

pub fn generate_galaxy(config: &GalaxyConfig) -> Vec<Star> {
    let mut rng = rand::rng();
    let r_max = config.radius * (1.0 + EDGE_RAMP_WIDTH * 2.0);

    let mut disk: Vec<(f32, f32, [f32; 3])> = (0..config.star_count)
        .map(|_| {
            let r = loop {
                let candidate = rng.random::<f32>().sqrt() * r_max;
                if rng.random::<f32>() < radial_weight(candidate, config.gap, config.radius) {
                    break candidate;
                }
            };
            let t = (r / config.radius).min(1.0);

            let phi = if rng.random::<f32>() < ARM_FRACTION {
                let arm = rng.random_range(0..config.arms) as f32;
                arm * 2.0 * PI / config.arms as f32 + ARM_TWIST * t + gaussian(&mut rng) * ARM_SPREAD
            } else {
                2.0 * PI * rng.random::<f32>()
            };

            let color = if rng.random::<f32>() < WOLF_RAYET_FRACTION {
                WOLF_RAYET_COLOR
            } else {
                spectrum_color(t)
            };

            (r, phi, color)
        })
        .collect();

    disk.sort_by(|a, b| a.0.total_cmp(&b.0));

    let [cx, cy] = config.center;
    let mut stars = vec![Star {
        pos: config.center,
        mass: config.core_mass,
        radius: config.core_radius,
        color: [1.0, 1.0, 1.0],
        ..bytemuck::Zeroable::zeroed()
    }];

    for (i, (r, phi, color)) in disk.into_iter().enumerate() {
        let m_enclosed = config.core_mass + config.star_mass * i as f32;
        let softened = (r * r + EPS * EPS).powf(1.5);
        let v = -(G * m_enclosed * r * r / softened).sqrt();

        stars.push(Star {
            pos: [cx + r * phi.cos(), cy + r * phi.sin()],
            vel: [-phi.sin() * v, phi.cos() * v],
            mass: config.star_mass,
            radius: config.star_radius,
            color,
            ..bytemuck::Zeroable::zeroed()
        });
    }

    stars
}
