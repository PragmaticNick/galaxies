use std::f32::consts::PI;

use rand::RngExt;

use crate::settings::{
    ARM_FRACTION, ARM_SPREAD, ARM_TWIST, CORE_COLOR, DISK_BRIGHTNESS, EDGE_RAMP_WIDTH,
    GAP_RAMP_WIDTH, NUM_ARMS, SPECTRAL_GRADIENT, WOLF_RAYET_BRIGHTNESS, WOLF_RAYET_COLOR,
    WOLF_RAYET_FRACTION,
};
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

fn gaussian(rng: &mut impl rand::RngExt) -> f32 {
    let u1: f32 = rng.random_range(1e-6f32..1.0);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// Soft rejection-sampling weight so the disk fades in past the core gap and
// fades out at the rim instead of being cut off sharply.
fn radial_weight(r: f32, gap: f32, radius: f32) -> f32 {
    let gap_ramp = gap * GAP_RAMP_WIDTH;
    let edge_ramp = radius * EDGE_RAMP_WIDTH;
    let inner = smoothstep(gap - gap_ramp, gap + gap_ramp, r);
    let outer = 1.0 - smoothstep(radius - edge_ramp, radius + edge_ramp, r);
    inner * outer
}

pub fn generate_galaxy(config: &GalaxyConfig, eps: f32, g: f32) -> Vec<Star> {
    let mut rng = rand::rng();

    let r_max = config.radius * (1.0 + EDGE_RAMP_WIDTH * 2.0);

    // (r, phi, color) of every disk star
    let mut disk: Vec<(f32, f32, [f32; 3])> = (0..config.star_count)
        .map(|_| {
            let r = loop {
                let candidate = rng.random::<f32>().sqrt() * r_max;
                if rng.random::<f32>() < radial_weight(candidate, config.gap, config.radius) {
                    break candidate;
                }
            };
            let t = (r / config.radius).clamp(0.0, 1.0);

            let phi = if rng.random::<f32>() < ARM_FRACTION {
                let arm_index = rng.random_range(0..NUM_ARMS) as f32;
                let arm_angle = arm_index * (2.0 * PI / NUM_ARMS as f32) + ARM_TWIST * t;
                arm_angle + gaussian(&mut rng) * ARM_SPREAD
            } else {
                2.0 * PI * rng.random::<f32>()
            };

            let color = if rng.random::<f32>() < WOLF_RAYET_FRACTION {
                WOLF_RAYET_COLOR.map(|c| c * WOLF_RAYET_BRIGHTNESS)
            } else {
                arm_color(t).map(|c| c * DISK_BRIGHTNESS)
            };

            (r, phi, color)
        })
        .collect();

    // sorted by radius, a star's index is the number of disk stars inside it
    disk.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut stars = vec![Star {
        pos: config.center,
        mass: config.core_mass,
        radius: config.core_radius,
        color: CORE_COLOR,
        ..bytemuck::Zeroable::zeroed()
    }];

    for (i, (r, phi, color)) in disk.into_iter().enumerate() {
        let m_enclosed = config.core_mass + config.star_mass * i as f32;
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
            color,
            ..bytemuck::Zeroable::zeroed()
        });
    }

    stars
}

fn arm_color(t: f32) -> [f32; 3] {
    let n = (SPECTRAL_GRADIENT.len() - 1) as f32;
    let seg = t.clamp(0.0, 1.0) * n;
    let i = seg.min(n - 1.0).floor();
    let f = seg - i;
    let i = i as usize;
    lerp(SPECTRAL_GRADIENT[i], SPECTRAL_GRADIENT[i + 1], f)
}

fn lerp(a: [f32; 3], b: [f32; 3], s: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * s,
        a[1] + (b[1] - a[1]) * s,
        a[2] + (b[2] - a[2]) * s,
    ]
}
