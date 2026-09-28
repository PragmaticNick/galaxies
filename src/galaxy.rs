use std::f32::consts::PI;

use rand::RngExt;

use crate::config::{GAP_RAMP_WIDTH, EDGE_RAMP_WIDTH, ARM_TWIST, ARM_SPREAD, ARM_FRACTION, BULGE_RADIUS, BULGE_COLOR, YOUNG_COLORS, OLD_COLORS, HII_FRACTION, HII_CLUMPS, HII_SPREAD, CLUSTER_COLOR, CLUSTER_RADIUS, LUMINOSITY_MAX, LUMINOSITY_POWER, LUMINOSITY_SIZE, HII_COLOR, EPS, G};
use crate::star::Star;

pub struct GalaxyConfig {
    pub center: [f32; 2],
    pub velocity: [f32; 2],
    pub radius: f32,
    pub star_count: u32,
    pub core_mass: f32,
    pub core_radius: f32,
    pub star_mass: f32,
    pub star_radius: f32,
    pub gap: f32,
    pub arms: u32,
    pub clusters: u32,
    pub cluster_mass: f32,
}

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

fn luminosity(rng: &mut impl RngExt) -> f32 {
    let u: f32 = rng.random_range(1e-6f32..1.0);
    let mean = LUMINOSITY_POWER / (LUMINOSITY_POWER - 1.0);
    (u.powf(-1.0 / LUMINOSITY_POWER) / mean).min(LUMINOSITY_MAX)
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|c| a[c] + (b[c] - a[c]) * t)
}

pub fn generate_galaxy(config: &GalaxyConfig) -> Vec<Star> {
    let mut rng = rand::rng();
    let r_max = config.radius * (1.0 + EDGE_RAMP_WIDTH * 2.0);

    let arm_angle = |arm: u32, t: f32| arm as f32 * 2.0 * PI / config.arms as f32 + ARM_TWIST * t;
    let clumps: Vec<(f32, f32)> = (0..HII_CLUMPS)
        .map(|_| {
            let t = rng.random_range(0.2..1.0);
            (t * config.radius, arm_angle(rng.random_range(0..config.arms), t))
        })
        .collect();

    let mut disk: Vec<(f32, f32, [f32; 3])> = (0..config.star_count)
        .map(|_| {
            if rng.random::<f32>() < HII_FRACTION {
                let (r, phi) = clumps[rng.random_range(0..HII_CLUMPS)];
                let spread = HII_SPREAD * config.radius;
                let r = (r + gaussian(&mut rng) * spread).max(config.gap);
                return (r, phi + gaussian(&mut rng) * spread / r, HII_COLOR);
            }

            let r = loop {
                let candidate = rng.random::<f32>().sqrt() * r_max;
                if rng.random::<f32>() < radial_weight(candidate, config.gap, config.radius) {
                    break candidate;
                }
            };
            let t = (r / config.radius).min(1.0);

            let in_arm = rng.random::<f32>() < ARM_FRACTION;
            let (phi, colors) = if in_arm {
                let arm = rng.random_range(0..config.arms);
                (arm_angle(arm, t) + gaussian(&mut rng) * ARM_SPREAD, YOUNG_COLORS)
            } else {
                (2.0 * PI * rng.random::<f32>(), OLD_COLORS)
            };

            let color = colors[rng.random_range(0..colors.len())];
            let bulge = (-(t / BULGE_RADIUS).powi(2)).exp();
            (r, phi, mix(color, BULGE_COLOR, bulge))
        })
        .collect();

    disk.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut clusters: Vec<(f32, f32)> = (0..config.clusters)
        .map(|_| {
            let t = rng.random_range(0.15..1.0);
            let arm = rng.random_range(0..config.arms);
            (t * config.radius, arm_angle(arm, t) + gaussian(&mut rng) * ARM_SPREAD)
        })
        .collect();
    clusters.sort_by(|a, b| a.0.total_cmp(&b.0));
    let cluster_radii: Vec<f32> = clusters.iter().map(|c| c.0).collect();

    let [cx, cy] = config.center;
    let [vx, vy] = config.velocity;
    let mut stars = vec![Star {
        pos: config.center,
        vel: config.velocity,
        mass: config.core_mass,
        radius: config.core_radius,
        color: [1.0, 1.0, 1.0],
        ..bytemuck::Zeroable::zeroed()
    }];

    let orbit = |r: f32, phi: f32, disk_inside: usize| {
        let clusters_inside = cluster_radii.partition_point(|&x| x < r);
        let m_enclosed = config.core_mass
            + config.star_mass * disk_inside as f32
            + config.cluster_mass * clusters_inside as f32;
        let softened = (r * r + EPS * EPS).powf(1.5);
        let v = -(G * m_enclosed * r * r / softened).sqrt();
        (
            [cx + r * phi.cos(), cy + r * phi.sin()],
            [vx - phi.sin() * v, vy + phi.cos() * v],
        )
    };

    for (i, &(r, phi, color)) in disk.iter().enumerate() {
        let (pos, vel) = orbit(r, phi, i);
        let l = luminosity(&mut rng);
        stars.push(Star {
            pos,
            vel,
            mass: config.star_mass,
            radius: config.star_radius * l.max(1.0).powf(LUMINOSITY_SIZE),
            color: color.map(|c| c * l),
            ..bytemuck::Zeroable::zeroed()
        });
    }

    for &(r, phi) in &clusters {
        let (pos, vel) = orbit(r, phi, disk.partition_point(|d| d.0 < r));
        stars.push(Star {
            pos,
            vel,
            mass: config.cluster_mass,
            radius: CLUSTER_RADIUS,
            color: CLUSTER_COLOR,
            ..bytemuck::Zeroable::zeroed()
        });
    }

    stars
}
