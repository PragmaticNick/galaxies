use std::f32::consts::PI;

use rand::RngExt;

use crate::physics::G;
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

const GAP_RAMP_WIDTH: f32 = 0.7;
const EDGE_RAMP_WIDTH: f32 = 0.15;

const WOLF_RAYET_FRACTION: f32 = 0.04;
const WOLF_RAYET_COLOR: [f32; 3] = [0.6, 0.2, 1.0];

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

pub fn generate_galaxy(config: &GalaxyConfig) -> Vec<Star> {
    let mut rng = rand::rng();

    let mut stars = vec![Star {
        pos: config.center,
        vel: [0.0, 0.0],
        mass: config.core_mass,
        radius: config.core_radius,
        color: [1.0, 1.0, 1.0],
    }];

    let r_max = config.radius * (1.0 + EDGE_RAMP_WIDTH * 2.0);

    for _ in 0..config.star_count {
        let r = loop {
            let candidate = rng.random::<f32>().sqrt() * r_max;
            if rng.random::<f32>() < radial_weight(candidate, config.gap, config.radius) {
                break candidate;
            }
        };
        let phi: f32 = 2.0 * PI * rng.random::<f32>();

        let v = -(G * config.core_mass / r.max(1.0)).sqrt();
        let t = (r / config.radius).clamp(0.0, 1.0);

        let color = if rng.random::<f32>() < WOLF_RAYET_FRACTION {
            // Rare, extremely hot star that has blown off its outer layers,
            // exposing a violet-blue core far past the O-class end of the
            // normal spectral gradient.
            WOLF_RAYET_COLOR.map(|c| c * 0.6)
        } else {
            arm_color(t).map(|c| c * 0.25)
        };

        stars.push(Star {
            pos: [
                config.center[0] + r * phi.cos(),
                config.center[1] + r * phi.sin(),
            ],
            vel: [-phi.sin() * v, phi.cos() * v],
            mass: config.star_mass,
            radius: config.star_radius,
            color,
        });
    }

    stars
}

fn arm_color(t: f32) -> [f32; 3] {
    // Full O-B-A-F-G-K-M spectral-class gradient, boosted to full saturation
    // so it still reads as color once dimmed by the * 0.25 above.
    const O_HOT: [f32; 3] = [0.55, 0.75, 1.0];
    const B_BLUE: [f32; 3] = [0.65, 0.8, 1.0];
    const A_WHITE: [f32; 3] = [0.85, 0.9, 1.0];
    const F_YELLOW_WHITE: [f32; 3] = [1.0, 0.95, 0.8];
    const G_SUN: [f32; 3] = [1.0, 0.75, 0.25];
    const K_ORANGE: [f32; 3] = [1.0, 0.55, 0.2];
    const M_COOL: [f32; 3] = [1.0, 0.35, 0.2];

    let n = 6.0;
    let seg = t.clamp(0.0, 1.0) * n;
    let i = seg.min(n - 1.0).floor();
    let f = seg - i;

    match i as u32 {
        0 => lerp(O_HOT, B_BLUE, f),
        1 => lerp(B_BLUE, A_WHITE, f),
        2 => lerp(A_WHITE, F_YELLOW_WHITE, f),
        3 => lerp(F_YELLOW_WHITE, G_SUN, f),
        4 => lerp(G_SUN, K_ORANGE, f),
        _ => lerp(K_ORANGE, M_COOL, f),
    }
}

fn lerp(a: [f32; 3], b: [f32; 3], s: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * s,
        a[1] + (b[1] - a[1]) * s,
        a[2] + (b[2] - a[2]) * s,
    ]
}
