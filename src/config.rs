use crate::galaxy::GalaxyConfig;

// Particle mesh grid: PM_GRID_SIZE x PM_GRID_SIZE cells of PM_CELL_SIZE world units, centered on
// the origin; stars outside it feel the whole system as one point mass.
pub const PM_GRID_SIZE: i32 = 1023;
pub const PM_CELL_SIZE: f32 = 15.0;

pub const G: f32 = 100.0;
pub const EPS: f32 = 15.0;
pub const TIME_SCALE: f32 = 1.0;
// galaxy i is generated from GALAXY_SEED + i, so every run starts from the same stars
pub const GALAXY_SEED: u64 = 1;

pub const GALAXIES: &[GalaxyConfig] = &[
    GalaxyConfig {
        center: [0.0, 0.0],
        velocity: [0.0, 0.0],
        radius: 1000.0,
        star_count: 9_999_999, // + the core = 10M
        core_radius: 2.0,
        core_mass: 500000.0,
        star_mass: 0.025,
        star_radius: 0.5,
        gap: 20.0,
        arms: 4,
    },
    // more entries, with their own center and velocity, make a collision
];

pub const GAP_RAMP_WIDTH: f32 = 0.7;
pub const EDGE_RAMP_WIDTH: f32 = 0.15;

pub const ARM_TWIST: f32 = 5.0;
pub const ARM_SPREAD: f32 = 0.4;
pub const ARM_FRACTION: f32 = 0.45;

pub const BULGE_RADIUS: f32 = 0.25;
pub const BULGE_COLOR: [f32; 3] = [0.35, 0.2, 0.07];
pub const YOUNG_COLORS: [[f32; 3]; 2] = [[0.08, 0.16, 0.45], [0.15, 0.25, 0.45]];
pub const OLD_COLORS: [[f32; 3]; 2] = [[0.12, 0.06, 0.02], [0.1, 0.04, 0.015]];

pub const HII_FRACTION: f32 = 0.03;
pub const HII_CLUMPS: usize = 300;
pub const HII_SPREAD: f32 = 0.01;
pub const HII_COLOR: [f32; 3] = [0.6, 0.08, 0.25];

pub const LUMINOSITY_POWER: f32 = 2.5;
pub const LUMINOSITY_MAX: f32 = 30.0;
pub const LUMINOSITY_SIZE: f32 = 0.5;

// half screen height in world units at startup
pub const VIEW_RADIUS: f32 = 1350.0;
pub const ZOOM_STEP: f32 = 0.9;
pub const PAN_SPEED: f32 = 0.8;
pub const ZOOM_SPEED: f32 = 2.5;

pub const MIN_RADIUS_PX: f32 = 1.0;
pub const GLOW_SIZE: f32 = 3.0;
pub const GLOW_STRENGTH: f32 = 0.3;
pub const EXPOSURE: f32 = 0.4;
pub const BLOOM_STRENGTH: f32 = 0.3;
pub const BLOOM_LEVELS: u32 = 6;
pub const SHOW_OVERLAY: bool = true;
// Twinkling starfield behind the galaxy.
pub const BG_COLOR: u32 = 0x0B0E1A;
// Stellar classification colors, each a (hot end, cool end) pair;
// every star picks a class and a random shade between its two ends.
pub const BG_STAR_COLORS: [(u32, u32); 7] = [
    (0xEAF2FF, 0x9BB8FF), // O: hot blue-white
    (0xF2F6FF, 0xBFD2FF), // B: blue-white
    (0xFFFFFF, 0xE6ECFF), // A: white
    (0xFFFDF2, 0xFCEFC7), // F: yellow-white
    (0xFFF4D6, 0xFBDD84), // G: yellow, Sun-like
    (0xFFD9A0, 0xF2A354), // K: orange
    (0xFFB08A, 0xE8683F), // M: red/orange
];
pub const BG_STAR_COUNT: u32 = 2000;
// as a fraction of the screen width
pub const BG_STAR_RADIUS_RANGE: (f32, f32) = (0.00014, 0.00035);
// baseline brightness each star pulses around
pub const BG_STAR_OPACITY_RANGE: (f32, f32) = (0.15, 0.5);
// seconds per pulse cycle
pub const BG_STAR_PERIOD_RANGE: (f32, f32) = (1.5, 4.0);
// at the dimmest/smallest point of a pulse, opacity and radius drop to this fraction
pub const BG_TWINKLE_MIN_OPACITY: f32 = 0.25;
pub const BG_TWINKLE_MIN_RADIUS: f32 = 0.7;
pub const BG_SEED: u64 = 7;
