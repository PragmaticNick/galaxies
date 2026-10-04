// Every tunable knob of the simulation and its look, in one place.

use crate::galaxy::GalaxyConfig;

// ---------------------------------------------------------------- simulation

pub const G: f32 = 100.0;
// gravitational softening length
pub const EPS: f32 = 30.0;
// simulation time step per frame; scales simulation speed
pub const DT: f32 = 0.05 * 0.016;

// --------------------------------------------------------------- galaxy shape

pub const GALAXY: GalaxyConfig = GalaxyConfig {
    center: [0.0, 0.0],
    radius: 400.0,
    star_count: 50000,
    core_radius: 2.0,
    core_mass: 500000.0,
    star_mass: 10.0,
    star_radius: 1.5,
    gap: 20.0,
};

// soft fade-in past the core gap, as a fraction of `gap`
pub const GAP_RAMP_WIDTH: f32 = 0.7;
// soft fade-out at the rim, as a fraction of `radius`
pub const EDGE_RAMP_WIDTH: f32 = 0.15;

pub const NUM_ARMS: u32 = 2;
// radians of arm rotation from center to rim
pub const ARM_TWIST: f32 = 2.5;
// angular gaussian spread of stars around an arm, radians
pub const ARM_SPREAD: f32 = 0.55;
// fraction of disk stars placed on arms; the rest are uniform in angle
pub const ARM_FRACTION: f32 = 0.45;

// -------------------------------------------------------------- galaxy colors

pub const CORE_COLOR: [f32; 3] = [1.0, 1.0, 1.0];

// O-B-A-F-G-K-M spectral-class gradient from center (first) to rim (last),
// boosted to full saturation so it still reads as color once dimmed
pub const SPECTRAL_GRADIENT: [[f32; 3]; 7] = [
    [0.55, 0.75, 1.0], // O: hot
    [0.65, 0.8, 1.0],  // B: blue
    [0.85, 0.9, 1.0],  // A: white
    [1.0, 0.95, 0.8],  // F: yellow-white
    [1.0, 0.75, 0.25], // G: Sun
    [1.0, 0.55, 0.2],  // K: orange
    [1.0, 0.35, 0.2],  // M: cool
];
pub const DISK_BRIGHTNESS: f32 = 0.25;

// Rare, extremely hot star that has blown off its outer layers, exposing a
// violet-blue core far past the O-class end of the normal spectral gradient.
pub const WOLF_RAYET_FRACTION: f32 = 0.04;
pub const WOLF_RAYET_COLOR: [f32; 3] = [0.6, 0.2, 1.0];
pub const WOLF_RAYET_BRIGHTNESS: f32 = 0.6;

// ----------------------------------------------------------------- background

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
// in manim units (frame is 14.22 wide), same values as background.py
pub const BG_STAR_RADIUS_RANGE: (f32, f32) = (0.002, 0.005);
// baseline brightness each star pulses around
pub const BG_STAR_OPACITY_RANGE: (f32, f32) = (0.15, 0.5);
// seconds per pulse cycle
pub const BG_STAR_PERIOD_RANGE: (f32, f32) = (1.5, 4.0);
// at the dimmest/smallest point of a pulse, opacity and radius drop to this fraction
pub const BG_TWINKLE_MIN_OPACITY: f32 = 0.25;
pub const BG_TWINKLE_MIN_RADIUS: f32 = 0.7;
pub const BG_SEED: u64 = 7;

// ------------------------------------------------------------------------ gpu

// threads per compute workgroup; also the tile size of the gravity kernel
pub const WORKGROUP_SIZE: u32 = 256;
