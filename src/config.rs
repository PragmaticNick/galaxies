use crate::galaxy::GalaxyConfig;
use crate::simulation::Strategy;

pub const STRATEGY: Strategy = Strategy::GpuParticleMeshFft { size: 1023, h: 5.0 };
// pub const STRATEGY: Strategy = Strategy::ParticleMesh { size: 128, h: 10.0 };
// pub const STRATEGY: Strategy = Strategy::GpuDirect;

pub const G: f32 = 100.0;
pub const EPS: f32 = 2.0;
pub const TIME_SCALE: f32 = 0.5;

pub const GALAXY: GalaxyConfig = GalaxyConfig {
    center: [0.0, 0.0],
    radius: 400.0,
    star_count: 2500000,
    core_radius: 2.0,
    core_mass: 500000.0,
    star_mass: 0.1,
    star_radius: 0.5,
    gap: 20.0,
    arms: 5,
};

pub const GAP_RAMP_WIDTH: f32 = 0.7;
pub const EDGE_RAMP_WIDTH: f32 = 0.15;

pub const ARM_TWIST: f32 = 2.5;
pub const ARM_SPREAD: f32 = 0.55;
pub const ARM_FRACTION: f32 = 0.45;

pub const BULGE_RADIUS: f32 = 0.25;
pub const BULGE_COLOR: [f32; 3] = [0.35, 0.2, 0.07];
pub const YOUNG_COLORS: [[f32; 3]; 2] = [[0.08, 0.16, 0.45], [0.15, 0.25, 0.45]];
pub const OLD_COLORS: [[f32; 3]; 2] = [[0.12, 0.06, 0.02], [0.1, 0.04, 0.015]];

pub const HII_FRACTION: f32 = 0.03;
pub const HII_CLUMPS: usize = 300;
pub const HII_SPREAD: f32 = 0.01;
pub const HII_COLOR: [f32; 3] = [0.6, 0.08, 0.25];

pub const VIEW_RADIUS: f32 = 800.0;
pub const ZOOM_STEP: f32 = 0.9;

pub const MIN_RADIUS_PX: f32 = 1.0;
pub const GLOW_SIZE: f32 = 3.0;
pub const GLOW_STRENGTH: f32 = 0.3;
pub const EXPOSURE: f32 = 0.4;
pub const GRID_BRIGHTNESS: f32 = 0.3;
