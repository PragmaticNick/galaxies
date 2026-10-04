struct CameraUniform {
    proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct Star {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    radius: f32,
    _pad: vec2<f32>,
};

@group(1) @binding(0) var<storage, read> stars: array<Star>;

struct SimParams {
    dt: f32,
    eps: f32,
    g: f32,
    radius: f32,
    center: vec2<f32>,
};
@group(2) @binding(0) var<uniform> sim: SimParams;

// Radial palette sampled from galaxy.jpg: blue-white core, gold ring, orange-pink rim.
const PALETTE_STEPS: u32 = 6u;
const PALETTE: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(0.80, 0.85, 1.00), // t = 0.0  blue-white core
    vec3<f32>(0.88, 0.89, 1.00), // t = 0.2
    vec3<f32>(1.00, 0.92, 0.80), // t = 0.4  warm white
    vec3<f32>(1.00, 0.82, 0.45), // t = 0.6  gold
    vec3<f32>(1.00, 0.66, 0.40), // t = 0.8  orange
    vec3<f32>(1.00, 0.58, 0.62), // t = 1.0  salmon-pink rim
);

fn arm_color(t: f32) -> vec3<f32> {
    let x = t * f32(PALETTE_STEPS - 1u);
    let i = min(u32(x), PALETTE_STEPS - 2u);
    var palette = PALETTE;
    return mix(palette[i], palette[i + 1u], x - f32(i));
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) local: vec2<f32>,
};

const QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),

    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    let star_index = i / 6u;
    let corner = i % 6u;

    let s = stars[star_index];
    let local = QUAD[corner];

    let world_pos = s.pos + local * s.radius;

    let dist = length(s.pos - sim.center);
    let t = clamp(dist / sim.radius, 0.0, 1.0);
    var color = arm_color(t) * 0.25;
    if (star_index == 0u) {
        color = vec3<f32>(1.0, 1.0, 1.0);
    }

    var out: VertexOutput;
    out.clip_position = camera.proj * vec4<f32>(world_pos, 0.0, 1.0);
    out.local = local;
    out.color = color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.local);

    if (dist > 1.0) {
        discard;
    }

    return vec4<f32>(in.color, 1.0);
}


