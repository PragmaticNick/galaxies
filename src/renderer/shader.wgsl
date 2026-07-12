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

fn arm_color(t: f32) -> vec3<f32> {
    let white = vec3<f32>(1.0, 1.0, 1.0);
    let cyan = vec3<f32>(0.3, 0.85, 1.0);
    let purple = vec3<f32>(0.6, 0.2, 0.9);

    if (t < 0.5) {
        return mix(white, cyan, t / 0.5);
    } else {
        return mix(cyan, purple, (t - 0.5) / 0.5);
    }
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


