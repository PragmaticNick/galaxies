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
    color: vec3<f32>,
};

@group(1) @binding(0) var<storage, read> stars: array<Star>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) local: vec2<f32>,
};

const GLOW_FACTOR: f32 = 3.5;

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

    let world_pos = s.pos + local * s.radius * GLOW_FACTOR;

    var out: VertexOutput;
    out.clip_position = camera.proj * vec4<f32>(world_pos, 0.0, 1.0);
    out.local = local * GLOW_FACTOR;
    out.color = s.color;

    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.local);

    let circle = 1.0 - smoothstep(0.85, 1.0, dist);

    let glow_t = clamp((dist - 1.0) / (GLOW_FACTOR - 1.0), 0.0, 1.0);
    let glow = pow(1.0 - glow_t, 1.6) * 1.5;

    let alpha = circle + (1.0 - circle) * glow;
    if alpha < 0.004 { discard; }

    return vec4<f32>(in.color, alpha);
}


