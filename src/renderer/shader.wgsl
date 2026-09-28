const MIN_RADIUS_PX: f32 = 1.0;

struct View {
    scale: vec2<f32>,
    center: vec2<f32>,
    grid_half: f32,
    world_per_pixel: f32,
};

@group(0) @binding(0) var<uniform> view: View;

struct Star {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    radius: f32,
    _pad: vec2<f32>,
    color: vec3<f32>,
    _pad2: f32,
};

@group(1) @binding(0) var<storage, read> stars: array<Star>;

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

    let radius = max(s.radius, MIN_RADIUS_PX * view.world_per_pixel);
    let world_pos = s.pos + local * radius;

    var out: VertexOutput;
    out.clip_position = vec4<f32>((world_pos - view.center) * view.scale, 0.0, 1.0);
    out.local = local;
    out.color = s.color * (s.radius * s.radius) / (radius * radius);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.local);
    let edge = fwidth(dist);
    let coverage = 1.0 - smoothstep(1.0 - edge, 1.0 + edge, dist);
    return vec4<f32>(in.color * coverage, 1.0);
}


