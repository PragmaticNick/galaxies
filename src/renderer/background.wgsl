struct BgStar {
    pos: vec2<f32>,
    radius: f32,
    opacity: f32,
    color: vec3<f32>,
    period: f32,
    phase: f32,
};

struct Params {
    time: f32,
    min_opacity: f32,
    screen: vec2<f32>,
    min_radius: f32,
};

@group(0) @binding(0) var<storage, read> stars: array<BgStar>;
@group(0) @binding(1) var<uniform> params: Params;

const PI: f32 = 3.14159265;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) local_px: vec2<f32>,
    @location(2) radius_px: f32,
    @location(3) alpha: f32,
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
    let s = stars[i / 6u];
    let local = QUAD[i % 6u];

    let osc = 0.5 + 0.5 * sin(2.0 * PI * params.time / s.period + s.phase);
    let radius_px = s.radius * params.screen.x * mix(params.min_radius, 1.0, osc);
    // one extra pixel of quad so the edge can be antialiased
    let half_px = radius_px + 1.0;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(s.pos + local * half_px * 2.0 / params.screen, 0.0, 1.0);
    out.color = s.color;
    out.local_px = local * half_px;
    out.radius_px = radius_px;
    out.alpha = s.opacity * mix(params.min_opacity, 1.0, osc);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coverage = clamp(in.radius_px + 0.5 - length(in.local_px), 0.0, 1.0);
    if (coverage <= 0.0) {
        discard;
    }
    return vec4<f32>(in.color, in.alpha * coverage);
}
