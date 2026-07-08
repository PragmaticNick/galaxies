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

    var out: VertexOutput;
    out.clip_position = camera.proj * vec4<f32>(world_pos, 0.0, 1.0);
    out.local = local;
    out.color = s.color;

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


