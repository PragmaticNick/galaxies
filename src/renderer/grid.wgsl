@group(0) @binding(0) var<uniform> view: vec4<f32>;

const CORNERS: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
    vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>(-1.0, -1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(CORNERS[i] * view.z * view.xy, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.3, 0.3, 0.3, 1.0);
}
