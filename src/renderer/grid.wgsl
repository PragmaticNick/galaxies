struct View {
    scale: vec2<f32>,
    center: vec2<f32>,
    grid_half: f32,
    world_per_pixel: f32,
};

@group(0) @binding(0) var<uniform> view: View;

const CORNERS: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
    vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>(-1.0, -1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>((CORNERS[i] * view.grid_half - view.center) * view.scale, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.3, 0.3, 0.3, 1.0);
}
