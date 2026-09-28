struct View {
    scale: vec2<f32>,
    center: vec2<f32>,
    grid_half: f32,
    world_per_pixel: f32,
    grid_cells: f32,
};

override GRID_BRIGHTNESS: f32;
override GRID_CELL_BRIGHTNESS: f32;

@group(0) @binding(0) var<uniform> view: View;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) brightness: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    let lines = u32(view.grid_cells) + 1u;
    let line = i / 2u;
    let k = line % lines;
    let across = -view.grid_half + f32(k) * 2.0 * view.grid_half / view.grid_cells;
    let along = select(-view.grid_half, view.grid_half, i % 2u == 1u);
    let world = select(vec2<f32>(along, across), vec2<f32>(across, along), line < lines);

    var out: VertexOutput;
    out.position = vec4<f32>((world - view.center) * view.scale, 0.0, 1.0);
    out.brightness = select(GRID_CELL_BRIGHTNESS, GRID_BRIGHTNESS, k == 0u || k == lines - 1u);
    return out;
}

@fragment
fn fs_main(@location(0) brightness: f32) -> @location(0) vec4<f32> {
    return vec4<f32>(vec3<f32>(brightness), 1.0);
}
