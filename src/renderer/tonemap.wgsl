override EXPOSURE: f32;

@group(0) @binding(0) var hdr: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let corners = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4<f32>(corners[i], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    let color = textureLoad(hdr, vec2<i32>(pixel.xy), 0).rgb;
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let mapped = 1.0 - exp(-luminance * EXPOSURE);
    return vec4<f32>(min(color * mapped / max(luminance, 1e-6), vec3<f32>(1.0)), 1.0);
}
