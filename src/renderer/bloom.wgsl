override UPSAMPLE: bool;

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    let corners = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VertexOutput;
    out.position = vec4<f32>(corners[i], 0.0, 1.0);
    out.uv = vec2<f32>(0.5 + 0.5 * corners[i].x, 0.5 - 0.5 * corners[i].y);
    return out;
}

fn tap(uv: vec2<f32>, texel: vec2<f32>, x: f32, y: f32) -> vec3<f32> {
    return textureSample(source, linear, uv + texel * vec2<f32>(x, y)).rgb;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(source));
    let uv = in.uv;
    var color: vec3<f32>;
    if UPSAMPLE {
        color = (tap(uv, texel, -1.0, -1.0) + 2.0 * tap(uv, texel, 0.0, -1.0) + tap(uv, texel, 1.0, -1.0)
            + 2.0 * tap(uv, texel, -1.0, 0.0) + 4.0 * tap(uv, texel, 0.0, 0.0) + 2.0 * tap(uv, texel, 1.0, 0.0)
            + tap(uv, texel, -1.0, 1.0) + 2.0 * tap(uv, texel, 0.0, 1.0) + tap(uv, texel, 1.0, 1.0)) / 16.0;
    } else {
        color = (tap(uv, texel, -1.0, -1.0) + tap(uv, texel, 1.0, -1.0)
            + tap(uv, texel, -1.0, 1.0) + tap(uv, texel, 1.0, 1.0)) / 4.0;
    }
    return vec4<f32>(color, 1.0);
}
