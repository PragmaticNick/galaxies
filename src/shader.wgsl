struct CameraUniform {
    proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) local: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) local: vec2<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.local = model.local;
    out.clip_position = camera.proj * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // local coords: 0=center, 1=circle edge, 2=quad edge (GLOW_FACTOR=2)
    let dist = length(in.local);

    // Solid circle with soft edge
    let circle = 1.0 - smoothstep(0.85, 1.0, dist);

    // Glow halo: exponential falloff from circle edge to quad edge
    let glow_t = clamp(dist - 1.0, 0.0, 1.0);   // 0 at circle edge, 1 at quad edge
    let glow = pow(1.0 - glow_t, 3.0) * 0.85;

    let alpha = circle + (1.0 - circle) * glow;
    if alpha < 0.005 { discard; }

    return vec4<f32>(in.color, alpha);
}
