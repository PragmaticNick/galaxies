struct Star {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    radius: f32,
    _pad: vec2<f32>,
    color: vec3<f32>,
};

@group(0) @binding(0) var<storage, read_write> stars: array<Star>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let index = global_invocation_id.x;
    let total = arrayLength(&stars);

    if (index >= total) {
        return;
    }

    stars[index].pos += vec2(1.0, 1.0);
}