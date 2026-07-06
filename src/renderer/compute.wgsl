const WORKGROUP_SIZE: u32 = 256;
const G: f32 = 100.0;
const EPS: f32 = 20;

struct Star {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    radius: f32,
    _pad: vec2<f32>,
    color: vec3<f32>,
};

@group(0) @binding(0) var<storage, read_write> stars: array<Star>;

struct SimParams {
    dt: f32,
};
@group(0) @binding(1) var<uniform> sim: SimParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn drift_half(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let total = arrayLength(&stars);
    let dt = sim.dt * 0.1;
    for (var i: u32 = index; i < total; i += WORKGROUP_SIZE) {
        if (i == 0u) { continue; }
        stars[i].pos += 0.5 * stars[i].vel * dt;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let total = arrayLength(&stars);
    let dt = sim.dt * 0.1;
    for (var i: u32 = index; i < total; i += WORKGROUP_SIZE) {
        if (i == 0u) { continue; }
        var a: vec2<f32> = vec2(0.0, 0.0);
        var t = stars[i];
        for (var j: u32 = 0u; j < total; j += 1u) {
            if (j == i) { continue; }
            let s = stars[j];
            let f = gravity_force(t.pos, t.mass, s.pos, s.mass);
            a += f / t.mass;
        }
        t.vel += a * dt;
        t._pad = t.pos + 0.5 * t.vel * dt;
        stars[i] = t;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn commit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let total = arrayLength(&stars);
    for (var i: u32 = index; i < total; i += WORKGROUP_SIZE) {
        if (i == 0u) { continue; }
        stars[i].pos = stars[i]._pad;
    }
}

fn gravity_force(pos_a: vec2<f32>, mass_a: f32, pos_b: vec2<f32>, mass_b: f32) -> vec2<f32> {
    let d = pos_b - pos_a;
    let dist_sq = dot(d, d) + EPS * EPS;
    let inv_dist = inverseSqrt(dist_sq);
    let inv_dist_cube = inv_dist * inv_dist * inv_dist;
    return d * (G * mass_a * mass_b * inv_dist_cube);
}