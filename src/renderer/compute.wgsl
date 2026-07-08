const WORKGROUP_SIZE: u32 = 256;
const G: f32 = 100.0;
const EPS: f32 = 20.0;

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
fn drift_half(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let index = gid.x;
    let stride = num_wg.x * WORKGROUP_SIZE;
    let total = arrayLength(&stars);
    let dt = sim.dt;
    for (var i: u32 = index; i < total; i += stride) {
        if (i == 0u) { continue; }
        stars[i].pos += 0.5 * stars[i].vel * dt;
    }
}

var<workgroup> tile_memory: array<vec4<f32>, WORKGROUP_SIZE>;

fn calculate_tile(pos: vec2<f32>, mass: f32) -> vec2<f32> {
    var f: vec2<f32> = vec2(0.0, 0.0);
    for (var i: u32 = 0; i < WORKGROUP_SIZE; i += 1) {
        f += gravity_force(pos, mass, tile_memory[i].xy, tile_memory[i].z);
    }

    return f;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn kick(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let global = gid.x;
    let local = lid.x;

    let total = arrayLength(&stars);
    let dt = sim.dt;

    var t = stars[global];

    let pos = t.pos;
    let mass = t.mass;

    let tile_size = WORKGROUP_SIZE;
    let tile_count = (total + tile_size - 1u) / tile_size;

    var force = vec2(0.0, 0.0);

    for (var tile: u32 = 0; tile < tile_count; tile++) {
        var load_index = tile * WORKGROUP_SIZE + local;
        tile_memory[local] = vec4(stars[load_index].pos, stars[load_index].mass, 0.0);

        workgroupBarrier();

        force += calculate_tile(pos, mass);
        workgroupBarrier();
    }

    let a = force / mass;
    t.vel += a * dt;
    t._pad = pos + 0.5 * t.vel * dt;

    stars[global] = t;

    // for (var i: u32 = index; i < total; i += stride) {
    //     if (i == 0u) { continue; }
    //     var a: vec2<f32> = vec2(0.0, 0.0);
    //     var t = stars[i];
    //     for (var j: u32 = 0u; j < total; j += 1u) {
    //         if (j == i) { continue; }
    //         let s = stars[j];
    //         let f = gravity_force(t.pos, t.mass, s.pos, s.mass);
    //         a += f / t.mass;
    //     }
    //     t.vel += a * dt;
    //     t._pad = t.pos + 0.5 * t.vel * dt;
    //     stars[i] = t;
    // }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn commit(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let index = gid.x;
    let stride = num_wg.x * WORKGROUP_SIZE;
    let total = arrayLength(&stars);
    for (var i: u32 = index; i < total; i += stride) {
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