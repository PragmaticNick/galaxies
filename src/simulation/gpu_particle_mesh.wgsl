override WORKGROUP_SIZE: u32 = 256u;

const MAX_P: u32 = 2048u;

const PI: f32 = 3.14159265358979;

struct Star {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    radius: f32,
    _pad: vec2<f32>,
    color: vec3<f32>,
    _pad2: f32,
};

struct Params {
    dt: f32,
    h: f32,
    origin: vec2<f32>,
    size: u32,
    p: u32,
    log_p: u32,
    mass_scale: f32,
    gm: f32,
    eps2: f32,
};

@group(0) @binding(0) var<storage, read_write> stars: array<Star>;
@group(0) @binding(1) var<storage, read_write> mass: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> field: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> kernel: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

struct Cell {
    i: i32,
    j: i32,
    dx: f32,
    dy: f32,
    ok: bool,
};

fn cell(pos: vec2<f32>) -> Cell {
    let g = (pos - params.origin) / params.h;
    let i = i32(floor(g.x));
    let j = i32(floor(g.y));
    let size = i32(params.size);
    let ok = i >= 0 && j >= 0 && i < size && j < size;
    return Cell(i, j, g.x - f32(i), g.y - f32(j), ok);
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn deposit(i: i32, j: i32, m: f32) {
    let n = i32(params.size) + 1;
    atomicAdd(&mass[i * n + j], i32(round(m)));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn drift_deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&stars) {
        return;
    }

    var pos = stars[index].pos;
    if index != 0u {
        pos += 0.5 * stars[index].vel * params.dt;
        stars[index].pos = pos;
    }

    let c = cell(pos);
    if !c.ok {
        return;
    }

    let m = stars[index].mass * params.mass_scale;
    deposit(c.i, c.j, m * (1.0 - c.dx) * (1.0 - c.dy));
    deposit(c.i, c.j + 1, m * (1.0 - c.dx) * c.dy);
    deposit(c.i + 1, c.j, m * c.dx * (1.0 - c.dy));
    deposit(c.i + 1, c.j + 1, m * c.dx * c.dy);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn load(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = params.p;
    let index = gid.x;
    if index >= p * p {
        return;
    }

    let row = index / p;
    let col = index % p;
    let n = params.size + 1u;

    var value = vec2(0.0, 0.0);
    if row < n && col < n {
        value.x = f32(atomicLoad(&mass[row * n + col])) / params.mass_scale;
    }
    field[index] = value;
}

var<workgroup> line: array<vec2<f32>, MAX_P>;

fn fft_line(l: u32, lid: u32, step: u32, stride: u32, sign: f32) {
    let p = params.p;

    for (var k = lid; k < p; k += WORKGROUP_SIZE) {
        line[reverseBits(k) >> (32u - params.log_p)] = field[l * step + k * stride];
    }
    workgroupBarrier();

    for (var half = 1u; half < p; half <<= 1u) {
        for (var k = lid; k < p / 2u; k += WORKGROUP_SIZE) {
            let pos = k % half;
            let i = (k / half) * 2u * half + pos;
            let j = i + half;

            let angle = sign * PI * f32(pos) / f32(half);
            let b = cmul(line[j], vec2(cos(angle), sin(angle)));
            let a = line[i];
            line[i] = a + b;
            line[j] = a - b;
        }
        workgroupBarrier();
    }

    for (var k = lid; k < p; k += WORKGROUP_SIZE) {
        field[l * step + k * stride] = line[k];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn fft_rows_forward(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    fft_line(wid.x, lid.x, params.p, 1u, -1.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn fft_cols_forward(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    fft_line(wid.x, lid.x, 1u, params.p, -1.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn fft_rows_inverse(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    fft_line(wid.x, lid.x, params.p, 1u, 1.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn fft_cols_inverse(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    fft_line(wid.x, lid.x, 1u, params.p, 1.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn multiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= params.p * params.p {
        return;
    }
    field[index] = cmul(field[index], kernel[index]);
}

fn node_acc(i: i32, j: i32) -> vec2<f32> {
    let p = params.p;
    return field[u32(i) * p + u32(j)] / f32(p * p);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn kick_drift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&stars) {
        return;
    }

    var pos = stars[index].pos;
    var vel = stars[index].vel;

    let c = cell(pos);
    var a: vec2<f32>;
    if c.ok {
        a = node_acc(c.i, c.j) * (1.0 - c.dx) * (1.0 - c.dy)
            + node_acc(c.i, c.j + 1) * (1.0 - c.dx) * c.dy
            + node_acc(c.i + 1, c.j) * c.dx * (1.0 - c.dy)
            + node_acc(c.i + 1, c.j + 1) * c.dx * c.dy;
    } else {
        let r2 = dot(pos, pos) + params.eps2;
        a = -params.gm * pos / (r2 * sqrt(r2));
    }
    vel += a * params.dt;

    if index != 0u {
        pos += 0.5 * vel * params.dt;
    }

    stars[index].pos = pos;
    stars[index].vel = vel;
}
