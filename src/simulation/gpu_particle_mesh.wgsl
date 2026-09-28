override WORKGROUP_SIZE: u32 = 256u;

/// Largest padded FFT side a workgroup can hold in shared memory.
const MAX_P: u32 = 1024u;

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
    /// cells per side, the grid has size + 1 nodes per side
    size: u32,
    /// padded FFT side
    p: u32,
    log_p: u32,
    /// fixed-point units per unit of mass
    mass_scale: f32,
};

@group(0) @binding(0) var<storage, read_write> stars: array<Star>;
/// node masses in fixed point, (size + 1)², x-major
@group(0) @binding(1) var<storage, read_write> mass: array<atomic<i32>>;
/// p² complex values: node masses, then their FFT, then node accelerations
@group(0) @binding(2) var<storage, read_write> field: array<vec2<f32>>;
/// FFT of the gravity kernel, p²
@group(0) @binding(3) var<storage, read> kernel: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

struct Cell {
    i: i32,
    j: i32,
    dx: f32,
    dy: f32,
    ok: bool,
};

/// Lower-left node of the cell containing `pos` and the position inside it.
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

/// First half drift, then the star's mass spread over the 4 nodes of its cell.
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

/// Node masses into the top-left corner of the zero-padded p x p field.
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

/// In-place radix-2 FFT of one line of the field, held in shared memory.
/// Element k of line `l` is field[l * step + k * stride]. `sign` is -1 for the
/// forward transform, +1 for the inverse (unnormalized, like rustfft).
fn fft_line(l: u32, lid: u32, step: u32, stride: u32, sign: f32) {
    let p = params.p;

    // bit-reversed order, so the butterflies below can go in place
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

/// Convolution theorem: multiply by the kernel in frequency space.
@compute @workgroup_size(WORKGROUP_SIZE)
fn multiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= params.p * params.p {
        return;
    }
    field[index] = cmul(field[index], kernel[index]);
}

/// Node acceleration: real part is ax, imaginary part ay. Forward + inverse
/// FFT scale by p² in total.
fn node_acc(i: i32, j: i32) -> vec2<f32> {
    let p = params.p;
    return field[u32(i) * p + u32(j)] / f32(p * p);
}

/// Acceleration interpolated back from the 4 nodes, kick, second half drift.
@compute @workgroup_size(WORKGROUP_SIZE)
fn kick_drift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&stars) {
        return;
    }

    var pos = stars[index].pos;
    var vel = stars[index].vel;

    let c = cell(pos);
    if c.ok {
        let a = node_acc(c.i, c.j) * (1.0 - c.dx) * (1.0 - c.dy)
            + node_acc(c.i, c.j + 1) * (1.0 - c.dx) * c.dy
            + node_acc(c.i + 1, c.j) * c.dx * (1.0 - c.dy)
            + node_acc(c.i + 1, c.j + 1) * c.dx * c.dy;
        vel += a * params.dt;
    }

    if index != 0u {
        pos += 0.5 * vel * params.dt;
    }

    stars[index].pos = pos;
    stars[index].vel = vel;
}
