use crate::physics::G;

#[derive(Clone, Copy)]
pub struct Multipole {
    pub center: [f32; 2],
    values: [f32; 6],
}

impl Multipole {
    pub fn new(center: [f32; 2]) -> Self {
        Self {
            center,
            values: [0.0; 6],
        }
    }

    pub fn add_point(&mut self, pos: [f32; 2], mass: f32) {
        let dx = pos[0] - self.center[0];
        let dy = pos[1] - self.center[1];
        self.values[0] += mass;
        self.values[1] += dx * mass;
        self.values[2] += dy * mass;
        self.values[3] += dx * dx * mass * 0.5;
        self.values[4] += dx * dy * mass;
        self.values[5] += dy * dy * mass * 0.5;
    }

    pub fn add_multipole(&mut self, other: &Multipole) {
        let t = other.translate(self.center);
        for i in 0..6 {
            self.values[i] += t.values[i];
        }
    }

    pub fn translate(&self, pos: [f32; 2]) -> Multipole {
        let dx = self.center[0] - pos[0];
        let dy = self.center[1] - pos[1];
        let v = &self.values;
        Multipole {
            center: pos,
            values: [
                v[0],
                v[1] + dx * v[0],
                v[2] + dy * v[0],
                v[3] + dx * v[1] + 0.5 * dx * dx * v[0],
                v[4] + dy * v[1] + dx * v[2] + dx * dy * v[0],
                v[5] + dy * v[2] + 0.5 * dy * dy * v[0],
            ],
        }
    }

    pub fn eval(&self, target_pos: [f32; 2], mt: f32) -> [f32; 2] {
        let a = target_pos[0] - self.center[0];
        let b = target_pos[1] - self.center[1];

        let d = (a * a + b * b).sqrt();
        let d3 = d * d * d;
        let d5 = d3 * d * d;
        let d7 = d5 * d * d;
        let v = &self.values;

        let mut rx = 0.0;
        rx += v[0] * (-a) / d3;
        rx += v[1] * (b * b - 2.0 * a * a) / d5;
        rx += v[2] * -3.0 * a * b / d5;
        rx += v[3] * (9.0 * a * b * b - 6.0 * a * a * a) / d7;
        rx += v[4] * (3.0 * b * b - 12.0 * a * a * b) / d7;
        rx += v[5] * (3.0 * a * a * a - 12.0 * a * b * b) / d7;

        let mut ry = 0.0;
        ry += v[0] * (-b) / d3;
        ry += v[1] * -3.0 * a * b / d5;
        ry += v[2] * (a * a - 2.0 * b * b) / d5;
        ry += v[3] * (3.0 * b * b * b - 12.0 * a * b * b) / d7;
        ry += v[4] * (3.0 * a * a * a - 12.0 * a * b * b) / d7;
        ry += v[5] * (9.0 * a * a * b - 6.0 * b * b * b) / d7;

        [G * mt * rx, G * mt * ry]
    }
}
