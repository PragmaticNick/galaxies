use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            proj: Self::ortho(width, height),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.proj = Self::ortho(width, height);
    }

    fn ortho(width: u32, height: u32) -> [[f32; 4]; 4] {
        let w = width.max(1) as f32;
        let h = height.max(1) as f32;
        [
            [2.0 / w, 0.0, 0.0, 0.0],
            [0.0, 2.0 / h, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}
