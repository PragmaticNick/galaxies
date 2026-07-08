use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Star {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub mass: f32,
    pub radius: f32,
    pub _pad: [f32; 2],
    pub color: [f32; 3],
    pub _pad2: f32,
}
