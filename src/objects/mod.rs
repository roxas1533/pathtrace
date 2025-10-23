pub mod base;
pub mod material;
pub mod object;
pub mod shape;

pub use base::{HitRecord, Hittable};
pub use material::{Emissive, Mirror, OrenNayar};
pub use object::Object;
pub use shape::SphereShape;
