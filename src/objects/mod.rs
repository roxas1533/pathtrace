pub mod base;
pub mod material;
pub mod object;
pub mod shape;

pub use base::{HitRecord, Hittable};
pub use material::Mirror;
pub use object::Object;
pub use shape::SphereShape;
