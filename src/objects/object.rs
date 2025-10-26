use crate::camera::Ray;
use crate::math::Vector3;

use super::base::{HitRecord, Hittable};
use super::material::Material;
use super::shape::Shape;

/// 汎用的なオブジェクト（形状 + マテリアル）
pub struct Object {
    /// 形状
    pub shape: Box<dyn Shape>,
    /// マテリアル
    pub material: Box<dyn Material>,
}

impl Object {
    /// 新しいObjectを作成
    ///
    /// # Arguments
    /// * `shape` - 形状
    /// * `material` - マテリアル
    pub fn new(shape: Box<dyn Shape>, material: Box<dyn Material + 'static>) -> Self {
        Self { shape, material }
    }
}

impl Hittable for Object {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        self.shape.hit(ray, t_min, t_max)
    }
}

impl Object {
    /// BRDFとPDFを同時に取得
    pub fn brdf_pdf(
        &self,
        x: &Vector3,
        i: &Vector3,
        o: &Vector3,
        normal: &Vector3,
    ) -> (Vector3, f64) {
        self.material.brdf_pdf(x, i, o, normal)
    }

    /// サンプリング方向を生成（ジェネリック版）
    pub fn sample_direction<R: rand::Rng>(
        &self,
        normal: &Vector3,
        incoming: &Ray,
        rng: &mut R,
    ) -> Vector3 {
        self.material.sample_direction(normal, incoming, rng)
    }
}
