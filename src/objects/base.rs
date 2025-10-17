use crate::camera::Ray;
use crate::math::Vector3;

/// レイとオブジェクトの交差情報
#[derive(Debug, Clone, Copy)]
pub struct HitRecord {
    /// 交差点の座標
    pub point: Vector3,
    /// 交差点の法線ベクトル（正規化済み）
    pub normal: Vector3,
    /// レイの原点からの距離
    pub t: f64,
    /// レイが表面から入射したか（true: 表面, false: 裏面）
    pub front_face: bool,
}

impl HitRecord {
    /// HitRecordを作成し、法線の向きを自動調整
    pub fn new(point: Vector3, outward_normal: Vector3, t: f64, ray: &Ray) -> Self {
        let front_face = ray.direction.dot(&outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            -outward_normal
        };

        Self {
            point,
            normal,
            t,
            front_face,
        }
    }
}

/// レイと交差判定可能なオブジェクトのトレイト（交差判定のみ）
pub trait Hittable: Send + Sync {
    /// レイとの交差判定
    ///
    /// # Arguments
    /// * `ray` - 判定するレイ
    /// * `t_min` - 有効な距離の最小値
    /// * `t_max` - 有効な距離の最大値
    ///
    /// # Returns
    /// 交差した場合は `Some(HitRecord)`、しなかった場合は `None`
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
}
