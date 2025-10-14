use crate::camera::Ray;
use crate::math::Vector3;

use super::base::HitRecord;

/// 形状を表すtrait（交差判定のみを担当）
pub trait Shape: Send + Sync {
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

/// 球形状
pub struct SphereShape {
    /// Center position of the sphere
    pub center: Vector3,
    /// Radius of the sphere
    pub radius: f64,
}

impl SphereShape {
    /// 新しい球形状を作成
    pub fn new(center: Vector3, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl Shape for SphereShape {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        // Ray-sphere intersection test
        // Ray: P(t) = origin + t * direction
        // Sphere: (P - center)^2 = r^2
        //
        // Substituting and expanding gives quadratic equation:
        // a*t^2 + b*t + c = 0
        // a = direction . direction
        // b = 2 * direction . (origin - center)
        // c = (origin - center) . (origin - center) - r^2

        let oc = ray.origin - self.center;
        let a = ray.direction.dot(&ray.direction);
        let half_b = oc.dot(&ray.direction);
        let c = oc.dot(&oc) - self.radius * self.radius;

        // Discriminant
        let discriminant = half_b * half_b - a * c;

        // No intersection
        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();

        // Find the nearest root that lies in the acceptable range
        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return None;
            }
        }

        // Calculate intersection information
        let point = ray.at(root);
        let outward_normal = (point - self.center) / self.radius;

        Some(HitRecord::new(point, outward_normal, root, ray))
    }
}
