use crate::camera::Ray;
use crate::math::Vector3;
use rand::{Rng, RngCore};

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

    /// 指定した点から見える形状の表面をサンプリング、またはPDF計算
    ///
    /// # Arguments
    /// * `hit_from` - 観測点（衝突点）
    /// * `target_hit` - MIS用：PDFを計算する目標点。Noneの場合はランダムサンプリング
    /// * `rng` - 乱数生成器
    ///
    /// # Returns
    /// (サンプリングされた点, 法線, 立体角測度PDF, 方向, 距離)
    fn sample_surface_from_point(
        &self,
        hit_from: &HitRecord,
        target_hit: Option<&HitRecord>,
        rng: &mut dyn RngCore,
    ) -> (Vector3, Vector3, f64, Vector3, f64);
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

    fn sample_surface_from_point(
        &self,
        hit_from: &HitRecord,
        target_hit: Option<&HitRecord>,
        rng: &mut dyn RngCore,
    ) -> (Vector3, Vector3, f64, Vector3, f64) {
        let to_center = self.center - hit_from.point;
        let distance_sq = to_center.dot(&to_center);
        let sin_theta_max_sq = (self.radius * self.radius) / distance_sq;
        let cos_theta_max = (1.0 - sin_theta_max_sq).max(0.0).sqrt();

        // 立体角測度でのPDF
        let solid_angle = 2.0 * std::f64::consts::PI * (1.0 - cos_theta_max);
        let pdf_omega = 1.0 / solid_angle;

        // target_hitがある場合はその点を使う
        let sampled_point = if let Some(target) = target_hit {
            target.point
        } else {
            // ランダムサンプリング
            let r1: f64 = rng.random();
            let r2: f64 = rng.random();

            let cos_theta = 1.0 - r1 + r1 * cos_theta_max;
            let sin_theta: f64 = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let phi = 2.0 * std::f64::consts::PI * r2;

            let w = to_center.normalize();
            let up = if w.y.abs() > 0.999 {
                Vector3::new(1.0, 0.0, 0.0)
            } else {
                Vector3::new(0.0, 1.0, 0.0)
            };
            let u = up.cross(&w).normalize();
            let v = w.cross(&u);

            let direction = u * (sin_theta * phi.cos()) + v * (sin_theta * phi.sin()) + w * cos_theta;
            let sample_ray = Ray::new(hit_from.point, direction);

            let oc = sample_ray.origin - self.center;
            let a = direction.dot(&direction);
            let half_b = oc.dot(&direction);
            let c = oc.dot(&oc) - self.radius * self.radius;
            let discriminant = half_b * half_b - a * c;

            let t = (-half_b - discriminant.sqrt()) / a;
            sample_ray.at(t)
        };

        let normal = (sampled_point - self.center).normalize();
        let light_dir = sampled_point - hit_from.point;
        let d = light_dir.length();

        (sampled_point, normal, pdf_omega, light_dir.normalize(), d)
    }
}

pub struct TriangleShape {
    pub v0: Vector3,
    pub v1: Vector3,
    pub v2: Vector3,
}

impl TriangleShape {
    pub fn new(v0: Vector3, v1: Vector3, v2: Vector3) -> Self {
        Self { v0, v1, v2 }
    }
}

impl Shape for TriangleShape {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        // Möller–Trumbore intersection algorithm
        let edge1 = self.v1 - self.v0;
        let edge2 = self.v2 - self.v0;
        let h = ray.direction.cross(&edge2);
        let a = edge1.dot(&h);

        if a.abs() < 1e-8 {
            return None; // Ray is parallel to triangle
        }

        let f = 1.0 / a;
        let s = ray.origin - self.v0;
        let u = f * s.dot(&h);

        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let q = s.cross(&edge1);
        let v = f * ray.direction.dot(&q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        // Compute t to find out where the intersection point is on the line
        let t = f * edge2.dot(&q);

        if t < t_min || t > t_max {
            return None;
        }

        let point = ray.at(t);
        let outward_normal = edge1.cross(&edge2).normalize();

        Some(HitRecord::new(point, outward_normal, t, ray))
    }

    fn sample_surface_from_point(
        &self,
        hit_from: &HitRecord,
        target_hit: Option<&HitRecord>,
        rng: &mut dyn RngCore,
    ) -> (Vector3, Vector3, f64, Vector3, f64) {
        // target_hitがある場合はその点を使い、ない場合はランダムサンプリング
        let sampled_point = if let Some(target) = target_hit {
            target.point
        } else {
            // 三角形の面積一様サンプリング
            let r1: f64 = rng.random();
            let r2: f64 = rng.random();

            // 三角形上の一様分布
            let sqrt_r1 = r1.sqrt();
            let u = 1.0 - sqrt_r1;
            let v = r2 * sqrt_r1;

            self.v0 + (self.v1 - self.v0) * u + (self.v2 - self.v0) * v
        };

        // 法線と面積
        let edge1 = self.v1 - self.v0;
        let edge2 = self.v2 - self.v0;
        let normal = edge1.cross(&edge2).normalize();
        let area = edge1.cross(&edge2).length() * 0.5;

        let to_light = sampled_point - hit_from.point;
        let d = to_light.length();
        let light_dir = to_light / d;
        let cos_light = normal.dot(&(-light_dir)).abs();  // 両面発光に対応

        // 面積測度PDFから立体角測度PDFへ変換
        let pdf_area = 1.0 / area;
        let pdf_omega = if cos_light > 1e-8 {
            pdf_area * (d * d) / cos_light
        } else {
            1e-8  // 完全に平行な場合の安全値
        };

        (sampled_point, normal, pdf_omega, light_dir, d)
    }
}
