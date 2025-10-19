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

    /// 指定した点から見える形状の表面をサンプリング
    ///
    /// # Arguments
    /// * `point` - 観測点（衝突点）
    /// * `rng` - 乱数生成器
    ///
    /// # Returns
    /// (サンプリングされた点, 法線, PDF)
    /// PDFは立体角測度または面積測度（実装に依存）
    fn sample_surface_from_point(
        &self,
        point: &Vector3,
        rng: &mut dyn RngCore,
    ) -> (Vector3, Vector3, f64);
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
        point: &Vector3,
        rng: &mut dyn RngCore,
    ) -> (Vector3, Vector3, f64) {
        let to_center = self.center - *point;
        let distance_sq = to_center.dot(&to_center);
        let distance = distance_sq.sqrt();

        // 球が点の内部または非常に近い場合は面積一様サンプリング
        if distance <= self.radius * 1.001 {
            return self.sample_surface_uniform(rng);
        }

        // 立体角サンプリング
        // cos(theta_max) = sqrt(distance^2 - radius^2) / distance
        let sin_theta_max_sq = (self.radius * self.radius) / distance_sq;
        let cos_theta_max = (1.0 - sin_theta_max_sq).max(0.0).sqrt();

        // コーン内で一様サンプリング
        let r1: f64 = rng.random();
        let r2: f64 = rng.random();

        let cos_theta = 1.0 - r1 + r1 * cos_theta_max;
        let sin_theta: f64 = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = 2.0 * std::f64::consts::PI * r2;

        // ローカル座標系を構築（to_centerをz軸とする）
        let w = to_center.normalize();
        let up = if w.y.abs() > 0.999 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let u = up.cross(&w).normalize();
        let v = w.cross(&u);

        // サンプリング方向
        let direction = u * (sin_theta * phi.cos())
            + v * (sin_theta * phi.sin())
            + w * cos_theta;

        // レイを飛ばして球との交差点を求める
        let sample_ray = Ray::new(*point, direction);

        // 球との交差計算
        let oc = sample_ray.origin - self.center;
        let a = direction.dot(&direction);
        let half_b = oc.dot(&direction);
        let c = oc.dot(&oc) - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            // 交差しない場合（理論的には起きないはず）
            return self.sample_surface_uniform(rng);
        }

        let t = (-half_b + discriminant.sqrt()) / a;
        let sampled_point = sample_ray.at(t);
        let normal = (sampled_point - self.center).normalize();

        // 立体角測度でのPDF = 1 / solid_angle
        let solid_angle = 2.0 * std::f64::consts::PI * (1.0 - cos_theta_max);
        let pdf = 1.0 / solid_angle;

        (sampled_point, normal, pdf)
    }
}

impl SphereShape {
    /// 球の表面を一様サンプリング（フォールバック用）
    fn sample_surface_uniform(&self, rng: &mut dyn RngCore) -> (Vector3, Vector3, f64) {
        let r1: f64 = rng.random();
        let r2: f64 = rng.random();

        let phi = 2.0 * std::f64::consts::PI * r1;
        let cos_theta = 1.0 - 2.0 * r2;
        let sin_theta: f64 = (1.0 - cos_theta * cos_theta).sqrt();

        let x = sin_theta * phi.cos();
        let y = sin_theta * phi.sin();
        let z = cos_theta;

        let local_point = Vector3::new(x, y, z);
        let sampled_point = self.center + local_point * self.radius;
        let normal = local_point;

        let area = 4.0 * std::f64::consts::PI * self.radius * self.radius;
        let pdf = 1.0 / area;

        (sampled_point, normal, pdf)
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
        _point: &Vector3,
        rng: &mut dyn RngCore,
    ) -> (Vector3, Vector3, f64) {
        // 三角形の面積一様サンプリング
        let r1: f64 = rng.random();
        let r2: f64 = rng.random();

        // 三角形上の一様分布
        let sqrt_r1 = r1.sqrt();
        let u = 1.0 - sqrt_r1;
        let v = r2 * sqrt_r1;

        // サンプリングされた点
        let sampled_point = self.v0 + (self.v1 - self.v0) * u + (self.v2 - self.v0) * v;

        // 法線と面積
        let edge1 = self.v1 - self.v0;
        let edge2 = self.v2 - self.v0;
        let normal = edge1.cross(&edge2).normalize();
        let area = edge1.cross(&edge2).length() * 0.5;

        // 面積測度でのPDF
        let pdf = 1.0 / area;

        (sampled_point, normal, pdf)
    }
}
