use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn one() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    // 内積 (dot product)
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    // 外積 (cross product)
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    // ベクトルの長さ
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    // ベクトルの長さの2乗
    pub fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    // 正規化
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len > 0.0 { *self / len } else { *self }
    }

    // 正規化（インプレース版）
    pub fn normalized(self) -> Self {
        self.normalize()
    }

    // 三角形の面から法線ベクトルを計算
    // v0, v1, v2: 三角形の3つの頂点
    pub fn normal_from_triangle(v0: &Self, v1: &Self, v2: &Self) -> Self {
        let edge1 = *v1 - *v0;
        let edge2 = *v2 - *v0;
        edge1.cross(&edge2).normalize()
    }

    // 入射ベクトルと法線から反射ベクトルを計算
    // incident: 入射ベクトル（表面に向かうベクトル）
    // normal: 法線ベクトル（正規化されている必要がある）
    pub fn reflect(&self, normal: &Self) -> Self {
        *self - *normal * 2.0 * self.dot(normal)
    }

    // 入射ベクトルと法線から屈折ベクトルを計算（スネルの法則）
    // incident: 入射ベクトル（正規化されている必要がある）
    // normal: 法線ベクトル（正規化されている必要がある）
    // eta: 屈折率の比（n1/n2）
    pub fn refract(&self, normal: &Self, eta: f64) -> Option<Self> {
        let cos_i = -self.dot(normal);
        let sin2_t = eta * eta * (1.0 - cos_i * cos_i);

        if sin2_t > 1.0 {
            // 全反射
            None
        } else {
            let cos_t = (1.0 - sin2_t).sqrt();
            Some(*self * eta + *normal * (eta * cos_i - cos_t))
        }
    }

    // 法線ベクトルの向きを調整（レイの方向と反対側を向くように）
    // ray_direction: レイの方向ベクトル
    pub fn face_forward(&self, ray_direction: &Self) -> Self {
        if self.dot(ray_direction) < 0.0 {
            *self
        } else {
            -*self
        }
    }

    // 法線を基準にした半球上の一様分布からランダムな方向を生成
    pub fn random_hemisphere_direction(normal: &Self, rng: &mut impl rand::Rng) -> Self {
        // 半球上の一様分布サンプリング
        let r1: f64 = rng.random();
        let r2: f64 = rng.random();

        let phi = 2.0 * std::f64::consts::PI * r1;
        let cos_theta = r2; // 一様分布
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let x = sin_theta * phi.cos();
        let y = sin_theta * phi.sin();
        let z = cos_theta;

        // ローカル座標系をワールド座標系に変換
        // 法線をZ軸とする座標系を構築
        let up = if normal.y.abs() > 0.999 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };

        let tangent = up.cross(normal).normalize();
        let bitangent = normal.cross(&tangent);

        (tangent * x + bitangent * y + *normal * z).normalize()
    }
}

// Vector3 + Vector3
impl Add for Vector3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl AddAssign for Vector3 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

// Vector3 - Vector3
impl Sub for Vector3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

// Vector3 * スカラー
impl Mul<f64> for Vector3 {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

// スカラー * Vector3
impl Mul<Vector3> for f64 {
    type Output = Vector3;

    fn mul(self, vector: Vector3) -> Vector3 {
        vector * self
    }
}

// Vector3 * Vector3 (成分ごとの積)
impl Mul<Vector3> for Vector3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

// Vector3 / スカラー
impl Div<f64> for Vector3 {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

// Vector3 / Vector3 (成分ごとの除算)
impl Div<Vector3> for Vector3 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        }
    }
}

// -Vector3
impl Neg for Vector3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

pub fn tan2(a: Vector3, b: Vector3) -> f64 {
    let cos_theta = a.dot(&b);
    let sin2_theta = 1.0 - cos_theta * cos_theta;
    if sin2_theta <= 0.0 {
        0.0
    } else {
        sin2_theta / (cos_theta * cos_theta)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector3_creation() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vector3_add() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(4.0, 5.0, 6.0);
        let result = v1 + v2;
        assert_eq!(result, Vector3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vector3_sub() {
        let v1 = Vector3::new(4.0, 5.0, 6.0);
        let v2 = Vector3::new(1.0, 2.0, 3.0);
        let result = v1 - v2;
        assert_eq!(result, Vector3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_vector3_mul_scalar() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let result = v * 2.0;
        assert_eq!(result, Vector3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_scalar_mul_vector3() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let result = 2.0 * v;
        assert_eq!(result, Vector3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_vector3_mul_vector3() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(2.0, 3.0, 4.0);
        let result = v1 * v2;
        assert_eq!(result, Vector3::new(2.0, 6.0, 12.0));
    }

    #[test]
    fn test_vector3_div_scalar() {
        let v = Vector3::new(2.0, 4.0, 6.0);
        let result = v / 2.0;
        assert_eq!(result, Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_vector3_neg() {
        let v = Vector3::new(1.0, -2.0, 3.0);
        let result = -v;
        assert_eq!(result, Vector3::new(-1.0, 2.0, -3.0));
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(4.0, 5.0, 6.0);
        let result = v1.dot(&v2);
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_cross_product() {
        let v1 = Vector3::new(1.0, 0.0, 0.0);
        let v2 = Vector3::new(0.0, 1.0, 0.0);
        let result = v1.cross(&v2);
        assert_eq!(result, Vector3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_length() {
        let v = Vector3::new(3.0, 4.0, 0.0);
        assert_eq!(v.length(), 5.0);
    }

    #[test]
    fn test_normalize() {
        let v = Vector3::new(3.0, 4.0, 0.0);
        let normalized = v.normalize();
        assert!((normalized.length() - 1.0).abs() < 1e-10);
        assert_eq!(normalized, Vector3::new(0.6, 0.8, 0.0));
    }

    #[test]
    fn test_normalized() {
        let v = Vector3::new(3.0, 4.0, 0.0);
        let normalized = v.normalized();
        assert!((normalized.length() - 1.0).abs() < 1e-10);
        assert_eq!(normalized, Vector3::new(0.6, 0.8, 0.0));
    }

    #[test]
    fn test_normal_from_triangle() {
        // 反時計回りの三角形 (右手座標系でZ+方向を向く)
        let v0 = Vector3::new(0.0, 0.0, 0.0);
        let v1 = Vector3::new(1.0, 0.0, 0.0);
        let v2 = Vector3::new(0.0, 1.0, 0.0);
        let normal = Vector3::normal_from_triangle(&v0, &v1, &v2);

        // 法線はZ+方向を向く
        assert!((normal.x).abs() < 1e-10);
        assert!((normal.y).abs() < 1e-10);
        assert!((normal.z - 1.0).abs() < 1e-10);
        assert!((normal.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reflect() {
        // 45度で入射する光線
        let incident = Vector3::new(1.0, -1.0, 0.0).normalize();
        let normal = Vector3::new(0.0, 1.0, 0.0);
        let reflected = incident.reflect(&normal);

        // 反射光は45度で反射する
        let expected = Vector3::new(1.0, 1.0, 0.0).normalize();
        assert!((reflected.x - expected.x).abs() < 1e-10);
        assert!((reflected.y - expected.y).abs() < 1e-10);
        assert!((reflected.z - expected.z).abs() < 1e-10);
    }

    #[test]
    fn test_refract() {
        // 垂直入射（屈折なし）
        let incident = Vector3::new(0.0, -1.0, 0.0);
        let normal = Vector3::new(0.0, 1.0, 0.0);
        let eta = 1.0 / 1.5; // 空気からガラスへ

        let refracted = incident.refract(&normal, eta);
        assert!(refracted.is_some());

        let refracted = refracted.unwrap();
        assert!((refracted.x).abs() < 1e-10);
        assert!(refracted.y < 0.0); // 下向き
    }

    #[test]
    fn test_refract_total_internal_reflection() {
        // 全反射が起こる角度
        let incident = Vector3::new(0.8, -0.6, 0.0).normalize();
        let normal = Vector3::new(0.0, 1.0, 0.0);
        let eta = 1.5 / 1.0; // ガラスから空気へ（大きな角度で入射）

        let refracted = incident.refract(&normal, eta);
        // 全反射のためNoneが返る
        assert!(refracted.is_none());
    }

    #[test]
    fn test_face_forward() {
        let normal = Vector3::new(0.0, 1.0, 0.0);

        // レイが法線と反対方向から来る場合（通常）
        let ray_from_outside = Vector3::new(0.0, -1.0, 0.0);
        let faced = normal.face_forward(&ray_from_outside);
        assert_eq!(faced, normal);

        // レイが法線と同じ方向から来る場合（裏面）
        let ray_from_inside = Vector3::new(0.0, 1.0, 0.0);
        let faced = normal.face_forward(&ray_from_inside);
        assert_eq!(faced, -normal);
    }
}
