use crate::math::{Vector3, tan2};
use rand::Rng;

/// マテリアルを表すtrait（BRDF、サンプリング、PDFを担当）
pub trait Material: Send + Sync {
    /// BRDFとPDFを同時に計算（効率的）
    ///
    /// # Arguments
    /// * `x` - 交差点の位置
    /// * `i` - 入射方向（カメラ方向）
    /// * `o` - 出射方向（サンプリングされた方向）
    /// * `normal` - 法線ベクトル
    ///
    /// # Returns
    /// (brdf, pdf) のタプル
    fn brdf_pdf(&self, x: &Vector3, i: &Vector3, o: &Vector3, normal: &Vector3) -> (Vector3, f64);

    /// BRDF値を取得（デフォルト実装はbrdf_pdfを使用）
    fn brdf(&self, x: &Vector3, i: &Vector3, o: &Vector3, normal: &Vector3) -> Vector3 {
        self.brdf_pdf(x, i, o, normal).0
    }

    /// サンプリング方向を生成
    ///
    /// # Arguments
    /// * `normal` - 法線ベクトル
    /// * `incoming` - 入射方向（カメラから来る方向）
    /// * `rng` - 乱数生成器
    fn sample_direction<R: Rng>(
        &self,
        normal: &Vector3,
        incoming: &Vector3,
        rng: &mut R,
    ) -> Vector3;
}

/// マテリアルの種類を表すenum
pub enum MaterialType {
    #[allow(dead_code)]
    Lambertian(Lambertian),
    LambertianCosineWeighted(LambertianCosineWeighted),
    Mirror(Mirror),
}

impl MaterialType {
    /// BRDFとPDFを同時に取得（効率的）
    pub fn brdf_pdf(
        &self,
        x: &Vector3,
        i: &Vector3,
        o: &Vector3,
        normal: &Vector3,
    ) -> (Vector3, f64) {
        match self {
            MaterialType::Lambertian(m) => m.brdf_pdf(x, i, o, normal),
            MaterialType::LambertianCosineWeighted(m) => m.brdf_pdf(x, i, o, normal),
            MaterialType::Mirror(m) => m.brdf_pdf(x, i, o, normal),
        }
    }

    /// サンプリング方向を生成
    pub fn sample_direction<R: Rng>(
        &self,
        normal: &Vector3,
        incoming: &Vector3,
        rng: &mut R,
    ) -> Vector3 {
        match self {
            MaterialType::Lambertian(m) => m.sample_direction(normal, incoming, rng),
            MaterialType::LambertianCosineWeighted(m) => m.sample_direction(normal, incoming, rng),
            MaterialType::Mirror(m) => m.sample_direction(normal, incoming, rng),
        }
    }
}

/// Lambertian（拡散反射）マテリアル
pub struct Lambertian {
    /// アルベド（反射率）
    pub albedo: Vector3,
}

#[allow(dead_code)]
impl Lambertian {
    /// 新しいLambertianマテリアルを作成
    pub fn new(albedo: Vector3) -> Self {
        Self { albedo }
    }
}

impl Lambertian {
    /// PDF値を計算（内部用）
    fn pdf(&self, _normal: &Vector3, _direction: &Vector3) -> f64 {
        // 一様半球分布のPDF = 1 / (2π)
        1.0 / (2.0 * std::f64::consts::PI)
    }
}

impl Material for Lambertian {
    fn brdf_pdf(
        &self,
        _x: &Vector3,
        _i: &Vector3,
        o: &Vector3,
        normal: &Vector3,
    ) -> (Vector3, f64) {
        let brdf = self.albedo / std::f64::consts::PI;
        let pdf = self.pdf(normal, o);
        (brdf, pdf)
    }

    fn sample_direction<R: Rng>(
        &self,
        normal: &Vector3,
        _incoming: &Vector3,
        rng: &mut R,
    ) -> Vector3 {
        // 一様半球サンプリング（入射方向は使わない）
        Vector3::random_hemisphere_direction(normal, rng)
    }
}

/// コサイン重み付きサンプリングを使用するLambertianマテリアル
pub struct LambertianCosineWeighted {
    /// アルベド（反射率）
    pub albedo: Vector3,
}

impl LambertianCosineWeighted {
    /// 新しいLambertianCosineWeightedマテリアルを作成
    pub fn new(albedo: Vector3) -> Self {
        Self { albedo }
    }
}

impl LambertianCosineWeighted {
    /// PDF値を計算（内部用）
    fn pdf(&self, normal: &Vector3, direction: &Vector3) -> f64 {
        // コサイン重み付き分布のPDF = cos(θ) / π
        let cos_theta = direction.dot(normal).max(0.0);
        cos_theta / std::f64::consts::PI
    }
}

impl Material for LambertianCosineWeighted {
    fn brdf_pdf(
        &self,
        _x: &Vector3,
        _i: &Vector3,
        o: &Vector3,
        normal: &Vector3,
    ) -> (Vector3, f64) {
        let brdf = self.albedo / std::f64::consts::PI;
        let pdf = self.pdf(normal, o);
        (brdf, pdf)
    }

    fn sample_direction<R: Rng>(
        &self,
        normal: &Vector3,
        _incoming: &Vector3,
        rng: &mut R,
    ) -> Vector3 {
        // コサイン重み付きサンプリング
        let r1: f64 = rng.random();
        let r2: f64 = rng.random();

        let phi = 2.0 * std::f64::consts::PI * r1;
        let cos_theta = r2.sqrt(); // コサイン重み付き
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let x = sin_theta * phi.cos();
        let y = sin_theta * phi.sin();
        let z = cos_theta;

        // ローカル座標系をワールド座標系に変換
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

/// 完全鏡面反射マテリアル
pub struct Mirror {
    /// 粗さ
    pub roughness: f64,
    /// 色
    pub color: Vector3,
}

impl Material for Mirror {
    fn brdf_pdf(&self, _x: &Vector3, i: &Vector3, o: &Vector3, normal: &Vector3) -> (Vector3, f64) {
        //D(GGX)
        let alpha = self.roughness * self.roughness;
        let alpha2 = alpha * alpha;
        let h = (*i + *o).normalize();
        let denom = (normal.dot(&h) * normal.dot(&h)) * (alpha2 - 1.0) + 1.0;
        let d = alpha2 / (std::f64::consts::PI * denom * denom);

        //G
        let lambda_a =
            |v: &Vector3| -> f64 { (-1.0 + (1.0 + alpha2 * tan2(*v, *normal)).sqrt()) / 2.0 };
        let g = 1.0 / (1.0 + lambda_a(i) + lambda_a(o));

        //F
        let cos_theta = i.dot(&h).max(0.0);
        let f = self.color + (Vector3::new(1.0, 1.0, 1.0) - self.color) * (1.0 - cos_theta).powi(5);

        let brdf = d * g * f / (4.0 * normal.dot(i).abs() * normal.dot(o).abs());
        let pdf = d * normal.dot(&h).abs() / (4.0 * i.dot(&h).abs());

        (brdf, pdf)
    }

    fn sample_direction<R: Rng>(
        &self,
        normal: &Vector3,
        incoming: &Vector3,
        rng: &mut R,
    ) -> Vector3 {
        let r1: f64 = rng.random();
        let alpha = self.roughness * self.roughness;
        let alpha2 = alpha * alpha;
        let phi: f64 = 2.0 * std::f64::consts::PI * r1;
        let r2: f64 = rng.random();
        let sin_theta = (alpha2 * r2 / (1.0 - r2 + alpha2 * r2)).sqrt();
        let cos_theta = (1.0 - sin_theta * sin_theta).sqrt();
        let x = sin_theta * phi.cos();
        let y = sin_theta * phi.sin();
        let z = cos_theta;
        // ローカル座標系をワールド座標系に変換
        let up = if normal.y.abs() > 0.999 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let tangent = up.cross(normal).normalize();
        let bitangent = normal.cross(&tangent);
        let half_vector = (tangent * x + bitangent * y + *normal * z).normalize();

        // incomingは表面に向かう方向なので、表面から外向きに変換
        let i = -*incoming;
        2.0 * i.dot(&half_vector) * half_vector - i
    }
}
