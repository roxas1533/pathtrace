use crate::{
    camera::Ray,
    math::{Vector3, tan2},
};
use rand::{Rng, RngCore};

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
    fn brdf_pdf(
        &self,
        x: &Vector3,
        i: &Vector3,
        o: &Vector3,
        normal: &Vector3,
    ) -> (Vector3, f64);
    fn btdf_pdf(
        &self,
        x: &Vector3,
        i: &Vector3,
        o: &Vector3,
        normal: &Vector3,
    ) -> (Vector3, f64) {
        self.brdf_pdf(x, i, o, normal)
    }

    /// サンプリング方向を生成
    ///
    /// # Arguments
    /// * `normal` - 法線ベクトル
    /// * `incoming` - 入射方向（カメラから来る方向）
    /// * `rng` - 乱数生成器
    fn sample_direction(&self, normal: &Vector3, incoming: &Ray, rng: &mut dyn RngCore) -> Vector3;

    fn sample_direction_btdf(
        &self,
        normal: &Vector3,
        incoming: &Ray,
        rng: &mut dyn RngCore,
    ) -> Vector3 {
        self.sample_direction(normal, incoming, rng)
    }

    fn get_eta(&self) -> f64 {
        1.0
    }

    /// 発光量を返す（デフォルトは発光しない）
    ///
    /// # Arguments
    /// * `_x` - 交差点の位置
    /// * `_normal` - 法線ベクトル
    ///
    /// # Returns
    /// 発光色と強度
    fn emit(&self, _x: &Vector3, _normal: &Vector3) -> Vector3 {
        Vector3::zero()
    }
}

pub struct LambertianCosineWeighted {
    pub albedo: Vector3,
}

impl LambertianCosineWeighted {
    /// 新しいLambertianCosineWeightedマテリアルを作成
    pub fn new(albedo: Vector3) -> Self {
        Self { albedo }
    }

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

    fn sample_direction(
        &self,
        normal: &Vector3,
        _incoming: &Ray,
        rng: &mut dyn RngCore,
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

/// 発光マテリアル（光源用）
pub struct Emissive {
    /// 発光色と強度
    pub emission: Vector3,
}

impl Emissive {
    /// 新しいEmissiveマテリアルを作成
    pub fn new(emission: Vector3) -> Self {
        Self { emission }
    }
}

impl Material for Emissive {
    fn brdf_pdf(
        &self,
        _x: &Vector3,
        _i: &Vector3,
        _o: &Vector3,
        _normal: &Vector3,
    ) -> (Vector3, f64) {
        // 発光マテリアルは光を反射しない（黒体）
        (Vector3::zero(), 1.0)
    }

    fn sample_direction(
        &self,
        normal: &Vector3,
        _incoming: &Ray,
        _rng: &mut dyn RngCore,
    ) -> Vector3 {
        // ダミー
        *normal
    }

    fn emit(&self, _x: &Vector3, _normal: &Vector3) -> Vector3 {
        self.emission
    }
}

/// Oren-Nayar拡散反射マテリアル（表面の粗さを考慮）
pub struct OrenNayar {
    /// アルベド（反射率）
    pub albedo: Vector3,
    /// 粗さ（ラジアン単位）
    pub roughness: f64,
    // 事前計算された係数
    a: f64,
    b: f64,
}

impl OrenNayar {
    /// 新しいOren-Nayarマテリアルを作成
    ///
    /// # Arguments
    /// * `albedo` - 反射率
    /// * `roughness` - 表面の粗さ（0.0 = 滑らか/Lambertian, 1.0 = 非常に粗い）
    pub fn new(albedo: Vector3, roughness: f64) -> Self {
        let sigma = roughness;
        let sigma2 = sigma * sigma;
        let a = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
        let b = 0.45 * sigma2 / (sigma2 + 0.09);
        Self {
            albedo,
            roughness,
            a,
            b,
        }
    }

    /// PDF値を計算（Lambertianと同じコサイン重み付き）
    fn pdf(&self, normal: &Vector3, direction: &Vector3) -> f64 {
        let cos_theta = direction.dot(normal).max(0.0);
        cos_theta / std::f64::consts::PI
    }

    /// 接線空間での方位角を計算
    fn compute_azimuth(&self, direction: &Vector3, normal: &Vector3, tangent: &Vector3) -> f64 {
        let bitangent = normal.cross(tangent);
        let x = direction.dot(tangent);
        let y = direction.dot(&bitangent);
        y.atan2(x)
    }

    /// 接線ベクトルを計算
    fn compute_tangent(&self, normal: &Vector3) -> Vector3 {
        let up = if normal.y.abs() > 0.999 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        up.cross(normal).normalize()
    }
}

impl Material for OrenNayar {
    fn brdf_pdf(
        &self,
        _x: &Vector3,
        i: &Vector3,
        o: &Vector3,
        normal: &Vector3,
    ) -> (Vector3, f64) {
        // 入射角と出射角（法線からの角度）
        let cos_theta_i = i.dot(normal).max(0.0);
        let cos_theta_o = o.dot(normal).max(0.0);

        // 角度を計算（acosは不安定なのでatan2を使用）
        let sin_theta_i = (1.0 - cos_theta_i * cos_theta_i).max(0.0).sqrt();
        let sin_theta_o = (1.0 - cos_theta_o * cos_theta_o).max(0.0).sqrt();

        // 方位角の差を計算
        let tangent = self.compute_tangent(normal);
        let phi_i = self.compute_azimuth(i, normal, &tangent);
        let phi_o = self.compute_azimuth(o, normal, &tangent);
        let cos_phi_diff = (phi_i - phi_o).cos().max(0.0);

        // αとβの計算
        let (sin_alpha, tan_beta) = if cos_theta_i > cos_theta_o {
            // theta_i < theta_o なので alpha = theta_o, beta = theta_i
            let tan_beta = if cos_theta_i > 1e-6 {
                sin_theta_i / cos_theta_i
            } else {
                0.0
            };
            (sin_theta_o, tan_beta)
        } else {
            // theta_i >= theta_o なので alpha = theta_i, beta = theta_o
            let tan_beta = if cos_theta_o > 1e-6 {
                sin_theta_o / cos_theta_o
            } else {
                0.0
            };
            (sin_theta_i, tan_beta)
        };

        // Oren-Nayar BRDF
        let oren_nayar_term = self.a + self.b * cos_phi_diff * sin_alpha * tan_beta;
        let brdf = self.albedo * (oren_nayar_term / std::f64::consts::PI);

        let pdf = self.pdf(normal, o);

        (brdf, pdf)
    }

    fn sample_direction(
        &self,
        normal: &Vector3,
        _incoming: &Ray,
        rng: &mut dyn RngCore,
    ) -> Vector3 {
        // コサイン重み付きサンプリング（Lambertianと同じ）
        let r1: f64 = rng.random();
        let r2: f64 = rng.random();

        let phi = 2.0 * std::f64::consts::PI * r1;
        let cos_theta = r2.sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let x = sin_theta * phi.cos();
        let y = sin_theta * phi.sin();
        let z = cos_theta;

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

/// 鏡面反射マテリアル（金属と誘電体の両対応）
pub struct Mirror {
    /// 粗さ
    pub roughness: f64,
    /// 色
    pub color: Vector3,
    /// 金属度合（0=誘電体、1=金属）
    pub metallic: f64,
    /// 屈折率（IOR）metallic < 1.0の場合に使用
    pub ior: f64,
}

impl Mirror {
    fn get_half_vector(&self, normal: &Vector3, rng: &mut dyn RngCore) -> Vector3 {
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
        (tangent * x + bitangent * y + *normal * z).normalize()
    }
}

impl Material for Mirror {
    fn brdf_pdf(
        &self,
        _x: &Vector3,
        i: &Vector3,
        o: &Vector3,
        normal: &Vector3,
    ) -> (Vector3, f64) {
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
        // 誘電体のF0をIORから計算: F0 = ((1-ior)/(1+ior))^2
        let f0_dielectric = ((1.0 - self.ior) / (1.0 + self.ior)).powi(2);
        let f0_dielectric_vec = Vector3::new(f0_dielectric, f0_dielectric, f0_dielectric);
        // 金属のF0は色（波長依存の反射率）
        let f0 = f0_dielectric_vec * (1.0 - self.metallic) + self.color * self.metallic;
        // Schlick近似でFresnelを計算
        let f = f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * (1.0 - cos_theta).powi(5);

        let brdf = d * g * f / (4.0 * normal.dot(i).abs() * normal.dot(o).abs());
        let pdf = d * normal.dot(&h).abs() / (4.0 * i.dot(&h).abs());

        (brdf, pdf)
    }

    fn btdf_pdf(
        &self,
        _x: &Vector3,
        i: &Vector3,
        o: &Vector3,
        normal: &Vector3,
    ) -> (Vector3, f64) {
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
        // 誘電体のF0をIORから計算: F0 = ((1-ior)/(1+ior))^2
        // let f0_dielectric = ((1.0 - self.ior) / (1.0 + self.ior)).powi(2);
        let f0_dielectric = 0.16 * self.ior * self.ior;
        let f0_dielectric_vec = Vector3::new(f0_dielectric, f0_dielectric, f0_dielectric);
        // 金属のF0は色（波長依存の反射率）
        let f0 = f0_dielectric_vec * (1.0 - self.metallic) + self.color * self.metallic;
        // Schlick近似でFresnelを計算
        let f = f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * (1.0 - cos_theta).powi(5);

        let i_h = i.dot(&h).abs();
        let o_h = o.dot(&h).abs();
        let i_n = normal.dot(i).abs();
        let o_n = normal.dot(o).abs();
        const ETA: f64 = 1.52;
        let alpha = i_h * o_h / (i_n * o_n);

        let btdf = alpha * (Vector3::one() - f) * g * d / (i.dot(&h) + ETA * o.dot(&h)).powi(2);
        // let jacobian_denom = eta*

        let pdf = d * normal.dot(&h).abs() / (4.0 * i.dot(&h).abs());


        (btdf, pdf)
    }

    fn sample_direction(&self, normal: &Vector3, incoming: &Ray, rng: &mut dyn RngCore) -> Vector3 {
        // incomingは表面に向かう方向なので、表面から外向きに変換
        let half_vector = self.get_half_vector(normal, rng);
        let i = -incoming.direction;
        2.0 * i.dot(&half_vector) * half_vector - i
    }

    fn sample_direction_btdf(
        &self,
        normal: &Vector3,
        incoming: &Ray,
        rng: &mut dyn RngCore,
    ) -> Vector3 {
        // incomingは表面に向かう方向なので、表面から外向きに変換
        let half_vector = self.get_half_vector(normal, rng);
        let i = -incoming.direction;
        let eta = incoming.eta_ratio;
        let cos_theta_i = normal.dot(&i);
        let cos_theta_t2 = 1.0 - (eta * eta) * (1.0 - cos_theta_i * cos_theta_i);
        if cos_theta_t2 < 0.0 {
            // 全反射
            return Vector3::zero();
        }
        let cos_theta_t = cos_theta_t2.sqrt();
        eta * -i + (eta * cos_theta_i - cos_theta_t) * half_vector
    }

    fn get_eta(&self) -> f64 {
        self.ior
    }
}

pub struct PBRMaterial {
    specular: Mirror,
    diffuse: OrenNayar,
    pub metallic: f64,
}

impl PBRMaterial {
    pub fn new(roughness: f64, base_color: Vector3, metallic: f64) -> Self {
        Self::new_with_ior(roughness, base_color, metallic, 0.5)
    }

    pub fn new_with_ior(roughness: f64, base_color: Vector3, metallic: f64, ior: f64) -> Self {
        Self {
            specular: Mirror {
                roughness,
                color: base_color,
                metallic,
                ior,
            },
            // Oren-Nayarを使用して表面の粗さを表現
            // roughnessが大きいほど、拡散反射に粗さの効果が現れる
            diffuse: OrenNayar::new(base_color, 0.8),
            metallic,
        }
    }

    /// Fresnel項（F項）- Schlick近似
    fn fresnel(&self, f0: &Vector3, cos_theta: f64) -> Vector3 {
        *f0 + (Vector3::new(1.0, 1.0, 1.0) - *f0) * (1.0 - cos_theta).powi(5)
    }
}

impl Material for PBRMaterial {
    fn brdf_pdf(
        &self,
        x: &Vector3,
        i: &Vector3,
        o: &Vector3,
        normal: &Vector3,
    ) -> (Vector3, f64) {
        // 鏡面反射成分を取得（Mirrorに委譲）
        let (specular_brdf, specular_pdf) = self.specular.brdf_pdf(x, i, o, normal);

        // 拡散反射成分を取得（Oren-Nayarに委譲）
        let (diffuse_brdf_raw, diffuse_pdf) = self.diffuse.brdf_pdf(x, i, o, normal);

        // Fresnel係数を計算
        let h = (*i + *o).normalize();
        let cos_theta = i.dot(&h).max(0.0);

        // F0を計算
        let f0_dielectric = ((1.0 - self.specular.ior) / (1.0 + self.specular.ior)).powi(2);
        let f0_dielectric_vec = Vector3::new(f0_dielectric, f0_dielectric, f0_dielectric);
        let f0 = f0_dielectric_vec * (1.0 - self.metallic) + self.specular.color * self.metallic;

        let f = self.fresnel(&f0, cos_theta);

        // 拡散反射BRDFにエネルギー保存則を適用
        // 金属は拡散反射しない。Fresnelで反射されなかった光が拡散反射される
        let diffuse_brdf = if self.metallic < 1.0 {
            let kd = (Vector3::new(1.0, 1.0, 1.0) - f) * (1.0 - self.metallic);
            diffuse_brdf_raw * kd
        } else {
            Vector3::zero()
        };

        // 合成BRDF
        let brdf = specular_brdf + diffuse_brdf;

        // 合成PDF（Fresnel重みで鏡面反射と拡散反射をブレンド）
        let f_avg = (f.x + f.y + f.z) / 3.0;
        let specular_weight = f_avg;
        let diffuse_weight = (1.0 - f_avg) * (1.0 - self.metallic);
        let total_weight = specular_weight + diffuse_weight;

        let pdf = if total_weight > 1e-6 {
            (specular_weight * specular_pdf + diffuse_weight * diffuse_pdf) / total_weight
        } else {
            specular_pdf
        };

        (brdf, pdf)
    }

    fn sample_direction(&self, normal: &Vector3, incoming: &Ray, rng: &mut dyn RngCore) -> Vector3 {
        // 入射方向からFresnelを概算
        let i = -incoming.direction;
        let cos_theta_i = i.dot(normal).max(0.0);
        let f0_scalar = if self.metallic > 0.5 {
            (self.specular.color.x + self.specular.color.y + self.specular.color.z) / 3.0
        } else {
            0.04
        };
        let f_approx = f0_scalar + (1.0 - f0_scalar) * (1.0 - cos_theta_i).powi(5);

        let specular_weight = f_approx;
        let diffuse_weight = (1.0 - f_approx) * (1.0 - self.metallic);
        let total_weight = specular_weight + diffuse_weight;

        let random_val: f64 = rng.random();
        let use_specular = if total_weight > 1e-6 {
            random_val < (specular_weight / total_weight)
        } else {
            true
        };

        if use_specular {
            // 鏡面反射をサンプリング（Mirrorに委譲）
            self.specular.sample_direction(normal, incoming, rng)
        } else {
            // 拡散反射をサンプリング（Oren-Nayarに委譲）
            self.diffuse.sample_direction(normal, incoming, rng)
        }
    }
}
