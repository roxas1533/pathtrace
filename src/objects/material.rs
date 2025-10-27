use crate::{
    camera::Ray,
    math::{Vector3, tan2},
};
use rand::{Rng, RngCore};

/// マテリアルを表すtrait（BSDF、サンプリング、PDFを担当）
pub trait Material: Send + Sync {
    /// BSDFとPDFを同時に計算（BRDF + BTDF）
    ///
    /// # Arguments
    /// * `x` - 交差点の位置
    /// * `ray` - 入射レイ
    /// * `o` - 出射方向（サンプリングされた方向）
    /// * `normal` - 法線ベクトル
    ///
    /// # Returns
    /// (bsdf, pdf) のタプル
    fn bsdf_pdf(&self, x: &Vector3, ray: &Ray, o: &Vector3, normal: &Vector3) -> (Vector3, f64);

    /// サンプリング方向を生成し、BSDFとPDFを同時に取得
    ///
    /// # Arguments
    /// * `x` - 交差点の位置
    /// * `ray` - 入射レイ
    /// * `normal` - 法線ベクトル
    /// * `rng` - 乱数生成器
    ///
    /// # Returns
    /// (sampled_direction, bsdf, pdf) のタプル
    fn bsdf_pdf_sample(
        &self,
        x: &Vector3,
        ray: &Ray,
        normal: &Vector3,
        rng: &mut dyn RngCore,
    ) -> (Vector3, Vector3, f64) {
        let sampled_direction = self.sample_direction(normal, ray, rng);
        let (bsdf, pdf) = self.bsdf_pdf(x, ray, &sampled_direction, normal);
        (sampled_direction, bsdf, pdf)
    }

    /// サンプリング方向を生成
    ///
    /// # Arguments
    /// * `normal` - 法線ベクトル
    /// * `incoming` - 入射方向（カメラから来る方向）
    /// * `rng` - 乱数生成器
    fn sample_direction(&self, normal: &Vector3, incoming: &Ray, rng: &mut dyn RngCore) -> Vector3;

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
    fn bsdf_pdf(&self, _x: &Vector3, _ray: &Ray, o: &Vector3, normal: &Vector3) -> (Vector3, f64) {
        // 拡散反射BRDFを計算
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
    fn bsdf_pdf(
        &self,
        _x: &Vector3,
        _ray: &Ray,
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
    fn bsdf_pdf(&self, _x: &Vector3, ray: &Ray, o: &Vector3, normal: &Vector3) -> (Vector3, f64) {
        let i = -ray.direction;

        // Oren-Nayar BRDFを計算
        // 入射角と出射角（法線からの角度）
        let cos_theta_i = i.dot(normal).max(0.0);
        let cos_theta_o = o.dot(normal).max(0.0);

        // 角度を計算（acosは不安定なのでatan2を使用）
        let sin_theta_i = (1.0 - cos_theta_i * cos_theta_i).max(0.0).sqrt();
        let sin_theta_o = (1.0 - cos_theta_o * cos_theta_o).max(0.0).sqrt();

        // 方位角の差を計算
        let tangent = self.compute_tangent(normal);
        let phi_i = self.compute_azimuth(&i, normal, &tangent);
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

    fn brdf(&self, ray: &Ray, o: &Vector3, normal: &Vector3) -> (Vector3, f64) {
        let i = -ray.direction;

        //D(GGX)
        let alpha = self.roughness * self.roughness;
        let alpha2 = alpha * alpha;
        let h = (i + *o).normalize();
        let denom = (normal.dot(&h) * normal.dot(&h)) * (alpha2 - 1.0) + 1.0;
        let d = alpha2 / (std::f64::consts::PI * denom * denom);

        //G
        let lambda_a =
            |v: &Vector3| -> f64 { (-1.0 + (1.0 + alpha2 * tan2(*v, *normal)).sqrt()) / 2.0 };
        let g = 1.0 / (1.0 + lambda_a(&i) + lambda_a(o));

        //F
        let cos_theta = i.dot(&h).max(0.0);
        // 誘電体のF0をIORから計算: F0 = ((1-ior)/(1+ior))^2
        let f0_dielectric = ((1.0 - self.ior) / (1.0 + self.ior)).powi(2);
        let f0_dielectric_vec = Vector3::new(f0_dielectric, f0_dielectric, f0_dielectric);
        // 金属のF0は色（波長依存の反射率）
        let f0 = f0_dielectric_vec * (1.0 - self.metallic) + self.color * self.metallic;
        // Schlick近似でFresnelを計算
        let f = f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * (1.0 - cos_theta).powi(5);

        let brdf = d * g * f / (4.0 * normal.dot(&i).abs() * normal.dot(o).abs());
        let pdf = d * normal.dot(&h).abs() / (4.0 * i.dot(&h).abs());

        (brdf, pdf)
    }

    fn btdf(&self, ray: &Ray, o: &Vector3, normal: &Vector3) -> (Vector3, f64) {
        let i = -ray.direction;
        let eta = ray.eta_ratio; // η_i / η_t

        let h = -(i * eta + *o).normalize();

        //D(GGX)
        let alpha = self.roughness * self.roughness;
        let alpha2 = alpha * alpha;
        let n_dot_h = normal.dot(&h);
        let denom = (n_dot_h * n_dot_h) * (alpha2 - 1.0) + 1.0;
        let d = alpha2 / (std::f64::consts::PI * denom * denom);

        //G
        let lambda_a =
            |v: &Vector3| -> f64 { (-1.0 + (1.0 + alpha2 * tan2(*v, *normal)).sqrt()) / 2.0 };
        let g = 1.0 / (1.0 + lambda_a(&i) + lambda_a(o));

        //F
        let i_dot_h = i.dot(&h);
        let o_dot_h = o.dot(&h);
        let cos_theta = i_dot_h.abs();

        // 誘電体のF0をIORから計算
        let f0_dielectric = ((1.0 - self.ior) / (1.0 + self.ior)).powi(2);
        let f0_dielectric_vec = Vector3::new(f0_dielectric, f0_dielectric, f0_dielectric);
        // 金属のF0は色（波長依存の反射率）
        let f0 = f0_dielectric_vec * (1.0 - self.metallic) + self.color * self.metallic;
        // Schlick近似でFresnelを計算
        let f = f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * (1.0 - cos_theta).powi(5);

        let i_n = normal.dot(&i).abs();
        let o_n = normal.dot(o).abs();

        let denom_term = eta * i_dot_h + o_dot_h;
        let btdf = (Vector3::one() - f) * d * g * i_dot_h.abs() * o_dot_h.abs()
            / (i_n * o_n * denom_term * denom_term);

        let jacobian = o_dot_h.abs() / (eta * eta * denom_term * denom_term);

        let pdf = d * n_dot_h.abs() * jacobian;

        (btdf, pdf)
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
}

impl Material for Mirror {
    fn bsdf_pdf(&self, _x: &Vector3, ray: &Ray, o: &Vector3, normal: &Vector3) -> (Vector3, f64) {
        // 入射と出射が法線に対して同じ側にあるかチェック（反射 vs 透過）
        let i = -ray.direction;
        let i_dot_n = i.dot(normal);
        let o_dot_n = o.dot(normal);

        if i_dot_n * o_dot_n > 0.0 {
            // 反射（BRDF）
            self.brdf(ray, o, normal)
        } else {
            // 透過（BTDF）
            self.btdf(ray, o, normal)
        }
    }

    fn bsdf_pdf_sample(
        &self,
        x: &Vector3,
        ray: &Ray,
        normal: &Vector3,
        rng: &mut dyn RngCore,
    ) -> (Vector3, Vector3, f64) {
        let i = -ray.direction; // 表面から外向きの方向
        let i_dot_n = i.dot(normal);

        if i_dot_n.abs() < 1e-6 {
            // デジェネレートケース
            return (Vector3::zero(), Vector3::zero(), 1.0);
        }

        // まずFresnelを評価して反射/透過を決定
        let cos_theta_i = i_dot_n.abs();
        let f0 = ((1.0 - self.ior) / (1.0 + self.ior)).powi(2);
        let fresnel = f0 + (1.0 - f0) * (1.0 - cos_theta_i).powi(5);

        // 透過可能かチェック
        let eta = ray.eta_ratio;
        let sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i;
        let sin2_theta_t = eta * eta * sin2_theta_i;
        let can_refract = sin2_theta_t < 1.0;

        // ロシアンルーレットで反射/透過を選択
        let is_reflect = rng.random::<f64>() < fresnel || !can_refract;

        if is_reflect {
            // 反射の場合：反射用のhalf vectorをサンプリング
            let half_vector = self.get_half_vector(normal, rng);
            let i_h = i.dot(&half_vector).abs();

            if i_h < 1e-6 {
                return (Vector3::zero(), Vector3::zero(), 1.0);
            }

            let o_reflect = 2.0 * i.dot(&half_vector) * half_vector - i;
            let (bsdf, pdf) = self.brdf(ray, &o_reflect, normal);
            (o_reflect, bsdf, pdf)
        } else {
            // 透過の場合：透過用のhalf vectorをサンプリング
            let half_vector = self.get_half_vector(normal, rng);
            let i_h = i.dot(&half_vector);

            if i_h.abs() < 1e-6 {
                return (Vector3::zero(), Vector3::zero(), 1.0);
            }

            // half vectorから透過方向を計算
            // h = -(eta_i * i + eta_t * o) / |...|
            // => o = -(h * |...| + eta_i * i) / eta_t
            // より簡単な形式: o = eta * (-i) + (eta * (i·h) - sqrt(...)) * h
            let eta_ih = eta * i_h;
            let k = 1.0 - eta * eta * (1.0 - i_h * i_h);

            let o_refract = if k >= 0.0 {
                eta * (-i) + (eta_ih - k.sqrt()) * half_vector
            } else {
                // 全反射の場合は反射方向に戻す
                2.0 * i.dot(&half_vector) * half_vector - i
            };

            let (bsdf, pdf) = self.btdf(ray, &o_refract, normal);
            (o_refract, bsdf, pdf)
        }
    }

    fn sample_direction(
        &self,
        normal: &Vector3,
        _incoming: &Ray,
        _rng: &mut dyn RngCore,
    ) -> Vector3 {
        // このメソッドはもう使われないが、traitで必須なのでダミー実装
        *normal
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
    fn bsdf_pdf(&self, x: &Vector3, ray: &Ray, o: &Vector3, normal: &Vector3) -> (Vector3, f64) {
        let i = -ray.direction;

        // 鏡面反射成分を取得（MirrorのBRDFに委譲）
        let (specular_brdf, specular_pdf) = self.specular.brdf(ray, o, normal);

        // 拡散反射成分を取得（Oren-NayarのBSDFに委譲）
        let (diffuse_brdf_raw, diffuse_pdf) = self.diffuse.bsdf_pdf(x, ray, o, normal);

        // Fresnel係数を計算
        let h = (i + *o).normalize();
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
