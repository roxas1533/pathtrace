use crate::{camera::Ray, math::Vector3, objects::material::Material};
use rand::{Rng, RngCore};

/// 鏡面反射マテリアル（金属と誘電体の両対応）
pub struct Mirror {
    /// 粗さ
    pub roughness: f64,
    /// 色
    pub color: Vector3,
    /// 金属度合（0=誘電体、1=金属）
    pub metallic: f64,
    /// 屈折率（IOR）
    pub ior: f64,
}

impl Mirror {
    fn sample_ggx_vndf(&self, view: &Vector3, normal: &Vector3, rng: &mut dyn RngCore) -> Vector3 {
        let alpha = self.roughness * self.roughness;

        // ローカル座標系を構築
        let up = if normal.y.abs() > 0.999 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let tangent = up.cross(normal).normalize();
        let bitangent = normal.cross(&tangent);

        // 視線ベクトルをローカル座標系に変換
        let v_local = Vector3::new(view.dot(&tangent), view.dot(&bitangent), view.dot(normal));

        let vh = Vector3::new(alpha * v_local.x, alpha * v_local.y, v_local.z).normalize();

        let lensq = vh.x * vh.x + vh.y * vh.y;
        let t1 = if lensq > 0.0 {
            Vector3::new(-vh.y, vh.x, 0.0) * (1.0 / lensq.sqrt())
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let t2 = vh.cross(&t1);

        let r1: f64 = rng.random();
        let r2: f64 = rng.random();
        let r = r1.sqrt();
        let phi = 2.0 * std::f64::consts::PI * r2;
        let t1_coord = r * phi.cos();
        let mut t2_coord = r * phi.sin();
        let s = 0.5 * (1.0 + vh.z);
        t2_coord = (1.0 - s) * (1.0 - t1_coord * t1_coord).sqrt() + s * t2_coord;

        let nh = t1 * t1_coord
            + t2 * t2_coord
            + vh * (1.0 - t1_coord * t1_coord - t2_coord * t2_coord)
                .max(0.0)
                .sqrt();

        let ne_local = Vector3::new(alpha * nh.x, alpha * nh.y, nh.z.max(0.0)).normalize();

        (tangent * ne_local.x + bitangent * ne_local.y + *normal * ne_local.z).normalize()
    }

    pub(crate) fn brdf(&self, ray: &Ray, o: &Vector3, normal: &Vector3) -> (Vector3, f64) {
        let i = -ray.direction;

        //D(GGX)
        let alpha = self.roughness * self.roughness;
        let alpha2 = alpha * alpha;
        let h = (i + *o).normalize();
        let denom = (normal.dot(&h) * normal.dot(&h)) * (alpha2 - 1.0) + 1.0;
        let d = alpha2 / (std::f64::consts::PI * denom * denom);

        //G
        let i_n = normal.dot(&i).max(0.0);
        let o_n = normal.dot(o).max(0.0);
        let g = self.get_g(i_n, o_n);

        //F
        let cos_theta = i.dot(&h).max(0.0);
        let f = self.get_f(cos_theta);

        let denom_brdf = 4.0 * i_n * o_n;
        let brdf = d * g * f / denom_brdf;

        let i_h = i.dot(&h).abs();
        let pdf = d * normal.dot(&h).abs() / (4.0 * i_h);

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
        let i_n = normal.dot(&i).abs();
        let o_n = normal.dot(o).abs();
        let g = self.get_g(i_n, o_n);

        //F
        let i_dot_h = i.dot(&h);
        let o_dot_h = o.dot(&h);
        let cos_theta = i_dot_h.abs();
        let denom_term = eta * i_dot_h + o_dot_h;

        let f = self.get_f(cos_theta);

        let btdf = (Vector3::one() - f) * d * g * i_dot_h.abs() * o_dot_h.abs()
            / (i_n * o_n * denom_term * denom_term);

        let jacobian = o_dot_h.abs() / (denom_term * denom_term);

        let pdf = d * n_dot_h.abs() * jacobian;

        (btdf, pdf)
    }

    fn get_f(&self, cos_theta: f64) -> Vector3 {
        // 誘電体のF0をIORから計算: F0 = ((1-ior)/(1+ior))^2
        let f0_dielectric = ((1.0 - self.ior) / (1.0 + self.ior)).powi(2);
        let f0_dielectric_vec = Vector3::new(f0_dielectric, f0_dielectric, f0_dielectric);
        let f0 = f0_dielectric_vec * (1.0 - self.metallic) + self.color * self.metallic;
        f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * (1.0 - cos_theta).powi(5)
    }

    /// Smith's G1 term (単一方向のshadowing-masking)
    /// 数値安定な公式を使用：G1 = 2*cos(θ) / (cos(θ) + sqrt(α² + (1-α²)*cos²(θ)))
    fn get_g1(&self, cos_theta: f64) -> f64 {
        // cos_theta <= 0 は物理的に寄与なし（grazing/back-facing）
        if cos_theta <= 0.0 {
            return 0.0;
        }

        let alpha = self.roughness * self.roughness;
        let alpha2 = alpha * alpha;
        let cos2_theta = cos_theta * cos_theta;

        // 数値安定な公式：tan^2を直接計算せず、別の等価な形を使用
        let term = alpha2 + (1.0 - alpha2) * cos2_theta;
        2.0 * cos_theta / (cos_theta + term.sqrt())
    }

    /// Smith's G2 term (双方向のshadowing-masking)
    /// 数値安定な公式を使用：G1を利用してG2を計算
    fn get_g(&self, cos_theta_i: f64, cos_theta_o: f64) -> f64 {
        // 物理的に寄与なし
        if cos_theta_i <= 0.0 || cos_theta_o <= 0.0 {
            return 0.0;
        }

        let alpha = self.roughness * self.roughness;
        let alpha2 = alpha * alpha;

        // 各方向のλ(ラムダ)を数値安定に計算
        // G1(v) = 1/(1+λ(v)) から λ(v) = (1/G1(v)) - 1 だが、
        // 直接計算する方が安定：λ = (-1 + sqrt(1 + α²tan²θ)) / 2
        let lambda = |cos_theta: f64| -> f64 {
            let cos2_theta = cos_theta * cos_theta;
            // sqrt(1 + α²tan²θ) = sqrt(1 + α²(1-cos²θ)/cos²θ)
            //                   = sqrt((cos²θ + α² - α²cos²θ) / cos²θ)
            //                   = sqrt(α² + (1-α²)cos²θ) / cos_theta
            let numerator = (alpha2 + (1.0 - alpha2) * cos2_theta).sqrt();
            (numerator - cos_theta) / (2.0 * cos_theta)
        };

        1.0 / (1.0 + lambda(cos_theta_i) + lambda(cos_theta_o))
    }
}

impl Material for Mirror {
    fn bsdf_pdf(&self, _x: &Vector3, ray: &Ray, o: &Vector3, normal: &Vector3) -> (Vector3, f64) {
        let i = -ray.direction;
        let i_dot_n = i.dot(normal);
        let o_dot_n = o.dot(normal);

        let is_reflection = i_dot_n * o_dot_n > 0.0;

        // 金属の場合、透過側への寄与は0
        if self.metallic > 0.99 && !is_reflection {
            return (Vector3::zero(), 1.0);
        }

        if is_reflection {
            // 反射（BRDF）
            self.brdf(ray, o, normal)
        } else {
            // 透過（BTDF）
            self.btdf(ray, o, normal)
        }
    }

    fn bsdf_pdf_sample(
        &self,
        _x: &Vector3,
        ray: &Ray,
        normal: &Vector3,
        rng: &mut dyn RngCore,
    ) -> (Vector3, Vector3, f64, f64) {
        let i = -ray.direction;
        let i_dot_n = i.dot(normal);
        let eta = ray.eta_ratio;

        // VNDF samplingを使ってhalf-vectorをサンプリング
        let h = self.sample_ggx_vndf(&i, normal, rng);

        let i_h = i.dot(&h);
        if i_h <= 0.0 {
            return (*normal, Vector3::zero(), 1.0, 0.0);
        }

        let mut fresnel_vec = self.get_f(i_h);

        // 透過可能かチェック
        let sin2_theta_i = 1.0 - i_h * i_h;
        let cos2_theta_t = 1.0 - (eta * eta) * sin2_theta_i;
        let total_reflection = cos2_theta_t < 0.0;

        let mut rr_f = fresnel_vec.x;
        if total_reflection || self.metallic > 0.99 {
            rr_f = 1.0;
            fresnel_vec = Vector3::new(1.0, 1.0, 1.0);
        }

        let is_reflect = rng.random::<f64>() < rr_f;

        let alpha = self.roughness * self.roughness;
        let alpha2 = alpha * alpha;
        let n_h = normal.dot(&h);

        let denom = (n_h * n_h) * (alpha2 - 1.0) + 1.0;
        let d = alpha2 / (std::f64::consts::PI * denom * denom);

        if is_reflect {
            let o = 2.0 * i_h * h - i;
            let o_normalized = o.normalize();
            let o_n = normal.dot(&o_normalized).max(0.0);
            let i_n = i_dot_n.max(0.0);

            let g = self.get_g(i_n, o_n);
            let denom_brdf = 4.0 * i_n * o_n;
            let brdf = fresnel_vec * d * g / (denom_brdf * rr_f);

            // VNDF PDF計算: pdf_vndf(h) = G1(v, h) * D(h) * max(0, v·h) / (v·n)
            // 反射のPDF = pdf_vndf(h) / (4 * |v·h|)
            let g1_v = self.get_g1(i_n);
            let pdf_vndf = g1_v * d * i_h.max(0.0) / i_n;
            let pdf = pdf_vndf / (4.0 * i_h.abs());

            // 数値エラーチェック（non-biased: 計算後に有効性を確認）
            if !brdf.x.is_finite()
                || !brdf.y.is_finite()
                || !brdf.z.is_finite()
                || !pdf.is_finite()
                || pdf <= 0.0
            {
                return (*normal, Vector3::zero(), 1.0, 0.0);
            }

            let cos_theta = o_n;
            (o_normalized, brdf, pdf, cos_theta)
        } else {
            let cos_theta_t = cos2_theta_t.sqrt();
            let o = h * (eta * i_h - cos_theta_t) - i * eta;
            let o_normalized = o.normalize();
            let o_h = o_normalized.dot(&h);
            let o_n = normal.dot(&o_normalized).abs();
            let i_n = i_dot_n.abs();

            let denom_term = eta * i_h + o_h;
            let g = self.get_g(i_n, o_n);
            let one_f = Vector3::one() - fresnel_vec;

            let btdf = one_f * d * g * i_h.abs() * o_h.abs()
                / (i_n * o_n * denom_term * denom_term * (1.0 - rr_f));

            let jacobian = o_h.abs() / (denom_term * denom_term);

            // VNDF PDF計算: pdf_vndf(h) = G1(v, h) * D(h) * max(0, v·h) / (v·n)
            // 透過のPDF = pdf_vndf(h) * jacobian
            let g1_v = self.get_g1(i_n);
            let pdf_vndf = g1_v * d * i_h.max(0.0) / i_n;
            let pdf = pdf_vndf * jacobian;

            // 数値エラーチェック（non-biased: 計算後に有効性を確認）
            if !btdf.x.is_finite()
                || !btdf.y.is_finite()
                || !btdf.z.is_finite()
                || !pdf.is_finite()
                || pdf <= 0.0
            {
                return (*normal, Vector3::zero(), 1.0, 0.0);
            }

            let cos_theta = o_n;
            (o_normalized, btdf, pdf, cos_theta)
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
