use crate::camera::Ray;
use crate::math::Vector3;
use crate::world::World;
use rand::Rng;
const MIN_DEPTH: u32 = 4;
const THRESHOLD: f64 = 0.01;

pub trait RenderingStrategy {
    fn ray_color(
        world: &World,
        ray: &Ray,
        depth: u32,
        rng: &mut impl Rng,
        throughput: Vector3,
    ) -> Vector3;
}

/// MIS（Multiple Importance Sampling）戦略
#[cfg(feature = "mis")]
pub struct MisStrategy;

#[cfg(feature = "mis")]
impl RenderingStrategy for MisStrategy {
    fn ray_color(
        world: &World,
        ray: &Ray,
        depth: u32,
        rng: &mut impl Rng,
        throughput: Vector3,
    ) -> Vector3 {
        if depth > MIN_DEPTH && throughput.max() < THRESHOLD {
            return Vector3::zero();
        }

        if let Some((hit, obj)) = world.hit_scene(ray, 0.001, f64::INFINITY) {
            let emitted = obj.material.emit(&hit.point, &hit.normal);
            if emitted.length() > 0.0 {
                if depth == 0 {
                    return emitted;
                } else {
                    return Vector3::zero();
                }
            }

            let mut total_radiance = Vector3::zero();

            let incoming = ray.direction;
            if let Some(light_sample) = world.sample_light_point(&hit, rng) {
                // 光源への方向ベクトル
                let to_light = light_sample.point - hit.point;
                let distance = to_light.length();
                let light_dir = to_light.normalize();

                let shadow_ray = Ray::new(hit.point, light_dir);
                let is_visible = world
                    .hit_scene(&shadow_ray, 0.001, distance - 0.001)
                    .is_none();

                if is_visible {
                    // シェーディング点でのcos項
                    let cos_theta = hit.normal.dot(&light_dir).abs();

                    // BRDFを計算
                    let (brdf, pdf_bsdf) =
                        obj.brdf_pdf(&hit.point, &(-incoming), &light_dir, &hit.normal);
                    let w_nee = light_sample.pdf / (light_sample.pdf + pdf_bsdf);

                    // 直接照明の計算: BRDF * emission * cos(θ) / pdf_omega
                    total_radiance +=
                        w_nee * brdf * light_sample.emission * cos_theta / light_sample.pdf;
                }
            }

            let scattered_direction = obj.sample_direction(&hit.normal, &incoming, rng);
            let scattered_ray = Ray::new(hit.point, scattered_direction);
            let cos_theta = scattered_direction.dot(&hit.normal).max(0.0);
            let (brdf, pdf) =
                obj.brdf_pdf(&hit.point, &(-incoming), &scattered_direction, &hit.normal);

            if let Some((scattered_hit, obj)) =
                world.hit_scene(&scattered_ray, 0.001, f64::INFINITY)
            {
                if obj
                    .material
                    .emit(&scattered_hit.point, &scattered_hit.normal)
                    .length()
                    > 0.0
                {
                    let (_sampled_point, _normal, pdf_shape, _light_dir, _d) = obj
                        .shape
                        .sample_surface_from_point(&hit, Some(&scattered_hit), rng);
                    let w_bsdf = pdf / (pdf + pdf_shape);
                    total_radiance += w_bsdf
                        * brdf
                        * obj
                            .material
                            .emit(&scattered_hit.point, &scattered_hit.normal)
                        * cos_theta
                        / pdf;
                } else {
                    let incoming_light =
                        Self::ray_color(world, &scattered_ray, depth + 1, rng, total_radiance);
                    total_radiance += brdf * incoming_light * cos_theta / pdf;
                }
            }

            return total_radiance;
        }

        // 背景は黒
        Vector3::zero()
    }
}
/// NEE（Next Event Estimation）戦略
#[cfg(feature = "nee")]
pub struct NeeStrategy;

#[cfg(feature = "nee")]
impl RenderingStrategy for NeeStrategy {
    fn ray_color(world: &World, ray: &Ray, depth: u32, rng: &mut impl Rng) -> Vector3 {
        if depth == 0 {
            return Vector3::zero();
        }

        if let Some((hit, obj)) = world.hit_scene(ray, 0.001, f64::INFINITY) {
            let emitted = obj.material.emit(&hit.point, &hit.normal);
            if emitted.length() > 0.0 {
                if is_primary_ray {
                    return emitted;
                } else {
                    return Vector3::zero();
                }
            }

            let mut total_radiance = Vector3::zero();

            let incoming = ray.direction;
            if let Some(light_sample) = world.sample_light_point(&hit, rng) {
                // 光源への方向ベクトル
                let to_light = light_sample.point - hit.point;
                let distance = to_light.length();
                let light_dir = to_light.normalize();

                let shadow_ray = Ray::new(hit.point, light_dir);
                let is_visible = world
                    .hit_scene(&shadow_ray, 0.001, distance - 0.001)
                    .is_none();

                if is_visible {
                    // シェーディング点でのcos項
                    let cos_theta = hit.normal.dot(&light_dir).abs();

                    // BRDFを計算
                    let (brdf, _) = obj.brdf_pdf(&hit.point, &(-incoming), &light_dir, &hit.normal);

                    // 直接照明の計算: BRDF * emission * cos(θ) / pdf_omega
                    total_radiance += brdf * light_sample.emission * cos_theta / light_sample.pdf;
                }
            }

            let scattered_direction = obj.sample_direction(&hit.normal, &incoming, rng);
            let scattered_ray = Ray::new(hit.point, scattered_direction);
            let cos_theta = scattered_direction.dot(&hit.normal).max(0.0);
            let (brdf, pdf) =
                obj.brdf_pdf(&hit.point, &(-incoming), &scattered_direction, &hit.normal);

            let incoming_light = Self::ray_color(world, &scattered_ray, depth - 1, false, rng);
            total_radiance += brdf * incoming_light * cos_theta / pdf;

            return total_radiance;
        }

        // 背景は黒
        Vector3::zero()
    }
}

/// BRDFサンプリングのみ戦略
#[cfg(feature = "brdf_only")]
pub struct BrdfOnlyStrategy;

#[cfg(feature = "brdf_only")]
impl RenderingStrategy for BrdfOnlyStrategy {
    fn ray_color(world: &World, ray: &Ray, depth: u32, rng: &mut impl Rng) -> Vector3 {
        // 最大再帰深度に達したら黒を返す
        if depth == 0 {
            return Vector3::zero();
        }

        if let Some((hit, obj)) = world.hit_scene(ray, 0.001, f64::INFINITY) {
            // 発光を取得
            let emitted = obj.material.emit(&hit.point, &hit.normal);

            // 入射方向（レイの方向）
            let incoming = ray.direction;

            // オブジェクトのサンプリング戦略を使って方向を生成
            let scattered_direction = obj.sample_direction(&hit.normal, &incoming, rng);
            let scattered_ray = Ray::new(hit.point, scattered_direction);

            // コサイン項（入射角度による減衰）
            let cos_theta = scattered_direction.dot(&hit.normal).max(0.0);

            // BRDFとPDFを同時に取得（効率的）
            let (brdf, pdf) =
                obj.brdf_pdf(&hit.point, &(-incoming), &scattered_direction, &hit.normal);

            // 入射輝度を再帰的に計算
            let incoming_light = Self::ray_color(world, &scattered_ray, depth - 1, false, rng);

            // レンダリング方程式: 発光 + brdf * 入射輝度 * cos(θ) / pdf
            return emitted + brdf * incoming_light * cos_theta / pdf;
        }

        // 背景は黒
        Vector3::zero()
    }
}
