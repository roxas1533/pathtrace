use crate::camera::{Camera, Ray};
use crate::math::Vector3;
use crate::objects::material::{LambertianCosineWeighted, PBRMaterial};
use crate::objects::shape::TriangleShape;
use crate::objects::{Emissive, HitRecord, Hittable, Mirror, Object, SphereShape};
#[cfg(feature = "brdf_only")]
use crate::rendering::BrdfOnlyStrategy;
#[cfg(feature = "mis")]
use crate::rendering::MisStrategy;
#[cfg(feature = "nee")]
use crate::rendering::NeeStrategy;
use crate::rendering::RenderingStrategy;
use rand::Rng;
use std::sync::Mutex;

pub const WIDTH: u32 = 400;
pub const HEIGHT: u32 = 400;
pub const SAMPLE_NUM: u32 = 2000; // 1ピクセルあたりのサンプル数

#[derive(Clone, Copy)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl std::ops::Add for Color {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            r: self.r.saturating_add(other.r),
            g: self.g.saturating_add(other.g),
            b: self.b.saturating_add(other.b),
            a: 255,
        }
    }
}

impl std::ops::AddAssign for Color {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

/// 光源サンプリングの結果
pub struct LightSample {
    pub point: Vector3,
    pub emission: Vector3,
    pub pdf: f64,
}

pub struct World {
    pub data: Mutex<[Color; WIDTH as usize * HEIGHT as usize]>,
    camera: Camera,
    objects: Vec<Object>,
    /// 光源オブジェクトのインデックスリスト
    light_indices: Vec<usize>,
}

impl World {
    pub fn new() -> Self {
        // カメラを設定
        let camera = Camera::new(
            Vector3::new(0.0, 0.0, 2.0), // カメラ位置
            WIDTH,
            HEIGHT,
            1.0,  // スクリーンとの距離
            35.0, // 視野角（度）
        );

        // コーネルボックスの定義
        let box_size = 1.0;
        let box_depth = -2.0;
        let light_size = 0.3;

        let objects: Vec<Object> = vec![
            // 左壁（赤）- 2つの三角形
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-box_size, -box_size, box_depth - box_size),
                    Vector3::new(-box_size, box_size, box_depth - box_size),
                    Vector3::new(-box_size, box_size, box_depth + box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.8, 0.1, 0.1))),
            ),
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-box_size, -box_size, box_depth - box_size),
                    Vector3::new(-box_size, box_size, box_depth + box_size),
                    Vector3::new(-box_size, -box_size, box_depth + box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.8, 0.1, 0.1))),
            ),
            // 右壁（緑）- 2つの三角形
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(box_size, -box_size, box_depth - box_size),
                    Vector3::new(box_size, box_size, box_depth + box_size),
                    Vector3::new(box_size, box_size, box_depth - box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.1, 0.8, 0.1))),
            ),
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(box_size, -box_size, box_depth - box_size),
                    Vector3::new(box_size, -box_size, box_depth + box_size),
                    Vector3::new(box_size, box_size, box_depth + box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.1, 0.8, 0.1))),
            ),
            // 奥壁（白）- 2つの三角形
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-box_size, -box_size, box_depth - box_size),
                    Vector3::new(box_size, -box_size, box_depth - box_size),
                    Vector3::new(box_size, box_size, box_depth - box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.8, 0.8, 0.8))),
            ),
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-box_size, -box_size, box_depth - box_size),
                    Vector3::new(box_size, box_size, box_depth - box_size),
                    Vector3::new(-box_size, box_size, box_depth - box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.8, 0.8, 0.8))),
            ),
            // 下壁（白）- 2つの三角形
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-box_size, -box_size, box_depth - box_size),
                    Vector3::new(box_size, -box_size, box_depth + box_size),
                    Vector3::new(box_size, -box_size, box_depth - box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.8, 0.8, 0.8))),
            ),
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-box_size, -box_size, box_depth - box_size),
                    Vector3::new(-box_size, -box_size, box_depth + box_size),
                    Vector3::new(box_size, -box_size, box_depth + box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.8, 0.8, 0.8))),
            ),
            // 上壁（白）- 2つの三角形
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-box_size, box_size, box_depth - box_size),
                    Vector3::new(box_size, box_size, box_depth - box_size),
                    Vector3::new(box_size, box_size, box_depth + box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.8, 0.8, 0.8))),
            ),
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-box_size, box_size, box_depth - box_size),
                    Vector3::new(box_size, box_size, box_depth + box_size),
                    Vector3::new(-box_size, box_size, box_depth + box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.8, 0.8, 0.8))),
            ),
            // 天井の光源
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-light_size, box_size - 0.01, box_depth - light_size),
                    Vector3::new(light_size, box_size - 0.01, box_depth - light_size),
                    Vector3::new(light_size, box_size - 0.01, box_depth + light_size),
                )),
                Box::new(Emissive::new(Vector3::new(15.0, 15.0, 15.0))),
            ),
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-light_size, box_size - 0.01, box_depth - light_size),
                    Vector3::new(light_size, box_size - 0.01, box_depth + light_size),
                    Vector3::new(-light_size, box_size - 0.01, box_depth + light_size),
                )),
                Box::new(Emissive::new(Vector3::new(15.0, 15.0, 15.0))),
            ),
            // 球体光源
            // Object::new(
            //     Box::new(SphereShape::new(
            //         Vector3::new(0.0, box_size - 0.21, box_depth),
            //         0.2,
            //     )),
            //     Box::new(Emissive::new(Vector3::new(15.0, 15.0, 15.0))),
            // ),
            // 中央の球体（テスト用）
            // Object::new(
            //     Box::new(SphereShape::new(Vector3::new(-0.4, -0.5, box_depth), 0.4)),
            //     Box::new(Mirror {
            //         roughness: 0.05,
            //         color: Vector3::new(0.1, 0.1, 0.1),
            //         metallic: 0.0,
            //         ior: 0.35,
            //     }),
            // ),
            Object::new(
                Box::new(SphereShape::new(Vector3::new(0.4, -0.5, box_depth), 0.4)),
                Box::new(PBRMaterial::new(
                    0.3,                              // roughness: やや滑らかなプラスチック
                    Vector3::new(0.1, 0.1, 0.8), // albedo: 青色
                    0.0                               // metallic: 非金属（プラスチック）
                )),
            ),
            // Object::new(
            //     Box::new(SphereShape::new(Vector3::new(0.4, -0.5, box_depth), 0.4)),
            //     Box::new(LambertianCosineWeighted::new(Vector3::new(0.8, 0.8, 0.8))),
            // ),
        ];

        // 光源オブジェクトを識別
        let light_indices: Vec<usize> = objects
            .iter()
            .enumerate()
            .filter(|(_, obj)| {
                // ダミーの点で発光チェック
                let emission = obj
                    .material
                    .emit(&Vector3::zero(), &Vector3::new(0.0, 1.0, 0.0));
                emission.length() > 0.0
            })
            .map(|(i, _)| i)
            .collect();

        Self {
            data: Mutex::new(
                [Color {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 255,
                }; WIDTH as usize * HEIGHT as usize],
            ),
            camera,
            objects,
            light_indices,
        }
    }

    /// 光源上の点をサンプリング
    ///
    /// # Arguments
    /// * `point` - 衝突点（光源サンプリングの基準点）
    /// * `rng` - 乱数生成器
    ///
    /// # Returns
    /// サンプリングされた光源情報、光源がない場合はNone
    pub fn sample_light_point(&self, hit: &HitRecord, rng: &mut impl Rng) -> Option<LightSample> {
        if self.light_indices.is_empty() {
            return None;
        }
        let light_idx = self.light_indices[rng.random_range(0..self.light_indices.len())];
        let light_obj = &self.objects[light_idx];
        let (sampled_point, normal, pdf_shape, _light_dir, _d) =
            light_obj.shape.sample_surface_from_point(hit, None, rng);
        let emission = light_obj.material.emit(&sampled_point, &normal);
        let pdf = pdf_shape / self.light_indices.len() as f64;

        Some(LightSample {
            point: sampled_point,
            emission,
            pdf,
        })
    }

    // シーン内のオブジェクトとの交差判定（ヒット情報とオブジェクトのペアを返す）
    pub fn hit_scene(
        &self,
        ray: &Ray,
        t_min: f64,
        t_max: f64,
    ) -> Option<(crate::objects::HitRecord, &Object)> {
        let mut closest_hit = None;
        let mut closest_so_far = t_max;
        let mut hit_obj = None;

        // すべてのオブジェクトに対して交差判定
        for obj in &self.objects {
            if let Some(hit) = obj.hit(ray, t_min, closest_so_far) {
                closest_so_far = hit.t;
                closest_hit = Some(hit);
                hit_obj = Some(obj);
            }
        }

        closest_hit.map(|hit| (hit, hit_obj.unwrap()))
    }

    // 1ピクセルをレンダリング（読み取り専用でカメラとオブジェクト情報を使用）
    pub fn render_pixel(&self, x: u32, y: u32, rng: &mut impl Rng) -> Color {
        let mut color_temp = Vector3::new(0.0, 0.0, 0.0);

        for _ in 0..SAMPLE_NUM {
            let mut ray =
                self.camera
                    .get_ray_with_offset(x, HEIGHT - 1 - y, rng.random(), rng.random());

            #[cfg(feature = "nee")]
            {
                color_temp += NeeStrategy::ray_color(self, &ray, 0, true, rng);
            }
            #[cfg(feature = "brdf_only")]
            {
                color_temp += BrdfOnlyStrategy::ray_color(self, &ray, 0, true, rng);
            }
            #[cfg(feature = "mis")]
            {
                color_temp += MisStrategy::ray_color(self, &mut ray, 0, rng, Vector3::one());
            }
        }

        color_temp = color_temp / SAMPLE_NUM as f64;

        // ガンマ補正 (gamma = 2.0)
        color_temp.x = color_temp.x.sqrt();
        color_temp.y = color_temp.y.sqrt();
        color_temp.z = color_temp.z.sqrt();

        // 0-1の範囲にクランプして0-255に変換
        Color {
            r: (color_temp.x.clamp(0.0, 1.0) * 255.0) as u8,
            g: (color_temp.y.clamp(0.0, 1.0) * 255.0) as u8,
            b: (color_temp.z.clamp(0.0, 1.0) * 255.0) as u8,
            a: 255,
        }
    }

    pub fn draw(&self, frame: &mut [u8]) {
        let data = self.data.lock().unwrap();
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let color = data[i];
            pixel.copy_from_slice(&[color.r, color.g, color.b, color.a]);
        }
    }
}
