use crate::camera::{Camera, Ray};
use crate::math::Vector3;
use crate::objects::material::LambertianCosineWeighted;
use crate::objects::shape::TriangleShape;
use crate::objects::{Emissive, Hittable, Mirror, Object, SphereShape};
use rand::Rng;

pub const WIDTH: u32 = 400;
pub const HEIGHT: u32 = 400;
pub const SAMPLE_NUM: u32 = 50; // 1ピクセルあたりのサンプル数

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
    /// サンプリングされた点
    pub point: Vector3,
    /// 法線
    pub normal: Vector3,
    /// 発光
    pub emission: Vector3,
    /// PDF（立体角測度または面積測度）
    pub pdf: f64,
}

pub struct World {
    pub data: [Color; WIDTH as usize * HEIGHT as usize],
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
            // 天井の光源（小さめの四角形）- 2つの三角形
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
            // 中央の球体（テスト用）
            Object::new(
                Box::new(SphereShape::new(Vector3::new(-0.4, -0.5, box_depth), 0.4)),
                Box::new(Mirror {
                    roughness: 0.01,
                    color: Vector3::new(0.9, 0.9, 0.9),
                }),
            ),
            Object::new(
                Box::new(SphereShape::new(Vector3::new(0.4, -0.5, box_depth), 0.4)),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.8, 0.8, 0.8))),
            ),
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
            data: [Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }; WIDTH as usize * HEIGHT as usize],
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
    pub fn sample_light_point(&self, point: &Vector3, rng: &mut impl Rng) -> Option<LightSample> {
        if self.light_indices.is_empty() {
            return None;
        }

        // 1. ランダムに光源を選択
        let light_idx = self.light_indices[rng.random_range(0..self.light_indices.len())];
        let light_obj = &self.objects[light_idx];

        // 2. 光源の形状上の点をサンプリング
        let (sampled_point, normal, pdf_shape) =
            light_obj.shape.sample_surface_from_point(point, rng);

        // 3. 発光を取得
        let emission = light_obj.material.emit(&sampled_point, &normal);

        // 4. 複数光源がある場合、選択確率を考慮
        // 光源選択確率 1/N を掛けるので、PDFは pdf_shape / N
        let pdf = pdf_shape / self.light_indices.len() as f64;

        Some(LightSample {
            point: sampled_point,
            normal,
            emission,
            pdf,
        })
    }

    // シーン内のオブジェクトとの交差判定（ヒット情報とオブジェクトのペアを返す）
    fn hit_scene(
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

    // ===== 元のBRDFサンプリング実装（コメントアウト） =====
    // fn ray_color(&self, ray: &Ray, depth: u32, rng: &mut impl Rng) -> Vector3 {
    //     // 最大再帰深度に達したら黒を返す
    //     if depth == 0 {
    //         return Vector3::zero();
    //     }
    //
    //     // オブジェクトとの交差をチェック（自己交差を避けるため0.001から）
    //     if let Some((hit, obj)) = self.hit_scene(ray, 0.001, f64::INFINITY) {
    //         // 発光を取得
    //         let emitted = obj.material.emit(&hit.point, &hit.normal);
    //
    //         // 入射方向（レイの方向）
    //         let incoming = ray.direction;
    //
    //         // オブジェクトのサンプリング戦略を使って方向を生成
    //         let scattered_direction = obj.sample_direction(&hit.normal, &incoming, rng);
    //         let scattered_ray = Ray::new(hit.point, scattered_direction);
    //
    //         // コサイン項（入射角度による減衰）
    //         let cos_theta = scattered_direction.dot(&hit.normal).max(0.0);
    //
    //         // BRDFとPDFを同時に取得（効率的）
    //         let (brdf, pdf) =
    //             obj.brdf_pdf(&hit.point, &(-incoming), &scattered_direction, &hit.normal);
    //
    //         // 入射輝度を再帰的に計算
    //         let incoming_light = self.ray_color(&scattered_ray, depth - 1, rng);
    //
    //         // レンダリング方程式: 発光 + brdf * 入射輝度 * cos(θ) / pdf
    //         return emitted + brdf * incoming_light * cos_theta / pdf;
    //     }
    //
    //     // 背景は黒
    //     Vector3::zero()
    // }

    // レイの色を計算（NEE + 間接照明）
    fn ray_color(&self, ray: &Ray, depth: u32, max_depth: u32, rng: &mut impl Rng) -> Vector3 {
        // 最大再帰深度に達したら黒を返す
        if depth == 0 {
            return Vector3::zero();
        }

        // オブジェクトとの交差をチェック（自己交差を避けるため0.001から）
        if let Some((hit, obj)) = self.hit_scene(ray, 0.001, f64::INFINITY) {
            // 発光を取得
            let emitted = obj.material.emit(&hit.point, &hit.normal);

            // 光源にヒットした場合の処理
            if emitted.length() > 0.0 {
                // カメラから直接見える場合（最初のレイ）のみ発光を返す
                if depth == max_depth {
                    return emitted;
                } else {
                    // 間接照明でヒットした場合は計算を打ち切る（NEEで計算済み）
                    return Vector3::zero();
                }
            }

            let mut total_radiance = Vector3::zero();

            // 入射方向（カメラから来る方向）
            let incoming = ray.direction;

            // NEE（Next Event Estimation）: 直接照明
            if let Some(light_sample) = self.sample_light_point(&hit.point, rng) {
                // 光源への方向ベクトル
                let to_light = light_sample.point - hit.point;
                let distance = to_light.length();
                let light_dir = to_light.normalize();

                let shadow_ray = Ray::new(hit.point, light_dir);
                let is_visible = if let Some((_shadow_hit, _)) =
                    self.hit_scene(&shadow_ray, 0.001, distance - 0.001)
                {
                    // 光源までの間に何かある場合は遮蔽されている
                    false
                } else {
                    true
                };

                if is_visible {
                    // 衝突点の法線とのコサイン（表面のみ）
                    let cos_theta = hit.normal.dot(&light_dir).abs();
                    let cos_light = light_sample.normal.dot(&(-light_dir)).abs();

                    // BRDFを計算
                    let (brdf, _) = obj.brdf_pdf(&hit.point, &(-incoming), &light_dir, &hit.normal);

                    // 面積測度PDFから立体角測度への変換を含む幾何項
                    let geometry_term = (cos_theta * cos_light) / (distance * distance);

                    // 直接照明の計算: BRDF * emission * geometry_term / pdf_area
                    total_radiance += brdf * light_sample.emission * geometry_term / light_sample.pdf;
                }
            }

            // BRDF sampling: 間接照明
            let scattered_direction = obj.sample_direction(&hit.normal, &incoming, rng);
            let scattered_ray = Ray::new(hit.point, scattered_direction);

            // コサイン項（入射角度による減衰）
            let cos_theta = scattered_direction.dot(&hit.normal).max(0.0);

            // BRDFとPDFを取得
            let (brdf, pdf) = obj.brdf_pdf(&hit.point, &(-incoming), &scattered_direction, &hit.normal);

            // 入射輝度を再帰的に計算（次の衝突点でもNEEが行われる）
            let incoming_light = self.ray_color(&scattered_ray, depth - 1, max_depth, rng);

            // 間接照明: brdf * 入射輝度 * cos(θ) / pdf
            total_radiance += brdf * incoming_light * cos_theta / pdf;

            return total_radiance;
        }

        // 背景は黒
        Vector3::zero()
    }

    // 1ピクセルをレンダリング（読み取り専用でカメラとオブジェクト情報を使用）
    pub fn render_pixel(&self, x: u32, y: u32, rng: &mut impl Rng) -> Color {
        let mut color_temp = Vector3::new(0.0, 0.0, 0.0);
        const MAX_DEPTH: u32 = 5; // 最大再帰深度

        for _ in 0..SAMPLE_NUM {
            let ray =
                self.camera
                    .get_ray_with_offset(x, HEIGHT - 1 - y, rng.random(), rng.random());

            color_temp += self.ray_color(&ray, MAX_DEPTH, MAX_DEPTH, rng);
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
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let color = self.data[i];
            pixel.copy_from_slice(&[color.r, color.g, color.b, color.a]);
        }
    }
}
