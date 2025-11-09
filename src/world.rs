use crate::camera::{Camera, Ray};
use crate::math::Vector3;
use crate::objects::material::LambertianCosineWeighted;
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
pub const SAMPLE_NUM: u32 = 3000;

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
    /// ガンマ補正前の輝度値（デバッグ用）
    pub luminance_data: Mutex<Vec<Vector3>>,
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
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.2, 0.2, 0.8))),
            ),
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-box_size, -box_size, box_depth - box_size),
                    Vector3::new(box_size, box_size, box_depth - box_size),
                    Vector3::new(-box_size, box_size, box_depth - box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.2, 0.2, 0.8))),
            ),
            // 下壁（白）- 2つの三角形
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-box_size, -box_size, box_depth - box_size),
                    Vector3::new(box_size, -box_size, box_depth + box_size),
                    Vector3::new(box_size, -box_size, box_depth - box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.2, 0.8, 0.8))),
            ),
            Object::new(
                Box::new(TriangleShape::new(
                    Vector3::new(-box_size, -box_size, box_depth - box_size),
                    Vector3::new(-box_size, -box_size, box_depth + box_size),
                    Vector3::new(box_size, -box_size, box_depth + box_size),
                )),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.2, 0.8, 0.8))),
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
            //     Box::new(Emissive::new(Vector3::new(36.0, 36.0, 36.0))),
            // ),
            // 鏡面反射球（metallic=1）
            // Object::new(
            //     Box::new(SphereShape::new(Vector3::new(-0.4, -0.6, box_depth), 0.4)),
            //     Box::new(Mirror {
            //         roughness: 0.01,
            //         color: Vector3::new(1.0, 1.0, 1.0),
            //         metallic: 1.0,
            //         ior: 1.5,
            //     }),
            // ),
            // ガラス球
            Object::new(
                Box::new(SphereShape::new(Vector3::new(0.4, -0.6, box_depth), 0.4)),
                Box::new(Mirror {
                    roughness: 0.3,
                    color: Vector3::new(1.0, 1.0, 1.0),
                    metallic: 0.0,
                    ior: 1.5,
                }),
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
            data: Mutex::new(
                [Color {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 255,
                }; WIDTH as usize * HEIGHT as usize],
            ),
            luminance_data: Mutex::new(vec![Vector3::zero(); WIDTH as usize * HEIGHT as usize]),
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
                color_temp += NeeStrategy::ray_color(self, &mut ray, 0, rng, Vector3::one());
            }
            #[cfg(feature = "brdf_only")]
            {
                color_temp += BrdfOnlyStrategy::ray_color(self, &mut ray, 0, rng, Vector3::one());
            }
            #[cfg(feature = "mis")]
            {
                color_temp += MisStrategy::ray_color(self, &mut ray, 0, rng, Vector3::one());
            }
        }

        color_temp = color_temp / SAMPLE_NUM as f64;

        // ガンマ補正前の輝度値を保存（デバッグ用）
        let index = (y * WIDTH + x) as usize;
        self.luminance_data.lock().unwrap()[index] = color_temp;

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

    /// 輝度値をCSVファイルに出力（デバッグ用）
    pub fn export_luminance(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;
        let data = self.luminance_data.lock().unwrap();

        // ヘッダー行
        writeln!(file, "x,y,r,g,b,luminance")?;

        // データ行
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let index = (y * WIDTH + x) as usize;
                let v = data[index];
                let luminance = 0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z;
                writeln!(
                    file,
                    "{},{},{:.6},{:.6},{:.6},{:.6}",
                    x, y, v.x, v.y, v.z, luminance
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_pixel_79_176() {
        // ピクセル(79, 176)の問題を調査
        let x = 79;
        let y = 176;

        // main.rsと同じseed計算
        let seed = ((y as u64) << 32) | (x as u64);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let world = World::new();

        let mut color_temp = Vector3::new(0.0, 0.0, 0.0);
        let mut max_sample = Vector3::zero();
        let mut max_luminance = 0.0;
        let mut max_sample_idx = 0;
        let mut high_contribution_count = 0;

        println!("\n=== Testing Pixel ({}, {}) with seed {} ===", x, y, seed);

        for sample_idx in 0..SAMPLE_NUM {
            let mut ray =
                world
                    .camera
                    .get_ray_with_offset(x, HEIGHT - 1 - y, rng.random(), rng.random());

            #[cfg(feature = "brdf_only")]
            let sample_color = crate::rendering::BrdfOnlyStrategy::ray_color(
                &world,
                &mut ray,
                0,
                &mut rng,
                Vector3::one(),
            );

            #[cfg(not(feature = "brdf_only"))]
            let sample_color = Vector3::zero();

            let luminance = sample_color.luminance();

            if luminance > 10.0 {
                high_contribution_count += 1;
                println!(
                    "Sample #{}: luminance={:.2}, RGB=({:.6}, {:.6}, {:.6})",
                    sample_idx, luminance, sample_color.x, sample_color.y, sample_color.z
                );
            }

            if luminance > max_luminance {
                max_luminance = luminance;
                max_sample = sample_color;
                max_sample_idx = sample_idx;
            }

            color_temp += sample_color;
        }

        color_temp = color_temp / SAMPLE_NUM as f64;

        println!("\n=== Results ===");
        println!(
            "Average color (before gamma): ({:.6}, {:.6}, {:.6})",
            color_temp.x, color_temp.y, color_temp.z
        );
        println!("Average luminance: {:.6}", color_temp.luminance());
        println!("Max sample index: #{}", max_sample_idx);
        println!("Max sample luminance: {:.2}", max_luminance);
        println!(
            "Max sample RGB: ({:.6}, {:.6}, {:.6})",
            max_sample.x, max_sample.y, max_sample.z
        );
        println!(
            "Max sample contribution to average: ({:.6}, {:.6}, {:.6})",
            max_sample.x / SAMPLE_NUM as f64,
            max_sample.y / SAMPLE_NUM as f64,
            max_sample.z / SAMPLE_NUM as f64
        );
        println!(
            "High contribution samples (>10): {}/{}",
            high_contribution_count, SAMPLE_NUM
        );

        // ガンマ補正
        let gamma_corrected = Vector3::new(
            color_temp.x.sqrt(),
            color_temp.y.sqrt(),
            color_temp.z.sqrt(),
        );
        println!(
            "After gamma correction: ({:.6}, {:.6}, {:.6})",
            gamma_corrected.x, gamma_corrected.y, gamma_corrected.z
        );

        // 期待値との比較（隣のピクセルの値: 0.103, 0.381, 0.377）
        // ガンマ補正後なので、補正前は (0.103^2, 0.381^2, 0.377^2) ≈ (0.011, 0.145, 0.142)
        println!("\nExpected (from neighbor): ~(0.011, 0.145, 0.142) before gamma");

        // 異常に高い値かチェック
        if color_temp.x > 0.5 || color_temp.y > 0.5 || color_temp.z > 0.5 {
            println!("\n⚠️  WARNING: Abnormally high values detected!");
        }
    }

    #[test]
    fn test_single_sample_analysis() {
        // Sample #26を含む最初の50サンプルを実行し、各サンプルの寄与を記録
        let x = 79;
        let y = 176;

        // デバッグログを初期化
        crate::log::init_log("test_sample_26.log").expect("Failed to create log file");

        let seed = ((y as u64) << 32) | (x as u64);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let world = World::new();

        println!(
            "\n=== Analyzing first 50 samples of Pixel ({}, {}) ===",
            x, y
        );

        for sample_idx in 0..50 {
            let mut ray =
                world
                    .camera
                    .get_ray_with_offset(x, HEIGHT - 1 - y, rng.random(), rng.random());

            #[cfg(feature = "brdf_only")]
            let sample_color = crate::rendering::BrdfOnlyStrategy::ray_color(
                &world,
                &mut ray,
                0,
                &mut rng,
                Vector3::one(),
            );

            #[cfg(not(feature = "brdf_only"))]
            let sample_color = Vector3::zero();

            let luminance = sample_color.luminance();

            if luminance > 1.0 {
                println!(
                    "Sample #{}: luminance={:.2}, RGB=({:.6}, {:.6}, {:.6})",
                    sample_idx, luminance, sample_color.x, sample_color.y, sample_color.z
                );
            }
        }

        println!("\nCheck test_sample_26.log for BTDF contribution warnings during this test.");
    }

    #[test]
    fn test_pixel_10_158_mis() {
        // MISで問題が報告されたピクセル(10, 158)を調査
        let x = 10;
        let y = 158;

        let seed = ((y as u64) << 32) | (x as u64);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let world = World::new();

        let mut color_temp = Vector3::new(0.0, 0.0, 0.0);
        let mut max_sample = Vector3::zero();
        let mut max_luminance = 0.0;
        let mut max_sample_idx = 0;
        let mut high_contribution_count = 0;

        println!("\n=== Testing Pixel ({}, {}) with MIS ===", x, y);

        for sample_idx in 0..SAMPLE_NUM {
            let mut ray =
                world
                    .camera
                    .get_ray_with_offset(x, HEIGHT - 1 - y, rng.random(), rng.random());

            #[cfg(feature = "mis")]
            let sample_color = crate::rendering::MisStrategy::ray_color(
                &world,
                &mut ray,
                0,
                &mut rng,
                Vector3::one(),
            );

            #[cfg(not(feature = "mis"))]
            let sample_color = Vector3::zero();

            let luminance = sample_color.luminance();

            if luminance > 10.0 {
                high_contribution_count += 1;
                if high_contribution_count <= 20 {
                    println!(
                        "Sample #{}: luminance={:.2}, RGB=({:.6}, {:.6}, {:.6})",
                        sample_idx, luminance, sample_color.x, sample_color.y, sample_color.z
                    );
                }
            }

            if luminance > max_luminance {
                max_luminance = luminance;
                max_sample = sample_color;
                max_sample_idx = sample_idx;
            }

            color_temp += sample_color;
        }

        color_temp = color_temp / SAMPLE_NUM as f64;

        println!("\n=== Results ===");
        println!(
            "Average color (before gamma): ({:.6}, {:.6}, {:.6})",
            color_temp.x, color_temp.y, color_temp.z
        );
        println!("Average luminance: {:.6}", color_temp.luminance());
        println!("Max sample index: #{}", max_sample_idx);
        println!("Max sample luminance: {:.2}", max_luminance);
        println!(
            "Max sample RGB: ({:.6}, {:.6}, {:.6})",
            max_sample.x, max_sample.y, max_sample.z
        );
        println!(
            "Max sample contribution to average: ({:.6}, {:.6}, {:.6})",
            max_sample.x / SAMPLE_NUM as f64,
            max_sample.y / SAMPLE_NUM as f64,
            max_sample.z / SAMPLE_NUM as f64
        );
        println!(
            "High contribution samples (>10): {}/{}",
            high_contribution_count, SAMPLE_NUM
        );
    }

    #[test]
    fn test_trace_sample_2369() {
        // Sample #2369のパス全体を詳細トレース
        let x = 10;
        let y = 158;
        let target_sample = 2369;

        crate::log::init_log("trace_sample_2369.log").expect("Failed to create log file");

        let seed = ((y as u64) << 32) | (x as u64);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let world = World::new();

        println!(
            "\n=== Tracing Sample #{} of Pixel ({}, {}) ===",
            target_sample, x, y
        );

        // target_sampleまでスキップ
        for _ in 0..target_sample {
            // get_ray_with_offsetで2回random()を呼ぶ
            let mut ray =
                world
                    .camera
                    .get_ray_with_offset(x, HEIGHT - 1 - y, rng.random(), rng.random());

            // ray_colorを実行（ログは無効）
            #[cfg(feature = "mis")]
            {
                crate::rendering::MisStrategy::ray_color(
                    &world,
                    &mut ray,
                    0,
                    &mut rng,
                    Vector3::one(),
                );
            }
        }

        // target_sampleを詳細トレース
        println!(
            "Executing sample #{} with detailed logging...",
            target_sample
        );

        let mut ray =
            world
                .camera
                .get_ray_with_offset(x, HEIGHT - 1 - y, rng.random(), rng.random());

        #[cfg(feature = "mis")]
        let sample_color =
            crate::rendering::MisStrategy::ray_color(&world, &mut ray, 0, &mut rng, Vector3::one());

        #[cfg(not(feature = "mis"))]
        let sample_color = Vector3::zero();

        println!(
            "\nFinal sample color: ({:.6}, {:.6}, {:.6})",
            sample_color.x, sample_color.y, sample_color.z
        );
        println!("Luminance: {:.2}", sample_color.luminance());
        println!("\nDetailed trace saved to trace_sample_2369.log");
    }
}
