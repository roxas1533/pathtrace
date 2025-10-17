#![deny(clippy::all)]
#![forbid(unsafe_code)]

mod camera;
mod math;
mod objects;

use camera::Camera;
use math::Vector3;
use objects::{Hittable, Mirror, Object, SphereShape};
use pixels::{Error, Pixels, SurfaceTexture};
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use crate::objects::material::LambertianCosineWeighted;

const WIDTH: u32 = 400;
const HEIGHT: u32 = 200;
const SAMPLE_NUM: u32 = 100; // 1ピクセルあたりのサンプル数

#[derive(Clone, Copy)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
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

struct World {
    data: [Color; WIDTH as usize * HEIGHT as usize],
    camera: Camera,
    objects: Vec<Object>,
}

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new().unwrap();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Hello Pixels")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };
    let world = Arc::new(Mutex::new(World::new()));

    // レンダリングスレッド: rayonで並列化してピクセル毎にレンダリング
    let world_clone = Arc::clone(&world);
    thread::spawn(move || {
        // 全ピクセルの座標リストを作成
        let pixels_coords: Vec<(u32, u32)> = (0..HEIGHT)
            .flat_map(|y| (0..WIDTH).map(move |x| (x, y)))
            .collect();

        // rayonで並列処理
        pixels_coords.par_iter().for_each(|&(x, y)| {
            // 各スレッドでローカルなrngを使用
            let mut rng = rand::rng();

            // ロックを取得してピクセルを計算・書き込み
            let mut world = world_clone.lock().unwrap();
            let color = world.render_pixel(x, y, &mut rng);
            let index = (y * WIDTH + x) as usize;
            world.data[index] = color;
            // ロック解放（スコープ終了）→ 描画スレッドが読み取れる
        });
    });

    window.request_redraw();

    let res = event_loop.run(|event, elwt| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            elwt.exit();
        }
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
            if let Ok(world) = world.lock() {
                world.draw(pixels.frame_mut());
            }
            if let Err(err) = pixels.render() {
                eprintln!("pixels.render() failed: {err}");
                elwt.exit();
            }
            thread::sleep(Duration::from_millis(16));
            window.request_redraw();
        }
        _ => {}
    });
    res.map_err(|e| Error::UserDefined(Box::new(e)))
}

impl World {
    fn new() -> Self {
        // カメラを設定
        let camera = Camera::new(
            Vector3::new(0.0, 0.0, 2.0), // カメラ位置
            WIDTH,
            HEIGHT,
            1.0,  // スクリーンとの距離
            35.0, // 視野角（度）
        );

        // シーンにオブジェクトを追加
        let objects: Vec<Object> = vec![
            Object::new(
                Box::new(SphereShape::new(Vector3::new(-1.0, 0.0, -1.0), 0.5)),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.1, 0.1, 0.1))), // 暗い球
            ),
            Object::new(
                Box::new(SphereShape::new(Vector3::new(0.0, 0.0, -1.0), 0.5)),
                Box::new(Mirror {
                    roughness: 0.01,
                    color: Vector3::new(0.9, 0.1, 0.1),
                }), // 鏡面反射する球
            ),
            Object::new(
                Box::new(SphereShape::new(Vector3::new(1.0, 0.0, -1.0), 0.5)),
                Box::new(Mirror {
                    roughness: 0.31,
                    color: Vector3::new(0.9, 0.9, 0.9),
                }), // 鏡面反射する球
            ),
            Object::new(
                Box::new(SphereShape::new(Vector3::new(0.0, -100.5, 0.0), 100.0)),
                Box::new(LambertianCosineWeighted::new(Vector3::new(0.8, 0.8, 0.0))), // 黄色っぽい地面
            ),
        ];

        Self {
            data: [Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }; WIDTH as usize * HEIGHT as usize],
            camera,
            objects,
        }
    }

    // シーン内のオブジェクトとの交差判定（ヒット情報とオブジェクトのペアを返す）
    fn hit_scene(
        &self,
        ray: &camera::Ray,
        t_min: f64,
        t_max: f64,
    ) -> Option<(objects::HitRecord, &Object)> {
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

    // レイの色を計算（再帰的パストレーシング）
    fn ray_color(&self, ray: &camera::Ray, depth: u32, rng: &mut impl Rng) -> Vector3 {
        // 最大再帰深度に達したら黒を返す
        if depth == 0 {
            return Vector3::zero();
        }

        // オブジェクトとの交差をチェック（自己交差を避けるため0.001から）
        if let Some((hit, obj)) = self.hit_scene(ray, 0.001, f64::INFINITY) {
            // 入射方向（レイの方向）
            let incoming = ray.direction;

            // オブジェクトのサンプリング戦略を使って方向を生成
            let scattered_direction = obj.sample_direction(&hit.normal, &incoming, rng);
            let scattered_ray = camera::Ray::new(hit.point, scattered_direction);

            // コサイン項（入射角度による減衰）
            let cos_theta = scattered_direction.dot(&hit.normal).max(0.0);

            // BRDFとPDFを同時に取得（効率的）
            let (brdf, pdf) =
                obj.brdf_pdf(&hit.point, &(-incoming), &scattered_direction, &hit.normal);

            // 入射輝度を再帰的に計算
            let incoming_light = self.ray_color(&scattered_ray, depth - 1, rng);

            // レンダリング方程式: brdf * 入射輝度 * cos(θ) / pdf
            return brdf * incoming_light * cos_theta / pdf;
        }

        // 背景のグラデーション（空の色）
        let dir = ray.direction.normalize();
        let t = 0.5 * (dir.y + 1.0);

        let r = (1.0 - t) * 1.0 + t * 0.5;
        let g = (1.0 - t) * 1.0 + t * 0.7;
        let b = (1.0 - t) * 1.0 + t * 1.0;

        Vector3::new(r, g, b)
    }

    // 1ピクセルをレンダリング（読み取り専用でカメラとオブジェクト情報を使用）
    fn render_pixel(&self, x: u32, y: u32, rng: &mut impl Rng) -> Color {
        let mut color_temp = Vector3::new(0.0, 0.0, 0.0);
        const MAX_DEPTH: u32 = 5; // 最大再帰深度

        for _ in 0..SAMPLE_NUM {
            // ピクセル座標からレイを生成
            // y座標を反転（画面座標系ではy=0が上、カメラ座標系ではy+が上）
            let ray =
                self.camera
                    .get_ray_with_offset(x, HEIGHT - 1 - y, rng.random(), rng.random());

            // レイの色を計算してサンプルごとに加算
            color_temp += self.ray_color(&ray, MAX_DEPTH, rng);
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

    fn draw(&self, frame: &mut [u8]) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let color = self.data[i];
            pixel.copy_from_slice(&[color.r, color.g, color.b, color.a]);
        }
    }
}
