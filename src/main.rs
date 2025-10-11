#![deny(clippy::all)]
#![forbid(unsafe_code)]

use pixels::{Error, Pixels, SurfaceTexture};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

#[derive(Clone, Copy)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

struct World {
    data: [Color; WIDTH as usize * HEIGHT as usize],
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

    // ワーカースレッドを起動してピクセルを更新
    let world_clone = Arc::clone(&world);
    thread::spawn(move || {
        let mut frame_count = 0u32;
        loop {
            if let Ok(mut world) = world_clone.lock() {
                // 例: フレームごとに色を変化させる
                for (i, color) in world.data.iter_mut().enumerate() {
                    let x = (i % WIDTH as usize) as u32;
                    let y = (i / WIDTH as usize) as u32;
                    color.r = ((x + frame_count) % 256) as u8;
                    color.g = ((y + frame_count) % 256) as u8;
                    color.b = ((frame_count) % 256) as u8;
                    color.a = 255;
                }
                frame_count = frame_count.wrapping_add(1);
            }
            thread::sleep(Duration::from_millis(16)); // 約60fps
        }
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
            window.request_redraw();
        }
        _ => {}
    });
    res.map_err(|e| Error::UserDefined(Box::new(e)))
}

impl World {
    fn new() -> Self {
        Self {
            data: [Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            }; WIDTH as usize * HEIGHT as usize],
        }
    }

    fn draw(&self, frame: &mut [u8]) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let color = self.data[i];
            pixel.copy_from_slice(&[color.r, color.g, color.b, color.a]);
        }
    }
}
