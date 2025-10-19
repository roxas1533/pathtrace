#![forbid(unsafe_code)]

mod camera;
mod math;
mod objects;
mod world;

use pixels::{Error, Pixels, SurfaceTexture};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use world::{HEIGHT, WIDTH, World};

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new().unwrap();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Path Tracer - Cornell Box")
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

    let world_clone = Arc::clone(&world);
    thread::spawn(move || {
        let pixels_coords: Vec<(u32, u32)> = (0..HEIGHT)
            .flat_map(|y| (0..WIDTH).map(move |x| (x, y)))
            .collect();

        // rayonで並列処理
        pixels_coords.par_iter().for_each(|&(x, y)| {
            let mut rng = rand::rng();

            let mut world = world_clone.lock().unwrap();
            let color = world.render_pixel(x, y, &mut rng);
            let index = (y * WIDTH + x) as usize;
            world.data[index] = color;
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
