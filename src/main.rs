use winit::event_loop::EventLoop;

mod app;
mod renderer;

pub fn run() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new()?;
    let mut app = app::App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}

fn main() -> anyhow::Result<()> {
    run()
}
