use winit::event_loop::EventLoop;

mod app;
mod export;
pub mod galaxy;
mod physics;
mod renderer;
pub mod star;

use export::ExportConfig;
use galaxy::{GalaxyConfig, generate_galaxy};
use physics::PhysicsStrategy;


pub fn run() -> anyhow::Result<()> {
    env_logger::init();

    if let Some(mut cfg) = parse_export_args()? {
        let config = GalaxyConfig {
            center: [0.0, 0.0],
            radius: 300.0,
            star_count: 6000,
            core_mass: 50000.0,
            core_radius: 6.0,
            star_mass: 0.1,
            star_radius: 2.0,
            gap: 20.0,
        };

        cfg.strategy = PhysicsStrategy::FmmSerial;

        let stars = generate_galaxy(&config);
        return export::export_video(&cfg, stars);
    }

    let event_loop = EventLoop::new()?;
    let mut app = app::App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}

fn parse_export_args() -> anyhow::Result<Option<ExportConfig>> {
    let args: Vec<String> = std::env::args().collect();
    let Some(output) = find_flag_value(&args, "--export") else {
        return Ok(None);
    };
    Ok(Some(ExportConfig {
        width: find_flag_value(&args, "--width")
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(1920),
        height: find_flag_value(&args, "--height")
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(1080),
        fps: find_flag_value(&args, "--fps")
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(60),
        seconds: find_flag_value(&args, "--seconds")
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(10.0),
        strategy: find_flag_value(&args, "--strategy")
            .map(|v| parse_strategy(&v))
            .transpose()?
            .unwrap_or(PhysicsStrategy::FmmSerial),
        output,
    }))
}

fn parse_strategy(value: &str) -> anyhow::Result<PhysicsStrategy> {
    match value {
        "direct" => Ok(PhysicsStrategy::Direct),
        "rayon" => Ok(PhysicsStrategy::Rayon),
        "fmm-serial" => Ok(PhysicsStrategy::FmmSerial),
        "fmm-rayon" => Ok(PhysicsStrategy::FmmRayon),
        other => anyhow::bail!(
            "unknown --strategy '{other}' (expected: direct, rayon, fmm-serial, fmm-rayon)"
        ),
    }
}

fn find_flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() -> anyhow::Result<()> {
    run()
}
