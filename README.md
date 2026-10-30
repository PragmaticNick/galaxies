# galaxies

A 10 million star galaxy simulated in real time with the particle mesh method on the GPU
(Rust + wgpu).

## Run

```
cargo run --release
```

Needs a GPU with Vulkan, DirectX 12 or Metal. The window opens fullscreen.

## Controls

- drag with the left mouse button / `W` `A` `S` `D`: move the camera
- mouse wheel / arrow up and down: zoom
- `Esc`: quit

## Settings

Everything is in `src/config.rs`: the galaxies, star count, grid size,
gravity softening, time scale, colors and the look of the renderer.
