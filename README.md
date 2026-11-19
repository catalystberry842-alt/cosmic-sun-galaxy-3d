# Cosmic Sun & Spiral Galaxy — Interactive 3D

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fcatalystberry842-alt%2Fcosmic-sun-galaxy-3d)

An interactive real-time 3D **Cosmic Sun & Spiral Galaxy** simulation featuring a blazing procedural solar photosphere, volumetric corona, twisting magnetic loops, erupting flare jets, 70,000 differentially rotating spiral galaxy stars, volumetric nebulae, dark dust lanes, and 4-octave HDR ACES Filmic bloom.

## Files

- `index.html` — Zero-dependency, self-contained **WebGL2** interactive 3D web bundle (ready for Vercel static hosting).
- `vercel.json` — Vercel static deployment configuration.
- `cosmic_sun_galaxy.py` — Standalone Python OpenGL 3.3+ (`pygame`, `moderngl`, `numpy`, `pillow`) desktop renderer.

## Interactive WebGL2 Controls

- **Move Mouse:** 3D hover parallax + real-time gravity swirl / repel wave / solar magnet force field
- **Click / Tap:** Trigger expanding 3D cosmic shockwave pulse
- **Left-Drag:** Inertial 3D orbit
- **Right-Drag / Shift-Drag:** Pan camera target
- **Scroll / Pinch:** Smooth deep zoom
- **Keys:** `1`-`4` camera presets, `M` cycle cursor force mode, `C` auto-orbit, `Space` pause/play, `R` reset, `S` PNG screenshot, `H` hide/show HUD, `F` fullscreen
