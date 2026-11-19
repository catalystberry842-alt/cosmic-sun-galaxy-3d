#!/usr/bin/env python3
"""
Cosmic Sun & Spiral Galaxy — Real-Time OpenGL 3.3+ Volumetric 3D Scene
======================================================================
A self-contained, production-grade Python 3D graphics engine powered by
pygame, ModernGL, and NumPy that renders a colossal blazing 3D Sun anchored
in the lower foreground beneath a sprawling, differentially rotating 3D
spiral galaxy in a vertical 9:16 cinematic portrait composition.

Requirements:
    pip install pygame moderngl numpy pillow

Controls:
    Left Mouse Drag   : Orbit camera (auto-switches to manual mode)
    Right Mouse Drag  : Pan camera target vertically / horizontally
    Mouse Scroll      : Smooth zoom in / out
    SPACE             : Pause / Resume time animation
    C                 : Toggle Cinematic Autopilot vs. Manual Camera
    R                 : Reset camera to Hero 9:16 composition
    1 / 2 / 3         : Camera framing presets (Hero / Solar Close-Up / Galactic Vista)
    S or F12          : Capture high-resolution PNG screenshot
    B                 : Cycle bloom intensity (Normal / High / Subtle)
    ESC or Q          : Exit

CLI Options:
    python cosmic_sun_galaxy.py                # Default 720x1280 9:16 window
    python cosmic_sun_galaxy.py --hd           # Full HD 1080x1920 9:16 window
    python cosmic_sun_galaxy.py --stars 80000  # Customize galaxy star count
    python cosmic_sun_galaxy.py --screenshot out.png --time 6.5  # Render & save frame
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import moderngl
import numpy as np
import pygame

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


# =============================================================================
# GLOBAL SCENE & SIMULATION CONFIGURATION
# =============================================================================

DEFAULT_WIDTH = 720
DEFAULT_HEIGHT = 1280
HD_WIDTH = 1080
HD_HEIGHT = 1920

# World-space layout tailored for 9:16 vertical framing
SUN_CENTER = np.array([0.0, -5.85, 0.0], dtype=np.float32)
SUN_RADIUS = 3.25

# Galaxy sits high and deep behind the sun, sweeping dramatically across the upper 2/3 sky
GALAXY_CENTER = np.array([-0.85, 11.2, -40.0], dtype=np.float32)
GALAXY_PITCH_DEG = -58.0   # Tilted toward viewer so logarithmic spiral arms fan out grandly
GALAXY_YAW_DEG = 8.0
GALAXY_ROLL_DEG = 24.0     # Diagonal cosmic sweep across the 9:16 portrait frame
GALAXY_RADIUS = 24.5

# Geometry / particle budgets
SUN_ICOSPHERE_SUBDIVISIONS = 5      # 20,480 triangles, 10,242 vertices (seamless, no polar pinch)
CORONA_SHELL_SUBDIVISIONS = 4       # 5,120 triangles per volumetric shell
CORONAL_LOOP_COUNT = 34             # 3D twisting magnetic flux spline tubes
CORONAL_LOOP_LENGTH_SEGS = 48
CORONAL_LOOP_RADIAL_SEGS = 10
SOLAR_FLARE_PARTICLE_COUNT = 9500   # Curved magnetic-field-line plasma jets
GALAXY_STAR_COUNT = 65000           # 50,000+ differential Keplerian spiral galaxy stars
NEBULA_CLOUD_COUNT = 2400           # Multi-colored 3D volumetric nebula billows
DUST_LANE_COUNT = 1350              # Dark silhouetted interstellar dust volumes
DEEP_STAR_COUNT = 9000              # Far deep-field background stars & clusters
DISTANT_GALAXY_COUNT = 64           # Tiny distant procedural spiral & elliptical galaxies


# =============================================================================
# 3D LINEAR ALGEBRA & MATRIX UTILITIES (COLUMN-MAJOR FOR OPENGL)
# =============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Return unit-length vector safely."""
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / norm).astype(np.float32)


def mat4_identity() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def mat4_perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Standard OpenGL right-handed perspective projection matrix."""
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-5)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def mat4_look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed look-at view matrix."""
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    if np.linalg.norm(s) < 1e-6:
        s = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


def mat4_translate(t: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = float(t[0])
    m[1, 3] = float(t[1])
    m[2, 3] = float(t[2])
    return m


def mat4_scale(sx: float, sy: float, sz: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def mat4_rotate_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = np.eye(4, dtype=np.float32)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def mat4_rotate_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def mat4_rotate_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def mat4_to_bytes(m: np.ndarray) -> bytes:
    """Convert row-major numpy 4x4 matrix to column-major OpenGL uniform bytes."""
    return np.ascontiguousarray(m.T, dtype=np.float32).tobytes()


def build_galaxy_model_matrix() -> np.ndarray:
    """Construct the world transform placing the spiral galaxy above and behind the sun."""
    t = mat4_translate(GALAXY_CENTER)
    rz = mat4_rotate_z(GALAXY_ROLL_DEG)
    ry = mat4_rotate_y(GALAXY_YAW_DEG)
    rx = mat4_rotate_x(GALAXY_PITCH_DEG)
    return (t @ rz @ ry @ rx).astype(np.float32)


# =============================================================================
# CINEMATIC CAMERA CONTROLLER WITH INTERACTIVE ORBIT / PAN / ZOOM
# =============================================================================

@dataclass
class CameraRig:
    """
    Hybrid cinematic autopilot + smooth damped interactive orbit camera.
    Keeps the blazing Sun anchored in the lower third and the sprawling
    spiral galaxy soaring overhead in 9:16 portrait composition.
    """
    cinematic_mode: bool = True
    fov_y: float = 52.0
    near: float = 0.25
    far: float = 600.0

    # Spherical orbit coordinates around target
    yaw: float = 0.0
    pitch: float = 0.06
    distance: float = 19.8
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.55, -2.5], dtype=np.float32))

    # Desired smoothed targets (for interactive damping)
    goal_yaw: float = 0.0
    goal_pitch: float = 0.06
    goal_distance: float = 19.8
    goal_target: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.55, -2.5], dtype=np.float32))

    # Cinematic internal clock
    cine_time: float = 0.0
    roll: float = 0.0
    preset_id: int = 1

    # Computed per frame
    eye: np.ndarray = field(default_factory=lambda: np.array([0.0, 2.0, 18.0], dtype=np.float32))
    right: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
    forward: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=np.float32))

    def apply_preset(self, preset: int) -> None:
        self.preset_id = preset
        if preset == 1:
            # Hero 9:16 Composition: colossal sun lower-center, galaxy sweeping above
            self.goal_yaw = 0.0
            self.goal_pitch = 0.06
            self.goal_distance = 19.8
            self.goal_target = np.array([0.0, 1.55, -2.5], dtype=np.float32)
        elif preset == 2:
            # Dramatic Solar Horizon Close-Up: grazing coronal loops with galaxy rising behind
            self.goal_yaw = 0.32
            self.goal_pitch = -0.14
            self.goal_distance = 12.2
            self.goal_target = np.array([0.0, -1.9, -1.2], dtype=np.float32)
        elif preset == 3:
            # High Cosmic Overlook: broad sweep across spiral arms and solar system anchor
            self.goal_yaw = -0.45
            self.goal_pitch = 0.24
            self.goal_distance = 27.5
            self.goal_target = np.array([0.4, 3.8, -6.0], dtype=np.float32)

    def reset(self) -> None:
        self.apply_preset(1)
        self.cinematic_mode = True
        self.cine_time = 0.0

    def drag_orbit(self, dx: float, dy: float) -> None:
        if self.cinematic_mode:
            self.cinematic_mode = False
            self.goal_yaw = self.yaw
            self.goal_pitch = self.pitch
            self.goal_distance = self.distance
            self.goal_target = self.target.copy()
        self.goal_yaw -= dx * 0.0052
        self.goal_pitch = max(-1.38, min(1.38, self.goal_pitch + dy * 0.0052))

    def drag_pan(self, dx: float, dy: float) -> None:
        if self.cinematic_mode:
            self.cinematic_mode = False
            self.goal_yaw = self.yaw
            self.goal_pitch = self.pitch
            self.goal_distance = self.distance
            self.goal_target = self.target.copy()
        scale = self.distance * 0.0014
        self.goal_target = self.goal_target - self.right * (dx * scale) + self.up * (dy * scale)

    def zoom(self, wheel_y: float) -> None:
        if self.cinematic_mode:
            self.cinematic_mode = False
            self.goal_yaw = self.yaw
            self.goal_pitch = self.pitch
            self.goal_distance = self.distance
            self.goal_target = self.target.copy()
        factor = math.pow(0.89, wheel_y)
        self.goal_distance = max(6.8, min(68.0, self.goal_distance * factor))

    def update(self, dt: float, paused: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Advance camera choreography or damped manual orbit and return (view, eye)."""
        if not paused and self.cinematic_mode:
            self.cine_time += dt

        if self.cinematic_mode:
            t = self.cine_time
            # Slow, graceful orbital glide around the sun-galaxy axis with figure-8 vertical drift
            base_yaw = self.goal_yaw + 0.42 * math.sin(t * 0.145) + 0.11 * math.sin(t * 0.31)
            base_pitch = self.goal_pitch + 0.085 * math.sin(t * 0.21 + 0.7) + 0.03 * math.cos(t * 0.09)
            base_dist = self.goal_distance + 2.35 * math.sin(t * 0.12 - 0.4) + 0.65 * math.cos(t * 0.27)
            drift_target = self.goal_target + np.array([
                0.75 * math.sin(t * 0.11),
                0.85 * math.sin(t * 0.17 + 0.5),
                0.60 * math.cos(t * 0.13),
            ], dtype=np.float32)

            alpha = 1.0 - math.exp(-3.8 * max(dt, 0.001))
            self.yaw += (base_yaw - self.yaw) * alpha
            self.pitch += (base_pitch - self.pitch) * alpha
            self.distance += (base_dist - self.distance) * alpha
            self.target = self.target + (drift_target - self.target) * alpha
            self.roll = 0.048 * math.sin(t * 0.135 + 0.3)
        else:
            alpha = 1.0 - math.exp(-9.5 * max(dt, 0.001))
            self.yaw += (self.goal_yaw - self.yaw) * alpha
            self.pitch += (self.goal_pitch - self.pitch) * alpha
            self.distance += (self.goal_distance - self.distance) * alpha
            self.target = self.target + (self.goal_target - self.target) * alpha
            self.roll *= math.exp(-5.0 * max(dt, 0.001))

        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        offset = np.array([
            self.distance * cp * sy,
            self.distance * sp,
            self.distance * cp * cy,
        ], dtype=np.float32)
        self.eye = (self.target + offset).astype(np.float32)

        world_up = np.array([math.sin(self.roll), math.cos(self.roll), 0.0], dtype=np.float32)
        view = mat4_look_at(self.eye, self.target, world_up)

        # Extract basis vectors from view matrix for billboard alignment
        self.right = view[0, 0:3].copy()
        self.up = view[1, 0:3].copy()
        self.forward = (-view[2, 0:3]).copy()
        return view, self.eye

    def set_cinematic_time(self, t: float) -> None:
        """Jump immediately to a specific cinematic timestamp (for deterministic renders)."""
        self.cinematic_mode = True
        self.cine_time = t
        self.yaw = self.goal_yaw + 0.42 * math.sin(t * 0.145) + 0.11 * math.sin(t * 0.31)
        self.pitch = self.goal_pitch + 0.085 * math.sin(t * 0.21 + 0.7) + 0.03 * math.cos(t * 0.09)
        self.distance = self.goal_distance + 2.35 * math.sin(t * 0.12 - 0.4) + 0.65 * math.cos(t * 0.27)
        self.target = self.goal_target + np.array([
            0.75 * math.sin(t * 0.11),
            0.85 * math.sin(t * 0.17 + 0.5),
            0.60 * math.cos(t * 0.13),
        ], dtype=np.float32)
        self.roll = 0.048 * math.sin(t * 0.135 + 0.3)


# =============================================================================
# SHARED GLSL 330 CORE NOISE, FBM & BLACKBODY SHADER LIBRARY
# =============================================================================

GLSL_NOISE_LIB = r"""
// 3D Simplex Noise (Stefan Gustavson / Ashima Arts, MIT-style public domain math)
vec4 _mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 _mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 _permute(vec4 x) { return _mod289(((x * 34.0) + 10.0) * x); }
vec4 _taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);

    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;

    i = _mod289(i);
    vec4 p = _permute(_permute(_permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0))
              + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);

    vec4 norm = _taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    vec4 m = max(0.5 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 105.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

// Non-axis-aligned octave rotation matrix to eliminate grid artifacts in fBm
const mat3 OCTAVE_ROT = mat3(
     0.00,  0.80,  0.60,
    -0.80,  0.36, -0.48,
    -0.60, -0.48,  0.64
);

float fbm5(vec3 p) {
    float v = 0.0;
    float a = 0.52;
    for (int i = 0; i < 5; ++i) {
        v += a * snoise(p);
        p = OCTAVE_ROT * p * 2.04 + vec3(13.7, -9.2, 5.4);
        a *= 0.50;
    }
    return v;
}

float fbm4(vec3 p) {
    float v = 0.0;
    float a = 0.54;
    for (int i = 0; i < 4; ++i) {
        v += a * snoise(p);
        p = OCTAVE_ROT * p * 2.03 + vec3(7.1, 11.3, -4.9);
        a *= 0.50;
    }
    return v;
}

float fbm3(vec3 p) {
    float v = 0.0;
    float a = 0.56;
    for (int i = 0; i < 3; ++i) {
        v += a * snoise(p);
        p = OCTAVE_ROT * p * 2.07 + vec3(-5.3, 8.7, 3.1);
        a *= 0.48;
    }
    return v;
}

// Ridged multifractal fBm: creates crisp cellular plasma filaments & granulation boundaries
float ridgedFbm4(vec3 p) {
    float sum = 0.0;
    float amp = 0.55;
    float weight = 1.0;
    for (int i = 0; i < 4; ++i) {
        float n = 1.0 - abs(snoise(p));
        n = n * n * weight;
        weight = clamp(n * 1.8, 0.0, 1.0);
        sum += n * amp;
        p = OCTAVE_ROT * p * 2.12 + vec3(4.2, -7.8, 9.1);
        amp *= 0.50;
    }
    return sum;
}

// 3D Cellular Voronoi distance for crisp solar convective granulation cells
vec3 _hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

// Returns vec2(minDistToCenter, edgeDist) for solar plasma granulation cells
vec2 voronoiGranules(vec3 x, float jitterTime) {
    vec3 n = floor(x);
    vec3 f = fract(x);
    float d1 = 8.0;
    float d2 = 8.0;
    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                vec3 g = vec3(float(i), float(j), float(k));
                vec3 rnd = _hash33(n + g);
                vec3 o = 0.5 + 0.42 * sin(jitterTime + 6.28318 * rnd);
                vec3 r = g + o - f;
                float d = dot(r, r);
                if (d < d1) {
                    d2 = d1;
                    d1 = d;
                } else if (d < d2) {
                    d2 = d;
                }
            }
        }
    }
    return vec2(sqrt(d1), sqrt(d2) - sqrt(d1));
}

// Physically inspired solar plasma temperature-to-HDR-RGB ramp (0=umbra -> 1.4=white-hot faculae)
vec3 solarBlackbody(float t) {
    t = clamp(t, 0.0, 1.45);
    vec3 c0 = vec3(0.015, 0.002, 0.0005); // Deep velvety magnetic sunspot umbra
    vec3 c1 = vec3(0.32,  0.022, 0.002);  // Dark crimson intergranular lanes & penumbra
    vec3 c2 = vec3(0.95,  0.16,  0.012);  // Molten volcanic orange plasma
    vec3 c3 = vec3(1.72,  0.58,  0.065);  // Rich golden photosphere granules
    vec3 c4 = vec3(2.45,  1.45,  0.42);   // Blazing yellow-white convective peaks
    vec3 c5 = vec3(4.60,  3.75,  2.35);   // Searing HDR faculae & flare plasma

    if (t < 0.22) return mix(c0, c1, smoothstep(0.0, 0.22, t));
    if (t < 0.48) return mix(c1, c2, smoothstep(0.22, 0.48, t));
    if (t < 0.74) return mix(c2, c3, smoothstep(0.48, 0.74, t));
    if (t < 0.98) return mix(c3, c4, smoothstep(0.74, 0.98, t));
    return mix(c4, c5, smoothstep(0.98, 1.42, t));
}
"""


# =============================================================================
# GLSL 330 SHADERS — PART A: HYPER-DETAILED 3D SUN, CORONA, LOOPS & FLARES
# =============================================================================

SUN_VERT_SHADER = (
    "#version 330 core\n"
    + GLSL_NOISE_LIB
    + r"""
in vec3 in_position;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_sun_center;
uniform float u_sun_radius;
uniform float u_time;

out vec3 v_sphere_dir;
out vec3 v_world_pos;
out vec3 v_world_normal;
out float v_disp_height;

// Differential solar rotation: equatorial plasma rotates faster than polar regions
vec3 applySolarRotation(vec3 dir, float t) {
    float lat = asin(clamp(dir.y, -1.0, 1.0));
    float omega = 0.055 + 0.032 * cos(lat) * cos(lat);
    float ang = t * omega;
    float c = cos(ang);
    float s = sin(ang);
    return vec3(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
}

float evalDisplacement(vec3 dir, float t) {
    vec3 rdir = applySolarRotation(dir, t);
    vec3 flow = vec3(t * 0.14, -t * 0.09, t * 0.11);
    float superGran = fbm3(rdir * 3.8 + flow * 0.45);
    float boil = snoise(rdir * 10.5 - flow * 1.15);
    float ridge = 1.0 - abs(snoise(rdir * 6.5 + vec3(0.0, t * 0.25, -t * 0.18)));
    return (superGran * 0.022 + boil * 0.0085 + ridge * 0.011);
}

void main() {
    vec3 dir = normalize(in_position);
    float disp = evalDisplacement(dir, u_time);

    // Finite-difference tangent basis to perturb vertex normal with boiling plasma waves
    vec3 upRef = abs(dir.y) < 0.95 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(upRef, dir));
    vec3 B = cross(dir, T);
    float eps = 0.022;
    float dT = evalDisplacement(normalize(dir + T * eps), u_time);
    float dB = evalDisplacement(normalize(dir + B * eps), u_time);
    vec3 perturbedN = normalize(dir - T * ((dT - disp) / eps) * 0.42 - B * ((dB - disp) / eps) * 0.42);

    float displacedRadius = u_sun_radius * (1.0 + disp);
    vec3 worldPos = u_sun_center + dir * displacedRadius;

    v_sphere_dir = dir;
    v_world_pos = worldPos;
    v_world_normal = perturbedN;
    v_disp_height = disp;

    gl_Position = u_proj * u_view * vec4(worldPos, 1.0);
}
"""
)

SUN_FRAG_SHADER = (
    "#version 330 core\n"
    + GLSL_NOISE_LIB
    + r"""
in vec3 v_sphere_dir;
in vec3 v_world_pos;
in vec3 v_world_normal;
in float v_disp_height;

uniform vec3 u_cam_pos;
uniform float u_time;

out vec4 fragColor;

vec3 applySolarRotation(vec3 dir, float t) {
    float lat = asin(clamp(dir.y, -1.0, 1.0));
    float omega = 0.055 + 0.032 * cos(lat) * cos(lat);
    float ang = t * omega;
    float c = cos(ang);
    float s = sin(ang);
    return vec3(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
}

void main() {
    vec3 N = normalize(v_world_normal);
    vec3 V = normalize(u_cam_pos - v_world_pos);
    float mu = clamp(dot(N, V), 0.0, 1.0); // Limb angle cosine

    vec3 dir = normalize(v_sphere_dir);
    vec3 p = applySolarRotation(dir, u_time);
    float lat = abs(dir.y);

    // 1. Domain-warped convective flow field for boiling supergranulation
    vec3 drift1 = vec3(u_time * 0.065, -u_time * 0.042, u_time * 0.051);
    vec3 warp = vec3(
        fbm3(p * 3.1 + drift1),
        fbm3(p * 3.1 + vec3(5.2, -3.7, 2.4) - drift1 * 0.8),
        fbm3(p * 3.1 + vec3(-4.1, 6.3, -2.9) + drift1 * 0.65)
    );
    vec3 warpedP = p + warp * 0.24;

    // 2. Multi-scale boiling plasma granulation (Voronoi cells + ridged fBm + fine turbulence)
    vec2 vor = voronoiGranules(warpedP * 16.5 + drift1 * 1.5, u_time * 0.95);
    float granuleCore = 1.0 - smoothstep(0.06, 0.64, vor.x);  // Bright rising hot cell centers
    float intergranularLane = smoothstep(0.02, 0.19, vor.y);  // Dark sinking convective cell borders
    float microGranules = fbm4(warpedP * 26.0 - vec3(u_time * 0.22, u_time * 0.15, -u_time * 0.18));
    float superGran = fbm5(warpedP * 4.8 + drift1 * 0.6);
    float plasmaRidges = ridgedFbm4(warpedP * 8.4 + vec3(0.0, u_time * 0.14, 0.0));

    // Base photosphere temperature calibrated for deep volcanic contrast [0.30 .. 1.02]
    float cellContrast = mix(-0.18, 0.24, granuleCore * intergranularLane);
    float temp = 0.56
               + 0.18 * superGran
               + cellContrast
               + 0.13 * (plasmaRidges - 0.38)
               + 0.09 * microGranules
               + v_disp_height * 3.8;

    // 3. Magnetic Sunspot Active Belts (concentrated near +/- 10..42 deg solar latitude)
    float activeBelt = smoothstep(0.02, 0.20, lat) * (1.0 - smoothstep(0.58, 0.85, lat));
    float magField = fbm4(p * 3.25 + warp * 0.36 + vec3(u_time * 0.015, 0.0, -u_time * 0.012));
    float spotSignal = (-magField) * (0.62 + 0.78 * activeBelt);

    float penumbraMask = smoothstep(0.24, 0.41, spotSignal);
    float umbraMask = smoothstep(0.42, 0.56, spotSignal);

    // Striated radial penumbral fibrils around sunspot edges
    float penumbraFibrils = ridgedFbm4(p * 28.0 + warp * 1.25 + vec3(u_time * 0.12));
    float penumbraCool = penumbraMask * (0.34 + 0.20 * (1.0 - penumbraFibrils));
    temp -= penumbraCool;
    temp = mix(temp, 0.025 + 0.03 * max(microGranules, 0.0), umbraMask * 0.96);

    // 4. Incandescent Magnetic Faculae Networks (web-like searing bright ridges near active regions)
    float faculaeZone = smoothstep(0.05, 0.28, spotSignal) * (1.0 - smoothstep(0.32, 0.48, spotSignal));
    float faculaeWeb = pow(clamp(ridgedFbm4(warpedP * 13.5 - drift1 * 0.9), 0.0, 1.0), 2.4);
    // Faculae appear dramatically brighter closer to the solar limb (real solar physics!)
    float limbFaculaeBoost = mix(1.0, 2.05, pow(1.0 - mu, 1.55));
    float faculaeBoost = (faculaeZone * 0.78 + 0.18 * activeBelt) * faculaeWeb * limbFaculaeBoost;
    temp += faculaeBoost;

    // 5. Evaluate HDR blackbody emission color
    vec3 color = solarBlackbody(temp);

    // Add intense white-gold HDR pop selectively on the hottest faculae filaments
    color += vec3(2.8, 2.1, 1.15) * pow(clamp(faculaeBoost, 0.0, 1.0), 1.85);

    // 6. Physically Inspired Wavelength-Dependent Eddington Limb Darkening
    // Blue cools fastest toward the limb, green moderately, red retains fiery crimson depth
    vec3 limbDark;
    limbDark.r = mix(0.22, 1.0, pow(mu, 0.52));
    limbDark.g = mix(0.05, 1.0, pow(mu, 0.88));
    limbDark.b = mix(0.01, 1.0, pow(mu, 1.35));
    color *= limbDark;

    // 7. Fiery Chromospheric Spicule Rim Grazing the Limb
    float fresnel = pow(1.0 - mu, 3.6);
    float spiculeNoise = 0.60 + 0.65 * ridgedFbm4(warpedP * 18.0 + vec3(u_time * 0.35));
    vec3 chromoRim = vec3(2.45, 0.58, 0.08) * fresnel * spiculeNoise * 1.10;
    float edgeSpark = pow(1.0 - mu, 6.8) * (0.65 + 0.55 * plasmaRidges);
    chromoRim += vec3(3.6, 1.55, 0.38) * edgeSpark * 0.72;

    color += chromoRim;

    fragColor = vec4(max(color, vec3(0.0)), 1.0);
}
"""
)

CORONA_SHELL_VERT_SHADER = (
    "#version 330 core\n"
    + GLSL_NOISE_LIB
    + r"""
in vec3 in_position;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_sun_center;
uniform float u_shell_radius;
uniform float u_layer_id;
uniform float u_time;

out vec3 v_dir;
out vec3 v_world_pos;
out vec3 v_normal;

void main() {
    vec3 dir = normalize(in_position);
    float pulse = fbm3(dir * (2.6 + u_layer_id * 0.9) + vec3(u_time * 0.16, -u_time * 0.12, u_time * 0.10));
    float r = u_shell_radius * (1.0 + pulse * (0.045 + 0.035 * u_layer_id));
    vec3 worldPos = u_sun_center + dir * r;
    v_dir = dir;
    v_world_pos = worldPos;
    v_normal = dir;
    gl_Position = u_proj * u_view * vec4(worldPos, 1.0);
}
"""
)

CORONA_SHELL_FRAG_SHADER = (
    "#version 330 core\n"
    + GLSL_NOISE_LIB
    + r"""
in vec3 v_dir;
in vec3 v_world_pos;
in vec3 v_normal;

uniform vec3 u_cam_pos;
uniform float u_layer_id;
uniform float u_time;

out vec4 fragColor;

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_cam_pos - v_world_pos);
    float ndotv = abs(dot(N, V));

    // Shell glows strongest grazing the silhouette limb, fades smoothly at center and outer rim
    float rim = pow(1.0 - ndotv, 2.1 + u_layer_id * 0.65) * smoothstep(0.0, 0.24, ndotv);

    vec3 p = normalize(v_dir);
    float ang = u_time * (0.05 + 0.015 * u_layer_id);
    mat2 rot = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
    p.xz = rot * p.xz;

    vec3 flow = vec3(u_time * 0.18, -u_time * 0.11, u_time * 0.14);
    float wisps = ridgedFbm4(p * (4.2 + u_layer_id * 1.4) + flow);
    float fine = fbm3(p * 9.5 - flow * 1.6) * 0.5 + 0.5;
    float intensity = rim * (0.55 + 0.85 * wisps) * (0.65 + 0.55 * fine);

    vec3 innerCol = vec3(2.2, 0.78, 0.14);
    vec3 outerCol = vec3(1.15, 0.22, 0.03);
    vec3 col = mix(innerCol, outerCol, clamp(u_layer_id * 0.45 + (1.0 - wisps) * 0.35, 0.0, 1.0));

    float layerGain = mix(0.46, 0.20, clamp(u_layer_id * 0.45, 0.0, 1.0));
    fragColor = vec4(col * intensity * layerGain, 1.0);
}
"""
)

CORONA_VOL_VERT_SHADER = r"""
#version 330 core
in vec2 in_quad;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_sun_center;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_billboard_radius;

out vec2 v_uv;
out vec3 v_world_pos;

void main() {
    v_uv = in_quad; // [-1, 1]
    vec3 worldPos = u_sun_center
                  + u_cam_right * (in_quad.x * u_billboard_radius)
                  + u_cam_up    * (in_quad.y * u_billboard_radius);
    v_world_pos = worldPos;
    gl_Position = u_proj * u_view * vec4(worldPos, 1.0);
}
"""

CORONA_VOL_FRAG_SHADER = (
    "#version 330 core\n"
    + GLSL_NOISE_LIB
    + r"""
in vec2 v_uv;
in vec3 v_world_pos;

uniform vec3 u_sun_center;
uniform vec3 u_cam_pos;
uniform float u_sun_radius;
uniform float u_billboard_radius;
uniform float u_time;

out vec4 fragColor;

void main() {
    float rNorm = length(v_uv); // 0 at center, 1 at billboard edge
    if (rNorm >= 1.0) discard;

    // Exact ray-sphere closest approach for physically seamless 3D volumetric glow around sun
    vec3 ro = u_cam_pos;
    vec3 rd = normalize(v_world_pos - u_cam_pos);
    vec3 oc = u_sun_center - ro;
    float tca = dot(oc, rd);
    float d2 = dot(oc, oc) - tca * tca;
    float b = sqrt(max(d2, 0.0));           // Impact parameter in world units
    float rRatio = b / u_sun_radius;        // 1.0 right at the solar photosphere limb

    // Angular direction in 3D space around the sun center (view-invariant 3D streamer structure)
    vec3 closestPt = ro + rd * tca;
    vec3 radialDir = normalize(closestPt - u_sun_center + vec3(1e-5));
    float angle = atan(v_uv.y, v_uv.x);

    // Mask interior behind solid sun disk cleanly while preserving subtle edge atmospheric bleed
    float outsideSun = smoothstep(0.95, 1.05, rRatio);
    float frontHaze = (1.0 - smoothstep(0.0, 1.15, rRatio)) * 0.04;

    // 1. Multi-layered radial helmet streamers & polar plumes driven by 3D Simplex fBm
    float radialOutflow = rRatio * 0.55 - u_time * 0.24;
    float streamerA = ridgedFbm4(vec3(radialDir.xy * 3.6, radialOutflow));
    float streamerB = fbm4(radialDir * 5.2 + vec3(radialOutflow * 0.85, -u_time * 0.14, u_time * 0.11));

    // Fine ray-like coronal spikes radiating outward
    float rayAngle1 = snoise(vec3(cos(angle) * 4.2, sin(angle) * 4.2, u_time * 0.16 - rRatio * 0.18));
    float rayAngle2 = snoise(vec3(cos(angle) * 10.5, sin(angle) * 10.5, -u_time * 0.22 + rRatio * 0.14));
    float rays = pow(clamp(0.52 + 0.36 * rayAngle1 + 0.24 * rayAngle2, 0.0, 1.0), 2.1);

    // Equatorial helmet-streamer belt boost vs delicate polar plumes
    float eqWeight = 0.62 + 0.58 * (1.0 - abs(radialDir.y));

    // 2. Smooth multi-exponent radial density decay (inner K-corona + mid streamers + wide F-corona)
    float distAboveLimb = max(rRatio - 0.98, 0.0);
    float innerK = exp(-distAboveLimb * 5.4) * 1.35;
    float midStreamers = exp(-distAboveLimb * 1.95) * (0.35 + 1.05 * streamerA * eqWeight) * (0.45 + 0.85 * rays);
    float outerF = exp(-distAboveLimb * 0.88) * (0.22 + 0.45 * (streamerB * 0.5 + 0.5)) * 0.34;

    // Clean window fade to zero before billboard edge so no rectangular seam ever appears
    float edgeFade = 1.0 - smoothstep(0.45, 1.0, rNorm);

    vec3 colInner = vec3(3.15, 1.55, 0.42);
    vec3 colMid   = vec3(2.05, 0.55, 0.09);
    vec3 colOuter = vec3(0.85, 0.16, 0.03);

    vec3 corona = colInner * innerK
                + colMid   * midStreamers * 0.88
                + colOuter * outerF * 0.85;

    corona = corona * outsideSun + vec3(1.1, 0.38, 0.08) * frontHaze;
    corona *= edgeFade;

    fragColor = vec4(max(corona, vec3(0.0)), 1.0);
}
"""
)

CORONAL_LOOP_VERT_SHADER = r"""
#version 330 core
in vec3 in_center;      // Spine position on magnetic arch
in vec3 in_radial;      // Unit radial offset around tube cross-section
in vec3 in_tangent;     // Unit tangent along the magnetic flux tube
in float in_u;          // Arc parameter [0..1] from footpoint A to footpoint B
in float in_v;          // Circumferential parameter [0..1]
in float in_radius;     // Base cross-section tube radius
in vec4 in_params;      // (loop_phase, speed, twist_rate, prominence_height)

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_sun_center;
uniform float u_time;

out vec3 v_world_pos;
out vec3 v_normal;
out float v_u;
out float v_v;
out vec4 v_params;

void main() {
    float phase = in_params.x;
    float speed = in_params.y;
    float twistRate = in_params.z;

    // Arch height envelope: 0 at anchored solar footpoints, 1 at apex
    float archEnv = sin(in_u * 3.14159265);

    // Dynamic magnetic breathing & slow lateral wave oscillation of the flux rope
    vec3 fromSun = normalize(in_center - u_sun_center);
    float breath = 1.0 + 0.11 * sin(u_time * speed * 0.85 + phase) * archEnv;
    vec3 binorm = normalize(cross(in_tangent, fromSun + vec3(1e-4)));
    float sway = 0.16 * sin(u_time * speed * 1.15 + in_u * 6.28318 + phase) * archEnv * archEnv;

    vec3 animatedCenter = u_sun_center + (in_center - u_sun_center) * breath + binorm * sway;

    // Helical twisting of the magnetic field tube cross-section along u + time
    float twistAngle = in_u * twistRate * 6.28318 + u_time * speed * 1.45 + phase;
    vec3 n1 = normalize(in_radial);
    vec3 n2 = normalize(cross(in_tangent, n1));
    vec3 twistedDir = cos(twistAngle) * n1 + sin(twistAngle) * n2;

    // Pinch slightly at footpoints, swell softly near apex with plasma pulses
    float pulseSwell = 1.0 + 0.25 * sin(in_u * 14.0 - u_time * speed * 4.5 + phase) * archEnv;
    float r = in_radius * (0.50 + 0.60 * archEnv) * pulseSwell;

    vec3 worldPos = animatedCenter + twistedDir * r;

    v_world_pos = worldPos;
    v_normal = twistedDir;
    v_u = in_u;
    v_v = in_v;
    v_params = in_params;

    gl_Position = u_proj * u_view * vec4(worldPos, 1.0);
}
"""

CORONAL_LOOP_FRAG_SHADER = (
    "#version 330 core\n"
    + GLSL_NOISE_LIB
    + r"""
in vec3 v_world_pos;
in vec3 v_normal;
in float v_u;
in float v_v;
in vec4 v_params;

uniform vec3 u_cam_pos;
uniform float u_time;

out vec4 fragColor;

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_cam_pos - v_world_pos);
    float coreAlign = abs(dot(N, V)); // 1 at tube center, 0 at glancing edge

    float phase = v_params.x;
    float speed = v_params.y;
    float twist = v_params.z;

    // Bidirectional plasma packets surging along twisted magnetic field lines
    float flow1 = fract(v_u * 2.5 - u_time * speed * 0.42 + phase * 0.159);
    float flow2 = fract(v_u * 3.8 + u_time * speed * 0.31 + phase * 0.271);
    float surge1 = exp(-pow((flow1 - 0.5) * 4.8, 2.0));
    float surge2 = exp(-pow((flow2 - 0.5) * 5.6, 2.0));

    // Helical braided plasma strands wrapped around tube
    float strandAng = (v_v + v_u * twist * 0.5) * 6.28318 * 3.0;
    float braids = 0.50 + 0.50 * sin(strandAng - u_time * speed * 2.2 + phase);

    // 3D noise filament breakup
    float turb = ridgedFbm4(vec3(v_u * 7.5, v_v * 3.5, u_time * 0.45 + phase));

    // Smooth entry/exit at both solar footpoints (strictly monotonic smoothstep)
    float footFade = smoothstep(0.0, 0.08, v_u) * (1.0 - smoothstep(0.92, 1.0, v_u));

    // Volumetric tube opacity: blazing white-gold core inside molten fiery sheath
    float radialProfile = pow(coreAlign, 1.45);
    float hotCore = pow(coreAlign, 3.6) * (0.40 + 0.95 * surge1 + 0.70 * surge2);

    float energy = (0.34 + 0.65 * surge1 + 0.48 * surge2) * braids * (0.50 + 0.80 * turb);
    float loopPulse = 0.72 + 0.35 * sin(u_time * speed * 0.65 + phase * 2.1);

    vec3 sheathCol = vec3(1.95, 0.36, 0.04);
    vec3 plasmaCol = vec3(3.60, 1.75, 0.45);
    vec3 color = sheathCol * radialProfile * energy * 1.05
               + plasmaCol * hotCore * (0.65 + 0.55 * turb);

    color *= footFade * loopPulse * 0.75;
    fragColor = vec4(max(color, vec3(0.0)), 1.0);
}
"""
)

SOLAR_FLARE_VERT_SHADER = r"""
#version 330 core
in vec3 in_anchor;      // Unit sphere launch point on sun
in vec3 in_tangent;     // Magnetic field line curvature axis
in vec4 in_params;      // (phase_offset, life_duration, speed, curve_strength)
in vec3 in_visual;      // (base_size, temp_seed, helix_amp)

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_sun_center;
uniform float u_sun_radius;
uniform float u_time;
uniform float u_viewport_h;

out float v_life;
out float v_temp;
out float v_seed;

void main() {
    float phase = in_params.x;
    float duration = in_params.y;
    float speed = in_params.z;
    float curveStrength = in_params.w;

    // Cyclic normalized particle age [0..1]
    float age = fract((u_time + phase) / duration);
    float tSec = age * duration;

    // Slowly rotate launch anchor with the sun
    float rotAng = u_time * 0.055;
    float cr = cos(rotAng), sr = sin(rotAng);
    vec3 anchor = normalize(vec3(
        cr * in_anchor.x + sr * in_anchor.z,
        in_anchor.y,
        -sr * in_anchor.x + cr * in_anchor.z
    ));
    vec3 tang = normalize(vec3(
        cr * in_tangent.x + sr * in_tangent.z,
        in_tangent.y,
        -sr * in_tangent.x + cr * in_tangent.z
    ));
    vec3 binorm = normalize(cross(anchor, tang + vec3(1e-4)));

    // Curved magnetic field-line eruption trajectory:
    // Launches radially, bends sharply along magnetic field line, then dissolves into space
    float radialDist = speed * (tSec * 0.72 + 0.16 * tSec * tSec * (1.0 - 0.35 * age));
    float arcBend = curveStrength * sin(age * 3.14159265 * 0.90) * (1.0 + 0.80 * age) * radialDist * 0.62;

    // Tight magnetic helical spiral around the erupting flux line
    float helixPhase = age * 15.0 + phase * 11.3;
    float helixR = in_visual.z * (0.08 + 0.92 * age * age);
    vec3 helixOffset = (cos(helixPhase) * binorm + sin(helixPhase) * tang) * helixR;

    // Low-energy ejecta fall back along arches; fast CME jets escape and dissolve
    float escapeBias = smoothstep(0.95, 2.35, speed);
    float height = mix(sin(age * 3.14159) * radialDist * 0.62, radialDist, escapeBias);

    vec3 worldPos = u_sun_center
                  + anchor * (u_sun_radius * 0.995 + height)
                  + tang * arcBend
                  + helixOffset;

    vec4 viewPos = u_view * vec4(worldPos, 1.0);
    gl_Position = u_proj * viewPos;

    // Crisp hot plasma sparks that expand softly as they dissolve
    float depth = max(-viewPos.z, 0.5);
    float sizeScale = mix(0.65, 1.55, pow(age, 0.7)) * (1.0 - 0.35 * pow(age, 2.2));
    gl_PointSize = clamp((in_visual.x * sizeScale * u_viewport_h) / (depth * 1.85), 1.4, 26.0);

    v_life = age;
    v_temp = in_visual.y;
    v_seed = phase;
}
"""

SOLAR_FLARE_FRAG_SHADER = r"""
#version 330 core
in float v_life;
in float v_temp;
in float v_seed;

uniform float u_time;

out vec4 fragColor;

void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;

    // Crisp incandescent plasma ember core + delicate glowing wisp halo
    float core = exp(-r2 * 10.5);
    float halo = pow(1.0 - r2, 2.8) * 0.42;
    float shape = core + halo;

    // Smooth birth and gradual thermal dissolution into deep space
    float fadeIn = smoothstep(0.0, 0.08, v_life);
    float fadeOut = pow(1.0 - smoothstep(0.16, 1.0, v_life), 1.85);
    float flicker = 0.80 + 0.25 * sin(u_time * 9.5 + v_seed * 23.0 + v_life * 18.0);

    // Thermal cooling curve: incandescent gold-white -> molten orange -> ember crimson
    float cool = clamp(v_life * 1.15 + (1.0 - v_temp) * 0.25, 0.0, 1.0);
    vec3 cHot = vec3(3.8, 2.65, 1.15);
    vec3 cMid = vec3(2.4, 0.75, 0.11);
    vec3 cCool = vec3(1.05, 0.12, 0.015);
    vec3 col = cool < 0.45
             ? mix(cHot, cMid, cool / 0.45)
             : mix(cMid, cCool, (cool - 0.45) / 0.55);

    vec3 finalCol = col * shape * fadeIn * fadeOut * flicker * 1.28;
    fragColor = vec4(finalCol, 1.0);
}
"""


# =============================================================================
# GLSL 330 SHADERS — PART B: SPIRAL GALAXY, NEBULAE, DUST LANES & DEEP FIELD
# =============================================================================

GALAXY_STAR_VERT_SHADER = r"""
#version 330 core
in vec3 in_cyl;      // (radius, base_angle, local_y)
in vec3 in_color;    // Spectral RGB color (O/B/A/F/G/K/M)
in vec4 in_props;    // (base_size, twinkle_phase, twinkle_freq, luminosity)

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_time;
uniform float u_viewport_h;

out vec3 v_color;
out float v_twinkle;
out float v_lum;
out float v_seed;

void main() {
    float r = in_cyl.x;
    float baseAng = in_cyl.y;
    float ly = in_cyl.z;

    // Realistic differential Keplerian-like rotation: inner stars orbit faster than outer stars
    float omega = 0.52 / pow(r + 1.55, 1.08);
    float ang = baseAng + u_time * omega;

    // Gentle vertical harmonic disk oscillation
    float yOsc = ly + 0.06 * sin(u_time * 0.35 + in_props.y + r * 0.4);
    vec3 localPos = vec3(r * cos(ang), yOsc, r * sin(ang));

    vec4 worldPos = u_model * vec4(localPos, 1.0);
    vec4 viewPos = u_view * worldPos;
    gl_Position = u_proj * viewPos;

    // Multi-harmonic scintillation / twinkling over time
    float tw1 = sin(u_time * in_props.z + in_props.y);
    float tw2 = sin(u_time * in_props.z * 2.17 + in_props.y * 1.73);
    float twinkle = 0.78 + 0.17 * tw1 + 0.09 * tw2;

    float depth = max(-viewPos.z, 1.0);
    float pSize = (in_props.x * u_viewport_h) / (depth * 2.25);
    pSize *= (0.90 + 0.16 * twinkle);
    gl_PointSize = clamp(pSize, 1.05, 12.5);

    v_color = in_color;
    v_twinkle = twinkle;
    v_lum = in_props.w;
    v_seed = in_props.y;
}
"""

GALAXY_STAR_FRAG_SHADER = r"""
#version 330 core
in vec3 v_color;
in float v_twinkle;
in float v_lum;
in float v_seed;

out vec4 fragColor;

void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;

    // Crisp telescope Airy PSF: diamond-sharp pinpoint core + delicate chromatic halo
    float core = exp(-r2 * 18.0);
    float halo = exp(-r2 * 4.4) * 0.34;

    // Subtle 4-point optical diffraction spikes on bright O/B supergiants
    float spikeStrength = smoothstep(1.55, 3.0, v_lum);
    float spX = exp(-abs(p.y) * 16.0) * exp(-abs(p.x) * 3.0);
    float spY = exp(-abs(p.x) * 16.0) * exp(-abs(p.y) * 3.0);
    float spikes = (spX + spY) * 0.48 * spikeStrength;

    float intensity = (core + halo + spikes) * v_lum * v_twinkle * 0.62;
    vec3 starCol = mix(v_color, vec3(1.0, 0.99, 0.97), clamp(core * 0.32, 0.0, 0.45));
    fragColor = vec4(starCol * intensity, 1.0);
}
"""

NEBULA_VERT_SHADER = r"""
#version 330 core
// Per-vertex quad corner [-1, 1]
in vec2 in_quad;

// Per-instance nebula cloud attributes
in vec3 in_cyl;      // (radius, base_angle, local_y)
in vec3 in_color;    // Multi-colored HDR nebula tint (indigo/violet/teal/magenta/amber)
in vec4 in_props;    // (size, rot_angle, seed, opacity)

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_time;

out vec2 v_uv;
out vec3 v_color;
out vec3 v_seed_data; // (seed, opacity, radius_norm)

void main() {
    float r = in_cyl.x;
    float baseAng = in_cyl.y;
    float ly = in_cyl.z;

    // Synchronized differential Keplerian rotation with the spiral galaxy stars
    float omega = 0.52 / pow(r + 1.55, 1.08);
    float ang = baseAng + u_time * omega;

    vec3 localCenter = vec3(r * cos(ang), ly, r * sin(ang));
    vec3 worldCenter = (u_model * vec4(localCenter, 1.0)).xyz;

    // Slow billowing billboard rotation
    float rot = in_props.y + u_time * (0.025 + 0.015 * fract(in_props.z));
    float c = cos(rot), s = sin(rot);
    vec2 rotQuad = vec2(c * in_quad.x - s * in_quad.y, s * in_quad.x + c * in_quad.y);

    float sz = in_props.x;
    vec3 worldPos = worldCenter
                  + u_cam_right * (rotQuad.x * sz)
                  + u_cam_up    * (rotQuad.y * sz);

    v_uv = in_quad;
    v_color = in_color;
    v_seed_data = vec3(in_props.z, in_props.w, r / 23.0);

    gl_Position = u_proj * u_view * vec4(worldPos, 1.0);
}
"""

NEBULA_FRAG_SHADER = (
    "#version 330 core\n"
    + GLSL_NOISE_LIB
    + r"""
in vec2 v_uv;
in vec3 v_color;
in vec3 v_seed_data;

uniform float u_time;

out vec4 fragColor;

void main() {
    float r = length(v_uv);
    if (r >= 1.0) discard;

    float seed = v_seed_data.x;
    float opacity = v_seed_data.y;

    // Animated 3D procedural volumetric fBm cloud wisps
    vec3 p = vec3(v_uv * 2.35, seed * 7.31 + u_time * 0.045);
    vec3 warp = vec3(
        fbm3(p + vec3(1.7, -2.3, 0.4)),
        fbm3(p + vec3(-3.1, 1.9, -1.2)),
        0.0
    );
    float n1 = fbm4(p + warp * 0.65) * 0.5 + 0.5;
    float wisps = ridgedFbm4(p * 1.55 + warp * 0.48 - vec3(0.0, 0.0, u_time * 0.03));

    // Strictly monotonic radial falloff -> zero hard billboard boundaries
    float radial = pow(1.0 - smoothstep(0.04, 1.0, r), 2.15);
    float density = radial * smoothstep(0.20, 0.88, n1 * 0.68 + wisps * 0.48);

    // Rich internal ionized glowing filaments
    vec3 knotCol = mix(v_color, v_color * 1.55 + vec3(0.14, 0.10, 0.18), pow(wisps, 2.0) * 0.48);
    vec3 finalCol = knotCol * density * opacity;

    fragColor = vec4(max(finalCol, vec3(0.0)), 1.0);
}
"""
)

DUST_VERT_SHADER = r"""
#version 330 core
in vec2 in_quad;

in vec3 in_cyl;      // (radius, base_angle, local_y)
in vec3 in_tint;     // Deep dark umber / charcoal-maroon edge scatter color
in vec4 in_props;    // (size, rot_angle, seed, optical_depth)

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_time;

out vec2 v_uv;
out vec3 v_tint;
out vec3 v_seed_data;

void main() {
    float r = in_cyl.x;
    float baseAng = in_cyl.y;
    float ly = in_cyl.z;

    // Exact same differential rotation so dark dust lanes trace the trailing spiral arm edges
    float omega = 0.52 / pow(r + 1.55, 1.08);
    float ang = baseAng + u_time * omega;

    vec3 localCenter = vec3(r * cos(ang), ly, r * sin(ang));
    vec3 worldCenter = (u_model * vec4(localCenter, 1.0)).xyz;

    float rot = in_props.y + u_time * 0.018;
    float c = cos(rot), s = sin(rot);
    vec2 rotQuad = vec2(c * in_quad.x - s * in_quad.y, s * in_quad.x + c * in_quad.y);

    float sz = in_props.x;
    vec3 worldPos = worldCenter
                  + u_cam_right * (rotQuad.x * sz)
                  + u_cam_up    * (rotQuad.y * sz);

    v_uv = in_quad;
    v_tint = in_tint;
    v_seed_data = vec3(in_props.z, in_props.w, r / 23.0);

    gl_Position = u_proj * u_view * vec4(worldPos, 1.0);
}
"""

DUST_FRAG_SHADER = (
    "#version 330 core\n"
    + GLSL_NOISE_LIB
    + r"""
in vec2 v_uv;
in vec3 v_tint;
in vec3 v_seed_data;

uniform float u_time;

out vec4 fragColor;

void main() {
    float r = length(v_uv);
    if (r >= 1.0) discard;

    float seed = v_seed_data.x;
    float opticalDepth = v_seed_data.y;

    // Filamentous dark interstellar dust tendrils driven by domain-warped 3D fBm
    vec3 p = vec3(v_uv * 2.85, seed * 9.17 + u_time * 0.025);
    vec3 warp = vec3(fbm3(p + vec3(2.1, -1.4, 0.0)), fbm3(p + vec3(-1.8, 3.2, 0.5)), 0.0);
    float n = fbm4(p + warp * 0.55) * 0.5 + 0.5;
    float clump = ridgedFbm4(p * 1.65 + warp * 0.45);

    // Strictly monotonic radial falloff -> organic wispy dust absorption with zero circle edges
    float radial = pow(1.0 - smoothstep(0.08, 1.0, r), 2.15);
    float alpha = radial * smoothstep(0.28, 0.86, n * 0.62 + clump * 0.48) * opticalDepth;
    if (alpha < 0.004) discard;

    // Dark silhouetted dust core with faint backlit reddish-brown scattering near edge
    float edgeScatter = smoothstep(0.32, 0.78, r) * (1.0 - smoothstep(0.78, 1.0, r));
    vec3 dustCol = v_tint * (0.08 + 0.42 * edgeScatter);

    fragColor = vec4(dustCol, clamp(alpha, 0.0, 0.64));
}
"""
)

GALAXY_CORE_VERT_SHADER = r"""
#version 330 core
in vec2 in_quad;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_core_radius;

out vec2 v_uv;

void main() {
    vec3 worldCenter = (u_model * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 worldPos = worldCenter
                  + u_cam_right * (in_quad.x * u_core_radius)
                  + u_cam_up    * (in_quad.y * u_core_radius);
    v_uv = in_quad;
    gl_Position = u_proj * u_view * vec4(worldPos, 1.0);
}
"""

GALAXY_CORE_FRAG_SHADER = (
    "#version 330 core\n"
    + GLSL_NOISE_LIB
    + r"""
in vec2 v_uv;

uniform float u_time;

out vec4 fragColor;

void main() {
    float rRaw = length(v_uv);
    if (rRaw >= 1.0) discard;

    // Slightly elliptical galactic bar profile tilted along galaxy plane
    float ca = cos(0.45), sa = sin(0.45);
    vec2 rotUv = vec2(ca * v_uv.x - sa * v_uv.y, sa * v_uv.x + ca * v_uv.y);
    vec2 ellUv = vec2(rotUv.x * 0.78, rotUv.y * 1.28);
    float r = length(ellUv);

    // Soft swirling inner core mist
    float swirl = fbm3(vec3(ellUv * 3.6, u_time * 0.035)) * 0.5 + 0.5;
    float nucleus = exp(-r * r * 36.0) * 1.65;
    float bulge   = exp(-r * 5.2) * (0.68 + 0.22 * swirl);
    float outer   = pow(1.0 - smoothstep(0.0, 1.0, min(r, 1.0)), 3.2) * (0.18 + 0.12 * swirl);

    vec3 colNucleus = vec3(3.10, 2.55, 1.75);
    vec3 colBulge   = vec3(1.45, 0.86, 0.40);
    vec3 colOuter   = vec3(0.42, 0.18, 0.58);

    vec3 col = colNucleus * nucleus + colBulge * bulge + colOuter * outer;
    col *= 1.0 - smoothstep(0.58, 1.0, rRaw);
    fragColor = vec4(max(col, vec3(0.0)), 1.0);
}
"""
)

DEEP_STAR_VERT_SHADER = r"""
#version 330 core
in vec3 in_pos;
in vec3 in_color;
in vec4 in_props;    // (size, phase, freq, lum)

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_time;
uniform float u_viewport_h;

out vec3 v_color;
out float v_lum;

void main() {
    // Remove camera translation so deep field stays at cosmic infinity (with subtle parallax)
    mat4 viewNoTrans = u_view;
    viewNoTrans[3] = vec4(u_view[3].xyz * 0.04, 1.0);
    vec4 viewPos = viewNoTrans * vec4(in_pos, 1.0);
    gl_Position = u_proj * viewPos;

    float tw = 0.78 + 0.22 * sin(u_time * in_props.z + in_props.y);
    gl_PointSize = clamp(in_props.x * (u_viewport_h / 1280.0) * (0.92 + 0.12 * tw), 1.0, 8.5);
    v_color = in_color;
    v_lum = in_props.w * tw;
}
"""

DEEP_STAR_FRAG_SHADER = r"""
#version 330 core
in vec3 v_color;
in float v_lum;

out vec4 fragColor;

void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;
    float shape = exp(-r2 * 10.5) + exp(-r2 * 3.0) * 0.25;
    fragColor = vec4(v_color * shape * v_lum, 1.0);
}
"""

DISTANT_GALAXY_VERT_SHADER = r"""
#version 330 core
in vec2 in_quad;

// Per-instance distant deep-field galaxy attributes
in vec3 in_pos;        // World position on far celestial sphere
in vec3 in_color;      // Outer arm / halo tint
in vec4 in_shape;      // (size, rot_angle, tilt_y, galaxy_type: <0.65 spiral, >=0.65 elliptical)
in vec4 in_detail;     // (arm_count, winding, brightness, seed)

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;

out vec2 v_uv;
out vec3 v_color;
out vec4 v_shape;
out vec4 v_detail;

void main() {
    float sz = in_shape.x;
    float rot = in_shape.y;
    float c = cos(rot), s = sin(rot);
    vec2 rQuad = vec2(c * in_quad.x - s * in_quad.y, s * in_quad.x + c * in_quad.y);

    // Subtle parallax translation scaling so distant galaxies feel truly cosmic
    mat4 viewFar = u_view;
    viewFar[3] = vec4(u_view[3].xyz * 0.06, 1.0);

    vec3 worldPos = in_pos
                  + u_cam_right * (rQuad.x * sz)
                  + u_cam_up    * (rQuad.y * sz);

    v_uv = in_quad;
    v_color = in_color;
    v_shape = in_shape;
    v_detail = in_detail;

    gl_Position = u_proj * viewFar * vec4(worldPos, 1.0);
}
"""

DISTANT_GALAXY_FRAG_SHADER = r"""
#version 330 core
in vec2 v_uv;
in vec3 v_color;
in vec4 v_shape;
in vec4 v_detail;

uniform float u_time;

out vec4 fragColor;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    float tilt = max(v_shape.z, 0.24);
    float galType = v_shape.w;
    float armCount = v_detail.x;
    float winding = v_detail.y;
    float brightness = v_detail.z;
    float seed = v_detail.w;

    // Foreshorten vertically to simulate 3D galaxy inclination
    vec2 p = vec2(v_uv.x, v_uv.y / tilt);
    float r = length(p);
    if (r >= 1.0) discard;

    float core = exp(-r * r * 36.0) * 1.45;
    float bulge = exp(-r * 6.2) * 0.72;
    vec3 coreCol = vec3(1.35, 1.15, 0.85);

    vec3 col = vec3(0.0);
    if (galType < 0.68) {
        // Distant Spiral Galaxy: logarithmic arm intensity + knots
        float theta = atan(p.y, p.x) + u_time * 0.015 * (seed > 0.5 ? 1.0 : -1.0);
        float logSpiral = armCount * (theta - winding * log(r + 0.08));
        float arms = pow(0.5 + 0.5 * cos(logSpiral), 2.4);
        float armMask = exp(-r * 2.9) * smoothstep(0.04, 0.24, r) * (1.0 - smoothstep(0.45, 1.0, r));
        // Star-forming knots along arms
        float knot = 0.75 + 0.45 * sin(logSpiral * 2.0 + r * 18.0 + seed * 12.0);
        float disk = exp(-r * 3.8) * 0.25;
        col = coreCol * (core + bulge * 0.85)
            + v_color * (arms * armMask * knot * 1.30 + disk);
    } else {
        // Distant Elliptical Galaxy: smooth Sersic-like golden spheroid
        float sersic = exp(-pow(r * 3.8, 0.85));
        col = mix(v_color, coreCol, exp(-r * 3.2)) * (core * 0.9 + sersic * 0.85);
    }

    float edge = 1.0 - smoothstep(0.70, 1.0, r);
    fragColor = vec4(col * brightness * edge, 1.0);
}
"""


# =============================================================================
# GLSL 330 SHADERS — PART C: HDR MULTI-PASS BLOOM, ACES, CHROMATIC ABERRATION & GRAIN
# =============================================================================

POST_VERT_SHADER = r"""
#version 330 core
in vec2 in_pos;
out vec2 v_uv;

void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

BLOOM_PREFILTER_FRAG_SHADER = r"""
#version 330 core
in vec2 v_uv;

uniform sampler2D u_scene_tex;
uniform float u_threshold;
uniform float u_knee;

out vec4 fragColor;

void main() {
    // 4-tap box prefilter to reduce high-frequency star aliasing before blur pyramid
    vec2 ts = 1.0 / vec2(textureSize(u_scene_tex, 0));
    vec3 c = 0.25 * (
        texture(u_scene_tex, v_uv + ts * vec2(-0.5, -0.5)).rgb +
        texture(u_scene_tex, v_uv + ts * vec2( 0.5, -0.5)).rgb +
        texture(u_scene_tex, v_uv + ts * vec2(-0.5,  0.5)).rgb +
        texture(u_scene_tex, v_uv + ts * vec2( 0.5,  0.5)).rgb
    );

    // Clamp extreme HDR spikes softly to prevent harsh square bokeh artifacts
    c = min(c, vec3(18.0));

    float br = max(c.r, max(c.g, c.b));
    float soft = clamp(br - u_threshold + u_knee, 0.0, 2.0 * u_knee);
    soft = (soft * soft) / max(4.0 * u_knee + 1e-5, 1e-5);
    float contrib = max(soft, br - u_threshold) / max(br, 1e-5);

    fragColor = vec4(c * contrib, 1.0);
}
"""

BLOOM_BLUR_FRAG_SHADER = r"""
#version 330 core
in vec2 v_uv;

uniform sampler2D u_tex;
uniform vec2 u_direction; // (1/w, 0) for horizontal pass, (0, 1/h) for vertical pass

out vec4 fragColor;

void main() {
    // 13-tap linear-filtered high-radius separable Gaussian kernel (7 texture fetches)
    const float weights[5] = float[](0.22702703, 0.19459459, 0.12162162, 0.05405405, 0.01621622);
    const float offsets[5] = float[](0.0, 1.38461538, 3.23076923, 5.07692308, 6.92307692);

    vec3 result = texture(u_tex, v_uv).rgb * weights[0];
    for (int i = 1; i < 5; ++i) {
        vec2 off = u_direction * offsets[i];
        result += texture(u_tex, v_uv + off).rgb * weights[i];
        result += texture(u_tex, v_uv - off).rgb * weights[i];
    }
    fragColor = vec4(result, 1.0);
}
"""

POST_COMPOSITE_FRAG_SHADER = r"""
#version 330 core
in vec2 v_uv;

uniform sampler2D u_scene_tex;
uniform sampler2D u_bloom_0;   // 1/2 res tight optical glow
uniform sampler2D u_bloom_1;   // 1/4 res medium halo
uniform sampler2D u_bloom_2;   // 1/8 res wide atmosphere glow
uniform sampler2D u_bloom_3;   // 1/16 res colossal solar atmospheric bloom

uniform float u_bloom_strength;
uniform float u_exposure;
uniform float u_time;
uniform vec2 u_resolution;

out vec4 fragColor;

// Fitted Stephen Hill ACES Filmic Tone Mapping (RRT + ODT)
const mat3 ACES_INPUT_MAT = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
);
const mat3 ACES_OUTPUT_MAT = mat3(
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
);

vec3 acesFilmic(vec3 color) {
    color = ACES_INPUT_MAT * color;
    vec3 a = color * (color + 0.0245786) - 0.000090537;
    vec3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
    color = a / b;
    return clamp(ACES_OUTPUT_MAT * color, 0.0, 1.0);
}

// High-frequency hash for fine analog film grain & anti-banding dither
float filmHash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec2 centered = v_uv - 0.5;
    float aspect = u_resolution.x / max(u_resolution.y, 1.0);
    vec2 aspectPos = vec2(centered.x * aspect, centered.y);
    float edgeDist2 = dot(centered, centered);

    // 1. Subtle Optical Radial Chromatic Aberration (strongest cleanly at frame edges)
    vec2 caDir = centered * (0.0046 * edgeDist2);
    float rScene = texture(u_scene_tex, clamp(v_uv + caDir, 0.001, 0.999)).r;
    float gScene = texture(u_scene_tex, v_uv).g;
    float bScene = texture(u_scene_tex, clamp(v_uv - caDir, 0.001, 0.999)).b;
    vec3 hdrColor = vec3(rScene, gScene, bScene);

    // 2. Multi-Pass Optical Gaussian Bloom Synthesis (weighted across 4 octaves)
    vec3 b0 = texture(u_bloom_0, v_uv).rgb;
    vec3 b1 = texture(u_bloom_1, v_uv).rgb;
    vec3 b2 = texture(u_bloom_2, v_uv).rgb;
    vec3 b3 = texture(u_bloom_3, v_uv).rgb;
    // Slight chromatic dispersion in the outer optical bloom halo
    vec3 bWideR = texture(u_bloom_2, clamp(v_uv + caDir * 1.6, 0.001, 0.999)).rgb;
    vec3 bWideB = texture(u_bloom_2, clamp(v_uv - caDir * 1.6, 0.001, 0.999)).rgb;
    vec3 b2Ca = vec3(bWideR.r, b2.g, bWideB.b);

    vec3 bloom = b0 * 0.24 + b1 * 0.34 + b2Ca * 0.42 + b3 * 0.55;
    hdrColor += bloom * u_bloom_strength;

    // 3. Subtle deep-space cosmic ambient gradient so empty sky has velvety depth
    vec3 deepSky = mix(vec3(0.004, 0.004, 0.014), vec3(0.011, 0.007, 0.022), smoothstep(0.0, 1.0, v_uv.y));
    hdrColor += deepSky;

    // 4. Exposure & ACES Filmic Tone Mapping
    vec3 mapped = acesFilmic(hdrColor * u_exposure);

    // 5. Subtle Cinematic Color Grading & Edge Vignette
    float vig = 1.0 - smoothstep(0.32, 0.92, length(aspectPos) * 1.08) * 0.32;
    mapped *= vig;

    // 6. sRGB Gamma Encoding
    mapped = pow(max(mapped, vec3(0.0)), vec3(1.0 / 2.2));

    // 7. Calibrated Fine Animated Film Grain (luminance-weighted, stronger in mid-shadows)
    vec2 px = v_uv * u_resolution;
    float tSeed = fract(u_time * 24.0) * 173.17;
    float g1 = filmHash(px + vec2(tSeed, tSeed * 1.37));
    float g2 = filmHash(px + vec2(-tSeed * 0.71, tSeed * 2.11));
    float triGrain = (g1 + g2 - 1.0); // Triangular PDF zero-mean noise
    float luma = dot(mapped, vec3(0.2126, 0.7152, 0.0722));
    float grainMask = mix(0.024, 0.009, smoothstep(0.05, 0.85, luma));
    mapped += vec3(triGrain) * grainMask;

    fragColor = vec4(clamp(mapped, 0.0, 1.0), 1.0);
}
"""


# =============================================================================
# PROCEDURAL 3D GEOMETRY & ASTROPHYSICAL PARTICLE GENERATORS (VECTORIZED NUMPY)
# =============================================================================

def generate_icosphere(subdivisions: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a seamless, uniform-triangle icosphere with zero polar pinching.
    subdivisions=5 -> 10,242 vertices, 20,480 triangles.
    subdivisions=4 ->  2,562 vertices,  5,120 triangles.
    """
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    verts: List[Tuple[float, float, float]] = [
        (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
        ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
        ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1),
    ]
    # Normalize base vertices onto unit sphere
    verts = [tuple((np.array(v, dtype=np.float64) / np.linalg.norm(v)).tolist()) for v in verts]

    faces: List[Tuple[int, int, int]] = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    for _ in range(subdivisions):
        midpoint_cache = {}
        new_faces: List[Tuple[int, int, int]] = []

        def get_midpoint(i1: int, i2: int) -> int:
            key = (i1, i2) if i1 < i2 else (i2, i1)
            idx = midpoint_cache.get(key)
            if idx is not None:
                return idx
            v1 = verts[i1]
            v2 = verts[i2]
            mx = (v1[0] + v2[0]) * 0.5
            my = (v1[1] + v2[1]) * 0.5
            mz = (v1[2] + v2[2]) * 0.5
            inv_len = 1.0 / math.sqrt(mx * mx + my * my + mz * mz)
            idx = len(verts)
            verts.append((mx * inv_len, my * inv_len, mz * inv_len))
            midpoint_cache[key] = idx
            return idx

        for a, b, c in faces:
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.append((a, ab, ca))
            new_faces.append((b, bc, ab))
            new_faces.append((c, ca, bc))
            new_faces.append((ab, bc, ca))
        faces = new_faces

    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.uint32).ravel()


def generate_coronal_loops(
    rng: np.random.Generator,
    sun_center: np.ndarray,
    sun_radius: float,
    loop_count: int = CORONAL_LOOP_COUNT,
    length_segs: int = CORONAL_LOOP_LENGTH_SEGS,
    radial_segs: int = CORONAL_LOOP_RADIAL_SEGS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D magnetic coronal loop spline tubes using cubic Bezier arches
    anchored at pairs of active sunspot footpoints and parallel-transport frames.
    Vertex layout (16 floats per vertex):
        center(3), radial(3), tangent(3), u(1), v(1), radius(1), params(4)
    """
    all_verts = []
    all_indices = []
    vert_offset = 0

    for idx in range(loop_count):
        # Concentrate 60% of prominent loops near the visible silhouette / upper limb
        if idx < int(loop_count * 0.55):
            # Framing the visible rim & upper hemisphere dramatically against space
            phi = rng.uniform(-0.35, 1.15)  # Latitude biased upper/mid hemisphere
            # Azimuth biased toward side rims (-pi/2, +pi/2) and upper crown
            side = rng.choice([-1.0, 1.0])
            theta = side * rng.uniform(0.65, 1.75) + rng.normal(0.0, 0.25)
        else:
            # Active solar belts across the full 3D sphere
            phi = rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 0.72)
            theta = rng.uniform(0.0, 2.0 * math.pi)

        mid_dir = normalize(np.array([
            math.cos(phi) * math.sin(theta),
            math.sin(phi),
            math.cos(phi) * math.cos(theta),
        ], dtype=np.float32))

        # Tangent frame on solar surface for footpoint separation
        ref_up = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(mid_dir[1]) < 0.92 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        t1 = normalize(np.cross(ref_up, mid_dir))
        t2 = normalize(np.cross(mid_dir, t1))
        orient = rng.uniform(0.0, 2.0 * math.pi)
        span_axis = normalize(math.cos(orient) * t1 + math.sin(orient) * t2)
        tilt_axis = normalize(np.cross(mid_dir, span_axis))

        # Footpoint angular separation and majestic arch apex height
        half_span = rng.uniform(0.14, 0.42)
        apex_height = sun_radius * rng.uniform(0.20, 0.78)
        if idx < 8:
            # Hero mega-prominence arches rising high off the solar limb
            apex_height = sun_radius * rng.uniform(0.45, 0.96)
            half_span = rng.uniform(0.24, 0.46)

        p0_dir = normalize(mid_dir * math.cos(half_span) - span_axis * math.sin(half_span))
        p3_dir = normalize(mid_dir * math.cos(half_span) + span_axis * math.sin(half_span))

        # Anchor slightly beneath photosphere so loops emerge seamlessly from plasma
        p0 = sun_center + p0_dir * (sun_radius * 0.965)
        p3 = sun_center + p3_dir * (sun_radius * 0.965)

        # Control points pull arch outward and introduce gentle S-curve magnetic shear
        shear = rng.uniform(-0.32, 0.32) * apex_height
        p1 = sun_center + normalize(p0_dir * 0.65 + mid_dir * 0.35) * (sun_radius + apex_height * 1.18) + tilt_axis * shear
        p2 = sun_center + normalize(p3_dir * 0.65 + mid_dir * 0.35) * (sun_radius + apex_height * 1.18) - tilt_axis * shear

        # Sample cubic Bezier spine
        us = np.linspace(0.0, 1.0, length_segs + 1, dtype=np.float32)
        one_m = 1.0 - us
        spine = (
            (one_m ** 3)[:, None] * p0[None, :]
            + (3.0 * (one_m ** 2) * us)[:, None] * p1[None, :]
            + (3.0 * one_m * (us ** 2))[:, None] * p2[None, :]
            + (us ** 3)[:, None] * p3[None, :]
        )
        # Analytic Bezier derivative for smooth tangents
        deriv = (
            (3.0 * (one_m ** 2))[:, None] * (p1 - p0)[None, :]
            + (6.0 * one_m * us)[:, None] * (p2 - p1)[None, :]
            + (3.0 * (us ** 2))[:, None] * (p3 - p2)[None, :]
        )
        tangents = deriv / np.maximum(np.linalg.norm(deriv, axis=1, keepdims=True), 1e-6)

        # Parallel-transport Bishop frame along spline to prevent Frenet flips
        n_vec = normalize(np.cross(tangents[0], tilt_axis))
        if np.linalg.norm(n_vec) < 1e-4:
            n_vec = normalize(np.cross(tangents[0], mid_dir))
        normals = np.zeros_like(tangents)
        binormals = np.zeros_like(tangents)
        normals[0] = n_vec
        binormals[0] = normalize(np.cross(tangents[0], n_vec))
        for i in range(1, length_segs + 1):
            t_prev = tangents[i - 1]
            t_curr = tangents[i]
            axis = np.cross(t_prev, t_curr)
            axis_len = float(np.linalg.norm(axis))
            if axis_len > 1e-6:
                axis /= axis_len
                cos_a = max(-1.0, min(1.0, float(np.dot(t_prev, t_curr))))
                ang = math.acos(cos_a)
                # Rodrigues rotation of previous normal
                n_prev = normals[i - 1]
                n_rot = (
                    n_prev * math.cos(ang)
                    + np.cross(axis, n_prev) * math.sin(ang)
                    + axis * float(np.dot(axis, n_prev)) * (1.0 - math.cos(ang))
                )
                normals[i] = normalize(n_rot - t_curr * float(np.dot(n_rot, t_curr)))
            else:
                normals[i] = normals[i - 1]
            binormals[i] = normalize(np.cross(t_curr, normals[i]))

        tube_radius = rng.uniform(0.055, 0.145)
        phase = float(rng.uniform(0.0, 20.0))
        speed = float(rng.uniform(0.65, 1.55))
        twist_rate = float(rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 4.2))
        prom_h = float(apex_height / sun_radius)

        # Build ring vertices
        ring_angles = np.linspace(0.0, 2.0 * math.pi, radial_segs + 1, dtype=np.float32)
        loop_verts = np.zeros(((length_segs + 1) * (radial_segs + 1), 16), dtype=np.float32)
        k = 0
        for i in range(length_segs + 1):
            c_pos = spine[i]
            t_v = tangents[i]
            n_v = normals[i]
            b_v = binormals[i]
            u_val = float(us[i])
            for j in range(radial_segs + 1):
                ang = float(ring_angles[j])
                rad_dir = math.cos(ang) * n_v + math.sin(ang) * b_v
                v_val = float(j) / float(radial_segs)
                loop_verts[k, 0:3] = c_pos
                loop_verts[k, 3:6] = rad_dir
                loop_verts[k, 6:9] = t_v
                loop_verts[k, 9] = u_val
                loop_verts[k, 10] = v_val
                loop_verts[k, 11] = tube_radius
                loop_verts[k, 12:16] = (phase, speed, twist_rate, prom_h)
                k += 1

        all_verts.append(loop_verts)

        # Build triangle indices for this tube
        stride = radial_segs + 1
        tris = []
        for i in range(length_segs):
            row0 = vert_offset + i * stride
            row1 = vert_offset + (i + 1) * stride
            for j in range(radial_segs):
                a = row0 + j
                b = row1 + j
                c = row1 + j + 1
                d = row0 + j + 1
                tris.extend((a, b, c, a, c, d))
        all_indices.append(np.asarray(tris, dtype=np.uint32))
        vert_offset += loop_verts.shape[0]

    return np.vstack(all_verts).astype(np.float32), np.concatenate(all_indices).astype(np.uint32)


def generate_solar_flare_particles(
    rng: np.random.Generator,
    count: int = SOLAR_FLARE_PARTICLE_COUNT,
) -> np.ndarray:
    """
    Generate erupting solar flare / CME plasma jet particles clustered around
    16 active magnetic eruption zones on the sun.
    Layout (13 floats per particle):
        anchor(3), tangent(3), params(4: phase, duration, speed, curve), visual(3: size, temp, helix_amp)
    """
    num_sites = 16
    site_anchors = np.zeros((num_sites, 3), dtype=np.float32)
    site_tangents = np.zeros((num_sites, 3), dtype=np.float32)
    site_curves = np.zeros(num_sites, dtype=np.float32)
    site_speeds = np.zeros(num_sites, dtype=np.float32)
    site_phases = rng.uniform(0.0, 12.0, size=num_sites).astype(np.float32)

    for s in range(num_sites):
        if s < 10:
            # Place major dramatic flare jets along visible limbs & upper corona crown
            lat = rng.uniform(-0.25, 1.05)
            side = 1.0 if (s % 2 == 0) else -1.0
            lon = side * rng.uniform(0.75, 1.85) + rng.normal(0.0, 0.22)
        else:
            lat = rng.uniform(-0.65, 0.85)
            lon = rng.uniform(0.0, 2.0 * math.pi)
        anc = normalize(np.array([
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
            math.cos(lat) * math.cos(lon),
        ], dtype=np.float32))
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(anc[1]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        t1 = normalize(np.cross(ref, anc))
        t2 = normalize(np.cross(anc, t1))
        ang = rng.uniform(0.0, 2.0 * math.pi)
        tang = normalize(math.cos(ang) * t1 + math.sin(ang) * t2)

        site_anchors[s] = anc
        site_tangents[s] = tang
        site_curves[s] = rng.uniform(0.45, 1.45) * rng.choice([-1.0, 1.0])
        site_speeds[s] = rng.uniform(0.95, 2.45)

    site_ids = rng.integers(0, num_sites, size=count)
    # Tight magnetic flux-tube collimation around each active eruption footpoint
    jitter = rng.normal(0.0, 0.028, size=(count, 3)).astype(np.float32)
    anchors = site_anchors[site_ids] + jitter
    anchors /= np.maximum(np.linalg.norm(anchors, axis=1, keepdims=True), 1e-6)

    tangents = site_tangents[site_ids] + rng.normal(0.0, 0.075, size=(count, 3)).astype(np.float32)
    # Re-orthogonalize tangent against anchor
    dot_at = np.sum(tangents * anchors, axis=1, keepdims=True)
    tangents = tangents - dot_at * anchors
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6)

    # Burst clustering in phase so flares visibly surge in episodic plasma waves
    burst_offset = rng.normal(0.0, 0.55, size=count).astype(np.float32)
    phases = (site_phases[site_ids] + burst_offset + rng.uniform(0.0, 5.5, size=count) * (rng.random(count) < 0.28)).astype(np.float32)
    durations = rng.uniform(2.4, 5.6, size=count).astype(np.float32)
    speeds = (site_speeds[site_ids] * rng.uniform(0.65, 1.35, size=count)).astype(np.float32)
    curves = (site_curves[site_ids] * rng.uniform(0.80, 1.25, size=count)).astype(np.float32)

    sizes = np.exp(rng.uniform(math.log(0.055), math.log(0.22), size=count)).astype(np.float32)
    temps = np.clip(rng.beta(2.4, 1.7, size=count), 0.05, 1.0).astype(np.float32)
    helix_amps = rng.uniform(0.03, 0.22, size=count).astype(np.float32)

    data = np.empty((count, 13), dtype=np.float32)
    data[:, 0:3] = anchors
    data[:, 3:6] = tangents
    data[:, 6] = phases
    data[:, 7] = durations
    data[:, 8] = speeds
    data[:, 9] = curves
    data[:, 10] = sizes
    data[:, 11] = temps
    data[:, 12] = helix_amps
    return data


def spectral_color_from_class(spec_t: np.ndarray) -> np.ndarray:
    """
    Map normalized stellar temperature parameter t in [0..1] to realistic HDR
    spectral RGB colors:
        0.00..0.22 -> M/K red & deep orange dwarfs
        0.22..0.48 -> G-type warm yellow solar-type stars
        0.48..0.72 -> F/A luminous pure-white & pale-ivory stars
        0.72..1.00 -> B/O scorching sapphire & electric blue-white supergiants
    """
    knots_x = np.array([0.00, 0.16, 0.36, 0.56, 0.76, 0.90, 1.00], dtype=np.float32)
    knots_r = np.array([1.25, 1.30, 1.25, 1.15, 0.85, 0.62, 0.52], dtype=np.float32)
    knots_g = np.array([0.38, 0.66, 0.96, 1.10, 0.98, 0.88, 0.82], dtype=np.float32)
    knots_b = np.array([0.16, 0.28, 0.58, 1.05, 1.38, 1.55, 1.72], dtype=np.float32)

    t = np.clip(spec_t, 0.0, 1.0)
    r = np.interp(t, knots_x, knots_r)
    g = np.interp(t, knots_x, knots_g)
    b = np.interp(t, knots_x, knots_b)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def generate_spiral_galaxy_stars(
    rng: np.random.Generator,
    count: int = GALAXY_STAR_COUNT,
    max_radius: float = GALAXY_RADIUS,
) -> np.ndarray:
    """
    Generate 65,000+ 3D spiral galaxy stars organized along logarithmic spiral arms,
    inter-arm spurs, a dense glowing ellipsoidal galactic core/bar, and outer halo.
    Layout (10 floats per star):
        cyl(3: radius, base_angle, local_y), color(3), props(4: size, phase, freq, lum)
    """
    n_core = int(count * 0.22)
    n_arms = int(count * 0.66)
    n_halo = count - n_core - n_arms

    radii = np.empty(count, dtype=np.float32)
    angles = np.empty(count, dtype=np.float32)
    heights = np.empty(count, dtype=np.float32)
    spec_t = np.empty(count, dtype=np.float32)
    sizes = np.empty(count, dtype=np.float32)
    lums = np.empty(count, dtype=np.float32)

    # --- 1. Dense Galactic Bulge & Barred Core (Population II warm gold/white stars) ---
    u_c = rng.random(n_core)
    r_core = (max_radius * 0.24) * (u_c ** 1.65)
    ang_core = rng.uniform(0.0, 2.0 * math.pi, size=n_core)
    # Slight central bar elongation along angle = 0.35 rad
    bar_ang = 0.35
    x_c = r_core * np.cos(ang_core) * 1.28
    z_c = r_core * np.sin(ang_core) * 0.78
    # Rotate bar
    cb, sb = math.cos(bar_ang), math.sin(bar_ang)
    rx = cb * x_c - sb * z_c
    rz = sb * x_c + cb * z_c
    radii[:n_core] = np.sqrt(rx * rx + rz * rz)
    angles[:n_core] = np.arctan2(rz, rx)
    # Spheroidal vertical bulge thickness
    bulge_h = 1.65 * np.exp(-radii[:n_core] / (max_radius * 0.14)) + 0.20
    heights[:n_core] = rng.normal(0.0, 1.0, size=n_core) * bulge_h
    spec_t[:n_core] = np.clip(rng.normal(0.38, 0.18, size=n_core), 0.05, 0.72)
    sizes[:n_core] = np.exp(rng.uniform(math.log(0.11), math.log(0.38), size=n_core))
    lums[:n_core] = rng.uniform(0.40, 1.40, size=n_core) * (1.25 - 0.35 * (radii[:n_core] / (max_radius * 0.24)))

    # --- 2. Logarithmic Spiral Arms (2 grand-design primary arms + 2 secondary spurs) ---
    s0 = n_core
    s1 = n_core + n_arms
    # Arm selection weights: two dominant opposite arms + two fainter inter-arms
    arm_probs = np.array([0.34, 0.34, 0.16, 0.16], dtype=np.float64)
    arm_ids = rng.choice(4, size=n_arms, p=arm_probs)
    arm_base_angles = np.array([0.0, math.pi, math.pi * 0.52, math.pi * 1.52], dtype=np.float32)

    # Radial density profile: exponential disk peaking near inner-mid arms
    u_r = rng.random(n_arms)
    r_arm = 1.1 + (max_radius - 1.1) * (u_r ** 1.16)

    # Logarithmic spiral winding: theta = base + k * ln(1 + r / r0)
    winding = 2.65
    r0_scale = 2.05
    spiral_angle = arm_base_angles[arm_ids] + winding * np.log(1.0 + r_arm / r0_scale)

    # Add realistic arm width dispersion + occasional clumpy OB star-forming associations
    is_ob_knot = rng.random(n_arms) < 0.26
    angular_spread = np.where(
        is_ob_knot,
        rng.normal(0.0, 0.13, size=n_arms),
        rng.normal(0.0, 0.32 + 0.14 * (r_arm / max_radius), size=n_arms),
    )
    # Slight radial ripple for filamentary spiral spurs
    r_arm = np.clip(r_arm + rng.normal(0.0, 0.45, size=n_arms), 0.65, max_radius * 1.05)

    radii[s0:s1] = r_arm
    angles[s0:s1] = spiral_angle + angular_spread

    # Realistic vertical galactic disk scale height (thin midplane, slight outer flare)
    disk_scale_h = 0.22 + 0.58 * np.exp(-r_arm / 4.8) + 0.32 * ((r_arm / max_radius) ** 2)
    disk_scale_h = np.where(is_ob_knot, disk_scale_h * 0.50, disk_scale_h)
    heights[s0:s1] = rng.normal(0.0, 1.0, size=n_arms) * disk_scale_h

    # Spectral class distribution: young hot O/B blue-white supergiants hug arm ridges;
    # older G/K/M stars fill wider disk
    rad_frac = np.clip(r_arm / max_radius, 0.0, 1.0)
    spec_arm = np.where(
        is_ob_knot,
        rng.uniform(0.74, 1.00, size=n_arms),
        np.clip(rng.normal(0.46 + 0.25 * rad_frac, 0.28, size=n_arms), 0.0, 1.0),
    )
    spec_t[s0:s1] = spec_arm

    # Power-law luminosity & size distribution (rare super-luminous O/B beacons)
    is_giant = rng.random(n_arms) < 0.055
    sizes[s0:s1] = np.where(
        is_giant,
        rng.uniform(0.44, 0.96, size=n_arms),
        np.exp(rng.uniform(math.log(0.11), math.log(0.38), size=n_arms)),
    )
    lums[s0:s1] = np.where(
        is_giant,
        rng.uniform(1.65, 3.60, size=n_arms),
        rng.uniform(0.32, 1.25, size=n_arms) * (0.78 + 0.48 * spec_arm),
    )

    # --- 3. Extended Galactic Disk & Halo Globular Clusters ---
    r_halo = max_radius * (rng.random(n_halo) ** 0.72) * 1.10
    ang_halo = rng.uniform(0.0, 2.0 * math.pi, size=n_halo)
    h_halo = rng.normal(0.0, 1.0, size=n_halo) * (0.78 + 2.1 * np.exp(-r_halo / 9.0))
    radii[s1:] = r_halo
    angles[s1:] = ang_halo
    heights[s1:] = h_halo
    spec_t[s1:] = np.clip(rng.normal(0.44, 0.26, size=n_halo), 0.0, 1.0)
    sizes[s1:] = np.exp(rng.uniform(math.log(0.10), math.log(0.34), size=n_halo))
    lums[s1:] = rng.uniform(0.22, 0.92, size=n_halo)

    colors = spectral_color_from_class(spec_t)
    twinkle_phase = rng.uniform(0.0, 2.0 * math.pi, size=count).astype(np.float32)
    twinkle_freq = rng.uniform(0.8, 3.8, size=count).astype(np.float32)

    data = np.empty((count, 10), dtype=np.float32)
    data[:, 0] = radii
    data[:, 1] = angles
    data[:, 2] = heights
    data[:, 3:6] = colors
    data[:, 6] = sizes
    data[:, 7] = twinkle_phase
    data[:, 8] = twinkle_freq
    data[:, 9] = lums
    return data


def generate_nebula_clouds(
    rng: np.random.Generator,
    count: int = NEBULA_CLOUD_COUNT,
    max_radius: float = GALAXY_RADIUS,
) -> np.ndarray:
    """
    Generate multi-colored volumetric nebula billboards swirling along the
    logarithmic spiral arms: deep indigo, violet, teal, magenta, and fiery amber.
    Layout (10 floats per instance):
        cyl(3: radius, base_angle, local_y), color(3), props(4: size, rot, seed, opacity)
    """
    # Rich astrophysical nebula emission/reflection palette (vivid deep-space chromatic balance)
    palette = np.array([
        [0.18, 0.14, 1.18],  # Deep Cosmic Indigo
        [0.64, 0.14, 1.22],  # Luminous Ultraviolet / Violet
        [0.04, 1.02, 1.06],  # Ionized O-III Cyan / Teal
        [1.26, 0.12, 0.62],  # H-Alpha Magenta / Rose
        [1.45, 0.56, 0.10],  # Fiery Inner-Arm Amber / Gold
    ], dtype=np.float32)

    arm_ids = rng.choice(4, size=count, p=[0.35, 0.35, 0.15, 0.15])
    arm_base = np.array([0.0, math.pi, math.pi * 0.52, math.pi * 1.52], dtype=np.float32)

    r = 1.5 + (max_radius * 0.94 - 1.5) * (rng.random(count) ** 1.06)
    winding = 2.65
    r0_scale = 2.05
    ang = arm_base[arm_ids] + winding * np.log(1.0 + r / r0_scale) + rng.normal(0.0, 0.20, size=count)
    ly = rng.normal(0.0, 0.45, size=count) * (0.45 + 0.70 * np.exp(-r / 7.0))

    # Spatially coherent color selection along arms: amber & magenta near inner arms,
    # vibrant teal, violet, magenta, and deep indigo swirling prominently through mid/outer arms
    rad_frac = np.clip((r - 1.5) / (max_radius * 0.94 - 1.5), 0.0, 1.0)
    colors = np.empty((count, 3), dtype=np.float32)
    for i in range(count):
        rf = float(rad_frac[i])
        if rf < 0.26:
            probs = [0.06, 0.14, 0.10, 0.28, 0.42]  # Fiery amber & magenta near inner arms
        elif rf < 0.62:
            probs = [0.18, 0.22, 0.28, 0.22, 0.10]  # Vibrant full chromatic swirl with bold teal/magenta
        else:
            probs = [0.30, 0.24, 0.34, 0.10, 0.02]  # Deep indigo, cosmic violet, and glowing teal outer arms
        c_idx = int(rng.choice(5, p=probs))
        tint = palette[c_idx] * rng.uniform(0.84, 1.24, size=3)
        colors[i] = tint

    sizes = (1.65 + 2.55 * (rad_frac ** 0.55)) * rng.uniform(0.68, 1.38, size=count)
    rots = rng.uniform(0.0, 2.0 * math.pi, size=count).astype(np.float32)
    seeds = rng.uniform(0.0, 100.0, size=count).astype(np.float32)
    opacities = (0.062 + 0.048 * (1.0 - 0.35 * rad_frac)) * rng.uniform(0.68, 1.32, size=count)

    data = np.empty((count, 10), dtype=np.float32)
    data[:, 0] = r
    data[:, 1] = ang
    data[:, 2] = ly
    data[:, 3:6] = colors
    data[:, 6] = sizes
    data[:, 7] = rots
    data[:, 8] = seeds
    data[:, 9] = opacities
    return data


def generate_dust_lanes(
    rng: np.random.Generator,
    count: int = DUST_LANE_COUNT,
    max_radius: float = GALAXY_RADIUS,
) -> np.ndarray:
    """
    Generate dark silhouetted interstellar dust lanes tracing the inner concave
    density-wave shock edges of the spiral arms.
    Layout (10 floats per instance):
        cyl(3: radius, base_angle, local_y), tint(3), props(4: size, rot, seed, optical_depth)
    """
    arm_ids = rng.choice(4, size=count, p=[0.38, 0.38, 0.12, 0.12])
    arm_base = np.array([0.0, math.pi, math.pi * 0.52, math.pi * 1.52], dtype=np.float32)

    r = 1.4 + (max_radius * 0.86 - 1.4) * (rng.random(count) ** 1.12)
    winding = 2.65
    r0_scale = 2.05
    # Offset angle tightly behind the bright arm crest to trace realistic dark dust lanes
    ang = arm_base[arm_ids] + winding * np.log(1.0 + r / r0_scale) - 0.17 + rng.normal(0.0, 0.12, size=count)
    # Dust settles tightly near the thin galactic midplane
    ly = rng.normal(0.0, 0.22, size=count)

    # Dark umber-charcoal absorption tint with faint warm reddening
    tints = np.stack([
        rng.uniform(0.04, 0.10, size=count),
        rng.uniform(0.015, 0.04, size=count),
        rng.uniform(0.025, 0.055, size=count),
    ], axis=-1).astype(np.float32)

    rad_frac = np.clip(r / max_radius, 0.0, 1.0)
    sizes = (1.25 + 1.65 * (rad_frac ** 0.5)) * rng.uniform(0.65, 1.30, size=count)
    rots = rng.uniform(0.0, 2.0 * math.pi, size=count).astype(np.float32)
    seeds = rng.uniform(0.0, 100.0, size=count).astype(np.float32)
    optical_depth = rng.uniform(0.42, 0.72, size=count) * (1.0 - 0.28 * rad_frac)

    data = np.empty((count, 10), dtype=np.float32)
    data[:, 0] = r
    data[:, 1] = ang
    data[:, 2] = ly
    data[:, 3:6] = tints
    data[:, 6] = sizes
    data[:, 7] = rots
    data[:, 8] = seeds
    data[:, 9] = optical_depth
    return data


def generate_deep_field_stars(
    rng: np.random.Generator,
    count: int = DEEP_STAR_COUNT,
) -> np.ndarray:
    """
    Generate 9,000 far background stars including 12 dense deep-field star clusters
    scattered across the celestial sphere (R = 140..250).
    Layout (10 floats per star):
        pos(3), color(3), props(4: size, phase, freq, lum)
    """
    n_clusters = 12
    n_cluster_stars = int(count * 0.24)
    n_field_stars = count - n_cluster_stars

    # Uniform points on sphere for general deep-field sky
    z = rng.uniform(-1.0, 1.0, size=n_field_stars)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n_field_stars)
    rad_xy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    dist_f = rng.uniform(145.0, 245.0, size=n_field_stars)
    pos_f = np.stack([rad_xy * np.cos(phi), z, rad_xy * np.sin(phi)], axis=-1) * dist_f[:, None]

    # Deep-field star clusters (rich globular & open clusters scattered mostly in visible backdrop)
    cluster_dirs = np.zeros((n_clusters, 3), dtype=np.float32)
    for c in range(n_clusters):
        # Bias slightly toward -Z sky hemisphere behind sun & galaxy so many are framed on launch
        cz = rng.uniform(-0.85, 0.90)
        cphi = rng.uniform(-math.pi * 0.95, math.pi * 0.95)
        rxy = math.sqrt(max(1.0 - cz * cz, 0.0))
        d = np.array([rxy * math.sin(cphi), cz, -abs(rxy * math.cos(cphi)) if c < 8 else rxy * math.cos(cphi)], dtype=np.float32)
        cluster_dirs[c] = normalize(d)

    c_ids = rng.integers(0, n_clusters, size=n_cluster_stars)
    c_spread = rng.normal(0.0, 0.036, size=(n_cluster_stars, 3))
    c_vec = cluster_dirs[c_ids] + c_spread
    c_vec /= np.maximum(np.linalg.norm(c_vec, axis=1, keepdims=True), 1e-6)
    dist_c = rng.uniform(155.0, 230.0, size=n_cluster_stars)
    pos_c = c_vec * dist_c[:, None]

    pos = np.vstack([pos_f, pos_c]).astype(np.float32)
    spec_t = np.clip(rng.normal(0.55, 0.28, size=count), 0.0, 1.0).astype(np.float32)
    colors = spectral_color_from_class(spec_t)

    is_bright = rng.random(count) < 0.035
    sizes = np.where(is_bright, rng.uniform(2.8, 5.8, size=count), rng.uniform(1.1, 2.7, size=count)).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=count).astype(np.float32)
    freqs = rng.uniform(0.6, 3.2, size=count).astype(np.float32)
    lums = np.where(is_bright, rng.uniform(0.95, 1.95, size=count), rng.uniform(0.18, 0.72, size=count)).astype(np.float32)

    data = np.empty((count, 10), dtype=np.float32)
    data[:, 0:3] = pos
    data[:, 3:6] = colors
    data[:, 6] = sizes
    data[:, 7] = phases
    data[:, 8] = freqs
    data[:, 9] = lums
    return data


def generate_distant_galaxies(
    rng: np.random.Generator,
    count: int = DISTANT_GALAXY_COUNT,
) -> np.ndarray:
    """
    Scatter dozens of tiny distant spiral and elliptical background galaxies
    across the deep celestial sphere for immense cosmic scale.
    Layout (14 floats per instance):
        pos(3), color(3), shape(4: size, rot, tilt, type), detail(4: arms, winding, brightness, seed)
    """
    gal_palette = np.array([
        [0.62, 0.78, 1.25],  # Distant blue spiral
        [0.92, 0.72, 1.18],  # Violet-tinted spiral
        [1.15, 0.88, 0.56],  # Golden elliptical / lenticular
        [0.58, 0.96, 1.08],  # Teal-blue starburst spiral
        [1.18, 0.68, 0.78],  # Warm rose-gold spiral
    ], dtype=np.float32)

    pos = np.empty((count, 3), dtype=np.float32)
    colors = np.empty((count, 3), dtype=np.float32)
    shape = np.empty((count, 4), dtype=np.float32)
    detail = np.empty((count, 4), dtype=np.float32)

    for i in range(count):
        if i < int(count * 0.72):
            # Populate the primary background hemisphere richly so dozens sparkle in frame
            az = rng.uniform(-1.35, 1.35)
            el = rng.uniform(-0.75, 0.92)
            d_vec = np.array([
                math.cos(el) * math.sin(az),
                math.sin(el),
                -math.cos(el) * math.cos(az),
            ], dtype=np.float32)
        else:
            z = rng.uniform(-1.0, 1.0)
            phi = rng.uniform(0.0, 2.0 * math.pi)
            rxy = math.sqrt(max(1.0 - z * z, 0.0))
            d_vec = np.array([rxy * math.cos(phi), z, rxy * math.sin(phi)], dtype=np.float32)

        dist = float(rng.uniform(150.0, 235.0))
        pos[i] = normalize(d_vec) * dist

        gal_type = float(rng.random())  # <0.68 spiral, >=0.68 elliptical
        if gal_type >= 0.68:
            base_col = gal_palette[2] * rng.uniform(0.85, 1.15, size=3)
        else:
            base_col = gal_palette[int(rng.integers(0, len(gal_palette)))] * rng.uniform(0.85, 1.15, size=3)
        colors[i] = base_col

        size = float(np.exp(rng.uniform(math.log(2.2), math.log(7.2))))
        rot = float(rng.uniform(0.0, 2.0 * math.pi))
        tilt = float(rng.uniform(0.26, 0.92))
        shape[i] = (size, rot, tilt, gal_type)

        arm_count = float(rng.choice([2.0, 2.0, 3.0, 4.0]))
        winding = float(rng.uniform(2.2, 4.6) * rng.choice([-1.0, 1.0]))
        brightness = float(rng.uniform(0.32, 0.88))
        seed = float(rng.random())
        detail[i] = (arm_count, winding, brightness, seed)

    return np.hstack([pos, colors, shape, detail]).astype(np.float32)


# =============================================================================
# REAL-TIME OPENGL 3.3+ RENDERING ENGINE & HDR POST-PROCESSING PIPELINE
# =============================================================================

class CosmicSceneRenderer:
    """
    Encapsulates all OpenGL 3.3+ shaders, GPU geometry buffers, RGBA16F HDR
    framebuffer, multi-pass separable Gaussian bloom pyramid, and render passes.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        star_count: int = GALAXY_STAR_COUNT,
        seed: int = 42,
    ) -> None:
        self.ctx = ctx
        self.width = width
        self.height = height
        self.star_count = star_count
        self.rng = np.random.default_rng(seed)

        # Rendering controls
        self.bloom_presets = [0.46, 0.72, 0.26]
        self.bloom_preset_idx = 0
        self.exposure = 1.02

        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.gc_mode = "auto"

        # Precompute galaxy world transform
        self.galaxy_model = build_galaxy_model_matrix()
        self.galaxy_model_bytes = mat4_to_bytes(self.galaxy_model)

        self._compile_shaders()
        self._build_scene_geometry()
        self._build_post_framebuffers(width, height)

    def _compile_shaders(self) -> None:
        c = self.ctx
        self.prog_sun = c.program(vertex_shader=SUN_VERT_SHADER, fragment_shader=SUN_FRAG_SHADER)
        self.prog_corona_shell = c.program(vertex_shader=CORONA_SHELL_VERT_SHADER, fragment_shader=CORONA_SHELL_FRAG_SHADER)
        self.prog_corona_vol = c.program(vertex_shader=CORONA_VOL_VERT_SHADER, fragment_shader=CORONA_VOL_FRAG_SHADER)
        self.prog_coronal_loops = c.program(vertex_shader=CORONAL_LOOP_VERT_SHADER, fragment_shader=CORONAL_LOOP_FRAG_SHADER)
        self.prog_solar_flares = c.program(vertex_shader=SOLAR_FLARE_VERT_SHADER, fragment_shader=SOLAR_FLARE_FRAG_SHADER)

        self.prog_galaxy_stars = c.program(vertex_shader=GALAXY_STAR_VERT_SHADER, fragment_shader=GALAXY_STAR_FRAG_SHADER)
        self.prog_nebula = c.program(vertex_shader=NEBULA_VERT_SHADER, fragment_shader=NEBULA_FRAG_SHADER)
        self.prog_dust = c.program(vertex_shader=DUST_VERT_SHADER, fragment_shader=DUST_FRAG_SHADER)
        self.prog_galaxy_core = c.program(vertex_shader=GALAXY_CORE_VERT_SHADER, fragment_shader=GALAXY_CORE_FRAG_SHADER)

        self.prog_deep_stars = c.program(vertex_shader=DEEP_STAR_VERT_SHADER, fragment_shader=DEEP_STAR_FRAG_SHADER)
        self.prog_distant_gal = c.program(vertex_shader=DISTANT_GALAXY_VERT_SHADER, fragment_shader=DISTANT_GALAXY_FRAG_SHADER)

        self.prog_bloom_prefilter = c.program(vertex_shader=POST_VERT_SHADER, fragment_shader=BLOOM_PREFILTER_FRAG_SHADER)
        self.prog_bloom_blur = c.program(vertex_shader=POST_VERT_SHADER, fragment_shader=BLOOM_BLUR_FRAG_SHADER)
        self.prog_composite = c.program(vertex_shader=POST_VERT_SHADER, fragment_shader=POST_COMPOSITE_FRAG_SHADER)

    def _build_scene_geometry(self) -> None:
        c = self.ctx

        # Shared unit billboard quad (two triangles in [-1, 1]^2)
        quad_verts = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
        ], dtype=np.float32)
        self.vbo_quad = c.buffer(quad_verts.tobytes())

        # Fullscreen post-processing quad
        self.vao_post_prefilter = c.vertex_array(self.prog_bloom_prefilter, [(self.vbo_quad, "2f", "in_pos")])
        self.vao_post_blur = c.vertex_array(self.prog_bloom_blur, [(self.vbo_quad, "2f", "in_pos")])
        self.vao_post_composite = c.vertex_array(self.prog_composite, [(self.vbo_quad, "2f", "in_pos")])

        # 1. High-subdivision 3D Sun icosphere (20,480 triangles)
        sun_verts, sun_indices = generate_icosphere(SUN_ICOSPHERE_SUBDIVISIONS)
        self.vbo_sun = c.buffer(sun_verts.tobytes())
        self.ibo_sun = c.buffer(sun_indices.tobytes())
        self.vao_sun = c.vertex_array(
            self.prog_sun,
            [(self.vbo_sun, "3f", "in_position")],
            index_buffer=self.ibo_sun,
        )

        # 2. Concentric 3D Corona atmosphere shells (5,120 triangles)
        shell_verts, shell_indices = generate_icosphere(CORONA_SHELL_SUBDIVISIONS)
        self.vbo_corona_shell = c.buffer(shell_verts.tobytes())
        self.ibo_corona_shell = c.buffer(shell_indices.tobytes())
        self.vao_corona_shell = c.vertex_array(
            self.prog_corona_shell,
            [(self.vbo_corona_shell, "3f", "in_position")],
            index_buffer=self.ibo_corona_shell,
        )

        # 3. Volumetric ray-integrated corona halo billboard
        self.vao_corona_vol = c.vertex_array(self.prog_corona_vol, [(self.vbo_quad, "2f", "in_quad")])

        # 4. Dynamic 3D twisting magnetic coronal loop spline tubes
        loop_verts, loop_indices = generate_coronal_loops(
            self.rng, SUN_CENTER, SUN_RADIUS,
            loop_count=CORONAL_LOOP_COUNT,
            length_segs=CORONAL_LOOP_LENGTH_SEGS,
            radial_segs=CORONAL_LOOP_RADIAL_SEGS,
        )
        self.vbo_loops = c.buffer(loop_verts.tobytes())
        self.ibo_loops = c.buffer(loop_indices.tobytes())
        self.vao_loops = c.vertex_array(
            self.prog_coronal_loops,
            [(self.vbo_loops, "3f 3f 3f 1f 1f 1f 4f",
              "in_center", "in_radial", "in_tangent", "in_u", "in_v", "in_radius", "in_params")],
            index_buffer=self.ibo_loops,
        )

        # 5. Erupting solar flare plasma particle jets
        flare_data = generate_solar_flare_particles(self.rng, SOLAR_FLARE_PARTICLE_COUNT)
        self.vbo_flares = c.buffer(flare_data.tobytes())
        self.vao_flares = c.vertex_array(
            self.prog_solar_flares,
            [(self.vbo_flares, "3f 3f 4f 3f", "in_anchor", "in_tangent", "in_params", "in_visual")],
        )

        # 6. Sprawling 3D Spiral Galaxy stars (65,000+ stars)
        # Split draw order: first 76% drawn beneath dark dust lanes, remaining 24% sparkling above
        galaxy_stars = generate_spiral_galaxy_stars(self.rng, self.star_count, GALAXY_RADIUS)
        # Shuffle so foreground/background split samples uniformly across arms & bulge
        perm = self.rng.permutation(self.star_count)
        galaxy_stars = np.ascontiguousarray(galaxy_stars[perm])
        self.galaxy_star_split = int(self.star_count * 0.76)
        self.vbo_galaxy_stars = c.buffer(galaxy_stars.tobytes())
        self.vao_galaxy_stars = c.vertex_array(
            self.prog_galaxy_stars,
            [(self.vbo_galaxy_stars, "3f 3f 4f", "in_cyl", "in_color", "in_props")],
        )

        # 7. Multi-colored volumetric nebula clouds (instanced billboards)
        nebula_data = generate_nebula_clouds(self.rng, NEBULA_CLOUD_COUNT, GALAXY_RADIUS)
        self.nebula_count = nebula_data.shape[0]
        self.vbo_nebula = c.buffer(nebula_data.tobytes())
        self.vao_nebula = c.vertex_array(
            self.prog_nebula,
            [
                (self.vbo_quad, "2f", "in_quad"),
                (self.vbo_nebula, "3f 3f 4f/i", "in_cyl", "in_color", "in_props"),
            ],
        )

        # 8. Dark silhouetted interstellar dust lanes (instanced alpha billboards)
        dust_data = generate_dust_lanes(self.rng, DUST_LANE_COUNT, GALAXY_RADIUS)
        self.dust_count = dust_data.shape[0]
        self.vbo_dust = c.buffer(dust_data.tobytes())
        self.vao_dust = c.vertex_array(
            self.prog_dust,
            [
                (self.vbo_quad, "2f", "in_quad"),
                (self.vbo_dust, "3f 3f 4f/i", "in_cyl", "in_tint", "in_props"),
            ],
        )

        # 9. Glowing galactic core volumetric halo
        self.vao_galaxy_core = c.vertex_array(self.prog_galaxy_core, [(self.vbo_quad, "2f", "in_quad")])

        # 10. Deep-field background stars & clusters
        deep_stars = generate_deep_field_stars(self.rng, DEEP_STAR_COUNT)
        self.vbo_deep_stars = c.buffer(deep_stars.tobytes())
        self.vao_deep_stars = c.vertex_array(
            self.prog_deep_stars,
            [(self.vbo_deep_stars, "3f 3f 4f", "in_pos", "in_color", "in_props")],
        )

        # 11. Distant procedural spiral & elliptical background galaxies
        dist_gal = generate_distant_galaxies(self.rng, DISTANT_GALAXY_COUNT)
        self.distant_gal_count = dist_gal.shape[0]
        self.vbo_distant_gal = c.buffer(dist_gal.tobytes())
        self.vao_distant_gal = c.vertex_array(
            self.prog_distant_gal,
            [
                (self.vbo_quad, "2f", "in_quad"),
                (self.vbo_distant_gal, "3f 3f 4f 4f/i", "in_pos", "in_color", "in_shape", "in_detail"),
            ],
        )

    def _release_post_framebuffers(self) -> None:
        for attr in ("hdr_fbo", "hdr_color_tex", "hdr_depth_rb"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()
        for lst_attr in ("bloom_fbo_a", "bloom_fbo_b", "bloom_tex_a", "bloom_tex_b"):
            objs = getattr(self, lst_attr, [])
            for obj in objs:
                obj.release()

    def _build_post_framebuffers(self, width: int, height: int) -> None:
        self.width = max(width, 64)
        self.height = max(height, 64)
        c = self.ctx

        # Primary RGBA16F floating-point HDR framebuffer with 24-bit depth buffer
        self.hdr_color_tex = c.texture((self.width, self.height), components=4, dtype="f2")
        self.hdr_color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hdr_color_tex.repeat_x = False
        self.hdr_color_tex.repeat_y = False
        self.hdr_depth_rb = c.depth_renderbuffer((self.width, self.height))
        self.hdr_fbo = c.framebuffer(color_attachments=[self.hdr_color_tex], depth_attachment=self.hdr_depth_rb)

        # 4-level multi-scale RGBA16F ping-pong Gaussian bloom pyramid (1/2, 1/4, 1/8, 1/16)
        self.bloom_sizes: List[Tuple[int, int]] = []
        self.bloom_tex_a: List[moderngl.Texture] = []
        self.bloom_tex_b: List[moderngl.Texture] = []
        self.bloom_fbo_a: List[moderngl.Framebuffer] = []
        self.bloom_fbo_b: List[moderngl.Framebuffer] = []

        bw, bh = self.width, self.height
        for _ in range(4):
            bw = max(bw // 2, 16)
            bh = max(bh // 2, 16)
            self.bloom_sizes.append((bw, bh))

            ta = c.texture((bw, bh), components=4, dtype="f2")
            ta.filter = (moderngl.LINEAR, moderngl.LINEAR)
            ta.repeat_x = False
            ta.repeat_y = False

            tb = c.texture((bw, bh), components=4, dtype="f2")
            tb.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tb.repeat_x = False
            tb.repeat_y = False

            self.bloom_tex_a.append(ta)
            self.bloom_tex_b.append(tb)
            self.bloom_fbo_a.append(c.framebuffer(color_attachments=[ta]))
            self.bloom_fbo_b.append(c.framebuffer(color_attachments=[tb]))

    def resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height:
            return
        self._release_post_framebuffers()
        self._build_post_framebuffers(width, height)

    def cycle_bloom(self) -> float:
        self.bloom_preset_idx = (self.bloom_preset_idx + 1) % len(self.bloom_presets)
        return self.bloom_presets[self.bloom_preset_idx]

    def render(self, camera: CameraRig, sim_time: float, real_time: float) -> None:
        c = self.ctx
        aspect = float(self.width) / float(max(self.height, 1))
        proj = mat4_perspective(camera.fov_y, aspect, camera.near, camera.far)
        view, eye = camera.update(0.0, paused=True)  # Read current view/eye (camera updated in main loop)

        view_bytes = mat4_to_bytes(view)
        proj_bytes = mat4_to_bytes(proj)
        cam_pos_t = (float(eye[0]), float(eye[1]), float(eye[2]))
        cam_right_t = (float(camera.right[0]), float(camera.right[1]), float(camera.right[2]))
        cam_up_t = (float(camera.up[0]), float(camera.up[1]), float(camera.up[2]))
        sun_center_t = (float(SUN_CENTER[0]), float(SUN_CENTER[1]), float(SUN_CENTER[2]))

        # =====================================================================
        # STAGE 1: RENDER FULL 3D HDR SCENE TO FLOATING-POINT RGBA16F FBO
        # =====================================================================
        self.hdr_fbo.use()
        c.viewport = (0, 0, self.width, self.height)
        c.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        # --- Pass 1A: Opaque Displaced 3D Sun Photosphere (writes exact 3D depth) ---
        c.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        c.disable(moderngl.BLEND)
        self.hdr_fbo.depth_mask = True

        self.prog_sun["u_view"].write(view_bytes)
        self.prog_sun["u_proj"].write(proj_bytes)
        self.prog_sun["u_sun_center"].value = sun_center_t
        self.prog_sun["u_sun_radius"].value = SUN_RADIUS
        self.prog_sun["u_time"].value = sim_time
        self.prog_sun["u_cam_pos"].value = cam_pos_t
        self.vao_sun.render(moderngl.TRIANGLES)

        # --- Pass 1B: Deep-Field Background Stars, Clusters & Distant Galaxies ---
        # Depth test ON (hdr_fbo.depth_mask OFF) so the foreground 3D Sun cleanly occludes backdrop
        c.disable(moderngl.CULL_FACE)
        c.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.hdr_fbo.depth_mask = False
        c.blend_func = (moderngl.ONE, moderngl.ONE)

        self.prog_deep_stars["u_view"].write(view_bytes)
        self.prog_deep_stars["u_proj"].write(proj_bytes)
        self.prog_deep_stars["u_time"].value = sim_time
        self.prog_deep_stars["u_viewport_h"].value = float(self.height)
        self.vao_deep_stars.render(moderngl.POINTS)

        self.prog_distant_gal["u_view"].write(view_bytes)
        self.prog_distant_gal["u_proj"].write(proj_bytes)
        self.prog_distant_gal["u_cam_right"].value = cam_right_t
        self.prog_distant_gal["u_cam_up"].value = cam_up_t
        self.prog_distant_gal["u_time"].value = sim_time
        self.vao_distant_gal.render(moderngl.TRIANGLES, instances=self.distant_gal_count)

        # --- Pass 1C: Sprawling 3D Spiral Galaxy (Core + Volumetric Nebulae + Stars) ---
        # 1) Dense Glowing Galactic Core Volumetric Halo
        self.prog_galaxy_core["u_model"].write(self.galaxy_model_bytes)
        self.prog_galaxy_core["u_view"].write(view_bytes)
        self.prog_galaxy_core["u_proj"].write(proj_bytes)
        self.prog_galaxy_core["u_cam_right"].value = cam_right_t
        self.prog_galaxy_core["u_cam_up"].value = cam_up_t
        self.prog_galaxy_core["u_core_radius"].value = 8.4
        self.prog_galaxy_core["u_time"].value = sim_time
        self.vao_galaxy_core.render(moderngl.TRIANGLES)

        # 2) Multi-Colored Volumetric Nebula Clouds (indigo, violet, teal, magenta, amber)
        self.prog_nebula["u_model"].write(self.galaxy_model_bytes)
        self.prog_nebula["u_view"].write(view_bytes)
        self.prog_nebula["u_proj"].write(proj_bytes)
        self.prog_nebula["u_cam_right"].value = cam_right_t
        self.prog_nebula["u_cam_up"].value = cam_up_t
        self.prog_nebula["u_time"].value = sim_time
        self.vao_nebula.render(moderngl.TRIANGLES, instances=self.nebula_count)

        # 3) Background half of the 65,000+ Keplerian Spiral Galaxy Stars
        self.prog_galaxy_stars["u_model"].write(self.galaxy_model_bytes)
        self.prog_galaxy_stars["u_view"].write(view_bytes)
        self.prog_galaxy_stars["u_proj"].write(proj_bytes)
        self.prog_galaxy_stars["u_time"].value = sim_time
        self.prog_galaxy_stars["u_viewport_h"].value = float(self.height)
        self.vao_galaxy_stars.render(moderngl.POINTS, vertices=self.galaxy_star_split, first=0)

        # 4) Dark Silhouetted Interstellar Dust Lanes (alpha absorption across arms & core)
        c.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.prog_dust["u_model"].write(self.galaxy_model_bytes)
        self.prog_dust["u_view"].write(view_bytes)
        self.prog_dust["u_proj"].write(proj_bytes)
        self.prog_dust["u_cam_right"].value = cam_right_t
        self.prog_dust["u_cam_up"].value = cam_up_t
        self.prog_dust["u_time"].value = sim_time
        self.vao_dust.render(moderngl.TRIANGLES, instances=self.dust_count)

        # 5) Foreground half of the Spiral Galaxy Stars sparkling above the dust lanes
        c.blend_func = (moderngl.ONE, moderngl.ONE)
        self.vao_galaxy_stars.render(
            moderngl.POINTS,
            vertices=self.star_count - self.galaxy_star_split,
            first=self.galaxy_star_split,
        )

        # --- Pass 1D: Multi-Layered Animated Solar Corona, Magnetic Loops & Flare Jets ---
        # 1) Concentric 3D Displaced Volumetric Corona Shells
        self.prog_corona_shell["u_view"].write(view_bytes)
        self.prog_corona_shell["u_proj"].write(proj_bytes)
        self.prog_corona_shell["u_sun_center"].value = sun_center_t
        self.prog_corona_shell["u_cam_pos"].value = cam_pos_t
        self.prog_corona_shell["u_time"].value = sim_time
        for layer_idx, scale in enumerate((1.055, 1.13, 1.24)):
            self.prog_corona_shell["u_shell_radius"].value = SUN_RADIUS * scale
            self.prog_corona_shell["u_layer_id"].value = float(layer_idx)
            self.vao_corona_shell.render(moderngl.TRIANGLES)

        # 2) Analytical Ray-Integrated Volumetric Corona Halo & Radial Streamers
        c.disable(moderngl.DEPTH_TEST)
        self.prog_corona_vol["u_view"].write(view_bytes)
        self.prog_corona_vol["u_proj"].write(proj_bytes)
        self.prog_corona_vol["u_sun_center"].value = sun_center_t
        self.prog_corona_vol["u_cam_pos"].value = cam_pos_t
        self.prog_corona_vol["u_cam_right"].value = cam_right_t
        self.prog_corona_vol["u_cam_up"].value = cam_up_t
        self.prog_corona_vol["u_sun_radius"].value = SUN_RADIUS
        self.prog_corona_vol["u_billboard_radius"].value = SUN_RADIUS * 3.55
        self.prog_corona_vol["u_time"].value = sim_time
        self.vao_corona_vol.render(moderngl.TRIANGLES)

        # 3) Dynamic Twisting 3D Magnetic Coronal Loop Spline Tubes (depth-occluded by Sun!)
        c.enable(moderngl.DEPTH_TEST)
        self.prog_coronal_loops["u_view"].write(view_bytes)
        self.prog_coronal_loops["u_proj"].write(proj_bytes)
        self.prog_coronal_loops["u_sun_center"].value = sun_center_t
        self.prog_coronal_loops["u_cam_pos"].value = cam_pos_t
        self.prog_coronal_loops["u_time"].value = sim_time
        self.vao_loops.render(moderngl.TRIANGLES)

        # 4) Erupting Solar Flare Plasma Particle Jets along curved magnetic field lines
        self.prog_solar_flares["u_view"].write(view_bytes)
        self.prog_solar_flares["u_proj"].write(proj_bytes)
        self.prog_solar_flares["u_sun_center"].value = sun_center_t
        self.prog_solar_flares["u_sun_radius"].value = SUN_RADIUS
        self.prog_solar_flares["u_time"].value = sim_time
        self.prog_solar_flares["u_viewport_h"].value = float(self.height)
        self.vao_flares.render(moderngl.POINTS)

        # =====================================================================
        # STAGE 2: MULTI-PASS PYRAMID SEPARABLE GAUSSIAN BLOOM (RGBA16F)
        # =====================================================================
        c.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.hdr_fbo.depth_mask = True

        # Prefilter HDR scene bright highlights into level 0 (1/2 res)
        bw0, bh0 = self.bloom_sizes[0]
        self.bloom_fbo_a[0].use()
        c.viewport = (0, 0, bw0, bh0)
        self.hdr_color_tex.use(location=0)
        self.prog_bloom_prefilter["u_scene_tex"].value = 0
        self.prog_bloom_prefilter["u_threshold"].value = 1.18
        self.prog_bloom_prefilter["u_knee"].value = 0.58
        self.vao_post_prefilter.render(moderngl.TRIANGLES)

        # Progressive downsample + multi-iteration separable Gaussian blur across 4 mip octaves
        self.prog_bloom_blur["u_tex"].value = 0
        for level in range(4):
            bw, bh = self.bloom_sizes[level]
            c.viewport = (0, 0, bw, bh)

            if level > 0:
                # Downsample from previous level's blurred output with an initial horizontal blur
                self.bloom_fbo_b[level].use()
                self.bloom_tex_a[level - 1].use(location=0)
                self.prog_bloom_blur["u_direction"].value = (1.0 / float(bw), 0.0)
                self.vao_post_blur.render(moderngl.TRIANGLES)

                self.bloom_fbo_a[level].use()
                self.bloom_tex_b[level].use(location=0)
                self.prog_bloom_blur["u_direction"].value = (0.0, 1.0 / float(bh))
                self.vao_post_blur.render(moderngl.TRIANGLES)

            # Two full separable H+V 13-tap Gaussian passes per mip for silky wide-radius bloom
            iters = 2 if level >= 1 else 1
            for _ in range(iters):
                self.bloom_fbo_b[level].use()
                self.bloom_tex_a[level].use(location=0)
                self.prog_bloom_blur["u_direction"].value = (1.0 / float(bw), 0.0)
                self.vao_post_blur.render(moderngl.TRIANGLES)

                self.bloom_fbo_a[level].use()
                self.bloom_tex_b[level].use(location=0)
                self.prog_bloom_blur["u_direction"].value = (0.0, 1.0 / float(bh))
                self.vao_post_blur.render(moderngl.TRIANGLES)

        # =====================================================================
        # STAGE 3: FINAL COMPOSITE (ACES FILMIC + CHROMATIC ABERRATION + GRAIN)
        # =====================================================================
        c.screen.use()
        c.viewport = (0, 0, self.width, self.height)
        self.hdr_color_tex.use(location=0)
        self.bloom_tex_a[0].use(location=1)
        self.bloom_tex_a[1].use(location=2)
        self.bloom_tex_a[2].use(location=3)
        self.bloom_tex_a[3].use(location=4)

        self.prog_composite["u_scene_tex"].value = 0
        self.prog_composite["u_bloom_0"].value = 1
        self.prog_composite["u_bloom_1"].value = 2
        self.prog_composite["u_bloom_2"].value = 3
        self.prog_composite["u_bloom_3"].value = 4
        self.prog_composite["u_bloom_strength"].value = self.bloom_presets[self.bloom_preset_idx]
        self.prog_composite["u_exposure"].value = self.exposure
        self.prog_composite["u_time"].value = real_time
        self.prog_composite["u_resolution"].value = (float(self.width), float(self.height))
        self.vao_post_composite.render(moderngl.TRIANGLES)

    def save_screenshot(self, filepath: Optional[str] = None) -> str:
        """Read back the default OpenGL framebuffer and write a crisp PNG image."""
        if filepath is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"cosmic_sun_galaxy_{ts}.png"

        raw = self.ctx.screen.read(components=3, alignment=1)
        if PILImage is not None:
            img = PILImage.frombytes("RGB", (self.width, self.height), raw)
            img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
            img.save(filepath, format="PNG")
        else:
            surf = pygame.image.fromstring(raw, (self.width, self.height), "RGB", True)
            pygame.image.save(surf, filepath)
        return filepath


# =============================================================================
# APPLICATION ENTRY POINT & INTERACTIVE EVENT LOOP
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time OpenGL 3.3+ 3D Blazing Sun & Spiral Galaxy (9:16 Portrait)"
    )
    parser.add_argument("--hd", action="store_true", help="Launch in Full HD 1080x1920 9:16 portrait resolution")
    parser.add_argument("--width", type=int, default=0, help="Custom window width (overrides default)")
    parser.add_argument("--height", type=int, default=0, help="Custom window height (overrides default)")
    parser.add_argument("--stars", type=int, default=GALAXY_STAR_COUNT, help="Spiral galaxy star count (default: 65000)")
    parser.add_argument("--seed", type=int, default=42, help="Procedural cosmos random seed")
    parser.add_argument("--screenshot", type=str, default="", help="Render one frame to PNG path and exit")
    parser.add_argument("--time", type=float, default=2.4, help="Simulation timestamp for --screenshot mode")
    parser.add_argument("--preset", type=int, default=1, choices=(1, 2, 3), help="Initial camera preset (1-3)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    width = args.width if args.width > 0 else (HD_WIDTH if args.hd else DEFAULT_WIDTH)
    height = args.height if args.height > 0 else (HD_HEIGHT if args.hd else DEFAULT_HEIGHT)

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    pygame.display.set_mode((width, height), flags)
    pygame.display.set_caption("Cosmic Sun & Spiral Galaxy — Initializing Shaders...")

    ctx = moderngl.create_context(require=330)
    renderer = CosmicSceneRenderer(ctx, width, height, star_count=max(args.stars, 50000), seed=args.seed)

    camera = CameraRig()
    camera.apply_preset(args.preset)

    # Headless/batch single-frame high-res capture mode
    if args.screenshot:
        camera.set_cinematic_time(args.time)
        camera.update(0.016, paused=True)
        renderer.render(camera, sim_time=args.time, real_time=args.time)
        pygame.display.flip()
        out_path = renderer.save_screenshot(args.screenshot)
        print(f"Saved screenshot to {out_path} ({width}x{height})")
        pygame.quit()
        return

    print("=" * 72)
    print(f"Cosmic Sun & Spiral Galaxy | {width}x{height} (9:16) | OpenGL {ctx.version_code}")
    print(f"Renderer: {ctx.info.get('GL_RENDERER', 'Unknown')} | Galaxy Stars: {renderer.star_count:,}")
    print("Controls: Left-Drag Orbit | Right-Drag Pan | Wheel Zoom | SPACE Pause")
    print("          C Cinematic/Manual | R Reset | 1/2/3 Presets | S/F12 Screenshot | B Bloom")
    print("=" * 72)

    clock = pygame.time.Clock()
    running = True
    paused = False
    sim_time = 0.0
    real_time = 0.0
    dragging_left = False
    dragging_right = False
    title_timer = 0.0

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        real_time += dt
        if not paused:
            sim_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = max(event.w, 128), max(event.h, 128)
                pygame.display.set_mode((width, height), flags)
                renderer.resize(width, height)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging_left = True
                elif event.button in (2, 3):
                    dragging_right = True
                elif event.button == 4:
                    camera.zoom(1.0)
                elif event.button == 5:
                    camera.zoom(-1.0)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_left = False
                elif event.button in (2, 3):
                    dragging_right = False
            elif event.type == pygame.MOUSEMOTION:
                dx, dy = event.rel
                if dragging_left:
                    camera.drag_orbit(dx, dy)
                elif dragging_right:
                    camera.drag_pan(dx, dy)
            elif event.type == pygame.MOUSEWHEEL:
                camera.zoom(float(event.y))
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_c:
                    camera.cinematic_mode = not camera.cinematic_mode
                elif event.key == pygame.K_r:
                    camera.reset()
                elif event.key == pygame.K_1:
                    camera.apply_preset(1)
                elif event.key == pygame.K_2:
                    camera.apply_preset(2)
                elif event.key == pygame.K_3:
                    camera.apply_preset(3)
                elif event.key == pygame.K_b:
                    bloom_val = renderer.cycle_bloom()
                    print(f"[Bloom] Intensity set to {bloom_val:.2f}")
                elif event.key in (pygame.K_s, pygame.K_F12):
                    saved = renderer.save_screenshot()
                    print(f"[Screenshot] Saved high-res PNG -> {saved}")

        camera.update(dt, paused=paused)
        renderer.render(camera, sim_time=sim_time, real_time=real_time)
        pygame.display.flip()

        title_timer += dt
        if title_timer >= 0.35:
            title_timer = 0.0
            fps = clock.get_fps()
            mode_str = "CINEMATIC" if camera.cinematic_mode else "MANUAL"
            state_str = "PAUSED" if paused else "LIVE"
            pygame.display.set_caption(
                f"Cosmic Sun & Spiral Galaxy | {fps:5.1f} FPS | {mode_str} | {state_str} | "
                f"[SPACE] Pause  [C] Cam  [R] Reset  [S] PNG"
            )

    pygame.quit()


if __name__ == "__main__":
    main()
