"""
GooferProceduralClip — Cinematic procedural motion graphics from goof data.

Generates animated frames using PIL/Pillow with stylized aesthetics.
No GPU required — pure CPU frame generation with cinematic post-processing.

Forked from DMM_ProceduralClip by Jeffrey A. Brick.
Adapted for movie goof visualization instead of weather/seismic data.

Styles:
  goof_neon      — dark bg, neon grid, film reel silhouettes, goof readouts,
                   chromatic aberration, film grain, bloom, animated gauges
  goof_minimal   — clean dark panels, typography-focused, subtle grain,
                   breathing dividers, goof category fade transitions
  goof_retro     — green-on-black terminal / hacker aesthetic,
                   CRT warp, phosphor bloom, flicker, "GOOF DETECTED" readout

Parses goof descriptions and movie metadata to render animated overlays
showing goof categories, counts, and descriptions as kinetic typography.

Author: Jeffrey A. Brick
"""

from __future__ import annotations

import logging
import math
import re
import time

import numpy as np
import torch

log = logging.getLogger("Goofer.ProceduralClip")


# ── Color palettes ────────────────────────────────────────────────────
NEON = {
    "bg":       (10, 10, 26),
    "grid":     (25, 25, 60),
    "pink":     (255, 20, 147),
    "cyan":     (0, 229, 255),
    "amber":    (255, 176, 0),
    "purple":   (180, 0, 255),
    "white":    (220, 220, 230),
    "dim":      (80, 80, 120),
    "red":      (255, 50, 50),
    "green":    (50, 255, 100),
    "film":     (40, 35, 20),
    "hot":      (255, 60, 60),
    "cool":     (60, 160, 255),
}

# ── Category colors ───────────────────────────────────────────────────
_CAT_COLORS = {
    "Continuity":         NEON["cyan"],
    "Factual Error":      NEON["amber"],
    "Revealing Mistake":  NEON["pink"],
    "Anachronism":        NEON["purple"],
    "Audio/Visual Unsync": NEON["red"],
    "Crew Visible":       NEON["green"],
    "Boom Mic Visible":   NEON["green"],
    "Plot Hole":          NEON["hot"],
    "Character Error":    NEON["cool"],
    "Geography Error":    NEON["amber"],
    "Miscellaneous":      NEON["dim"],
}


# ── Helpers (from DMM_ProceduralClip) ─────────────────────────────────
def _font(size: int):
    """Load a monospace font with cross-platform fallbacks."""
    from PIL import ImageFont
    for path in [
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\lucon.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 2


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def _typewriter(text: str, progress: float) -> str:
    n = int(len(text) * min(max(progress, 0.0), 1.0))
    return text[:n]


# ── AI goof labeler ───────────────────────────────────────────────────
# Produces 4-word summaries using a transformers pipeline (T5-small,
# ~240 MB, downloaded once to the HuggingFace cache on first run).
# Falls back to smart keyword extraction if the model is unavailable.

_SUMMARIZER = None          # lazy-loaded, one per ComfyUI session
_SUMMARIZER_READY = False   # set to False permanently if load fails

# Stop words for the keyword-extraction fallback
_STOP = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its", "this",
    "that", "these", "those", "i", "we", "you", "he", "she", "they", "their",
    "our", "my", "his", "her", "as", "when", "where", "which", "who", "what",
    "how", "so", "if", "then", "there", "here", "not", "no", "up", "out",
    "about", "into", "during", "before", "after", "between", "also", "just",
    "more", "than", "very", "too", "such", "same", "each", "both", "few",
    "many", "most", "other", "some", "any", "all", "while", "however",
    "although", "because", "since", "though", "even", "clearly", "visible",
    "scene", "shot", "film", "movie", "can", "see", "seen", "shows", "shown",
    "appears", "appear", "around", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten", "through", "across",
})


def _get_summarizer():
    """Lazy-load a T5-small summarization pipeline (once per session)."""
    global _SUMMARIZER, _SUMMARIZER_READY
    if _SUMMARIZER is not None:
        return _SUMMARIZER
    if _SUMMARIZER_READY is False and _SUMMARIZER is None:
        # first-ever call — try to load
        try:
            from transformers import pipeline as hf_pipeline
            log.info("[ProceduralClip] Loading T5-small summarizer...")
            _SUMMARIZER = hf_pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-6-6",
                device=-1,          # CPU only — GPU needed for LTX
                truncation=True,
            )
            _SUMMARIZER_READY = True
            log.info("[ProceduralClip] Summarizer ready")
        except Exception as e:
            log.warning("[ProceduralClip] LLM summarizer unavailable (%s) "
                        "— using keyword extraction", e)
            _SUMMARIZER_READY = None   # None = permanently failed, skip retry
    return _SUMMARIZER


def _keyword_label(text: str, max_words: int = 4) -> str:
    """Extractive keyword summary: drops stop words, keeps content words."""
    words = re.findall(r'\b[A-Za-z][a-zA-Z\'\.]*\b', text)
    seen: set = set()
    result = []
    for w in words:
        lw = w.lower().rstrip(".")
        if lw not in _STOP and len(lw) > 2 and lw not in seen:
            seen.add(lw)
            result.append(w)
            if len(result) >= max_words:
                break
    if not result:
        # absolute fallback: first max_words of original
        result = text.split()[:max_words]
    return " ".join(result[:max_words]) + ("..." if len(text.split()) > max_words else "")


def _ai_label(text: str, max_words: int = 4) -> str:
    """AI-summarize a goof description to max_words for on-screen display.

    Uses a DistilBART summarization pipeline when available (downloads
    ~300 MB once to the HuggingFace cache on first use).  Falls back to
    keyword extraction so the node always produces output.
    """
    if not text or not text.strip():
        return text

    summarizer = _get_summarizer()
    if summarizer is not None:
        try:
            out = summarizer(
                text,
                max_length=12,
                min_length=3,
                do_sample=False,
            )
            summary = out[0].get("summary_text", "").strip()
            if summary:
                words = summary.split()
                label = " ".join(words[:max_words])
                if len(words) > max_words:
                    label += "..."
                log.debug("[ProceduralClip] AI label: '%s' → '%s'", text[:40], label)
                return label
        except Exception as e:
            log.debug("[ProceduralClip] Summarizer inference failed: %s", e)

    # Keyword extraction fallback
    return _keyword_label(text, max_words)


def _glow_text(draw, xy, text, font, color, glow_radius=2, intensity=0.33):
    x, y = xy
    for dx in range(-glow_radius, glow_radius + 1):
        for dy in range(-glow_radius, glow_radius + 1):
            if dx == 0 and dy == 0:
                continue
            dist = abs(dx) + abs(dy)
            if dist <= glow_radius:
                falloff = 1.0 - dist / (glow_radius + 1)
                g = tuple(max(0, int(c * intensity * falloff)) for c in color)
                draw.text((x + dx, y + dy), text, font=font, fill=g)
    draw.text((x, y), text, font=font, fill=color)


def _dim(color, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _breathing(t, freq=1.0, lo=0.7, hi=1.0):
    v = (math.sin(t * freq * math.pi * 2) + 1.0) / 2.0
    return lo + v * (hi - lo)


def _smooth_noise(x, y, seed=0):
    def _hash(ix, iy, s):
        n = int(ix * 374761393 + iy * 668265263 + s * 1274126177) & 0x7fffffff
        n = (n ^ (n >> 13)) * 1103515245
        return ((n ^ (n >> 16)) & 0xffff) / 65535.0
    ix, iy = int(math.floor(x)), int(math.floor(y))
    fx, fy = x - ix, y - iy
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    n00 = _hash(ix, iy, seed)
    n10 = _hash(ix + 1, iy, seed)
    n01 = _hash(ix, iy + 1, seed)
    n11 = _hash(ix + 1, iy + 1, seed)
    nx0 = n00 + (n10 - n00) * fx
    nx1 = n01 + (n11 - n01) * fx
    return nx0 + (nx1 - nx0) * fy


def _fbm(x, y, octaves=4, seed=0):
    value = 0.0
    amplitude = 0.5
    frequency = 1.0
    for _ in range(octaves):
        value += amplitude * _smooth_noise(x * frequency, y * frequency, seed)
        amplitude *= 0.5
        frequency *= 2.0
    return value


def _spring_overshoot(t, damping=5.0, freq=8.0):
    if t <= 0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return 1.0 - math.exp(-damping * t) * math.cos(freq * t)


# ── Cellular Automata ─────────────────────────────────────────────────
class _CellularAutomata:
    def __init__(self, cols, rows, seed=42):
        rng = np.random.RandomState(seed)
        self.grid = (rng.random((rows, cols)) < 0.18).astype(np.uint8)
        self.rows = rows
        self.cols = cols

    def step(self):
        g = self.grid
        n = (np.roll(g, 1, 0) + np.roll(g, -1, 0) +
             np.roll(g, 1, 1) + np.roll(g, -1, 1) +
             np.roll(np.roll(g, 1, 0), 1, 1) +
             np.roll(np.roll(g, 1, 0), -1, 1) +
             np.roll(np.roll(g, -1, 0), 1, 1) +
             np.roll(np.roll(g, -1, 0), -1, 1))
        self.grid = ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)

    def render(self, w, h, color, intensity=0.08):
        color_arr = np.array(color, dtype=np.float32) * intensity
        small = self.grid[:, :, np.newaxis].astype(np.float32) * color_arr
        cell_h = max(1, h // self.rows)
        cell_w = max(1, w // self.cols)
        overlay = np.repeat(np.repeat(small, cell_h, axis=0), cell_w, axis=1)
        oh, ow = overlay.shape[:2]
        if oh < h or ow < w:
            padded = np.zeros((h, w, 3), dtype=np.float32)
            padded[:min(oh, h), :min(ow, w), :] = overlay[:min(oh, h), :min(ow, w), :]
            overlay = padded
        else:
            overlay = overlay[:h, :w, :]
        return overlay


# ── Mandelbrot border accent ──────────────────────────────────────────
def _mandelbrot_line(y_norm, x_start, x_end, steps=80, max_iter=20):
    points = []
    cy = (y_norm - 0.5) * 2.5
    for i in range(steps):
        x_norm = x_start + (x_end - x_start) * i / steps
        cx = (x_norm - 0.5) * 3.5 - 0.5
        z = complex(0, 0)
        c = complex(cx, cy)
        escape = 0
        for n in range(max_iter):
            if abs(z) > 2.0:
                escape = n / max_iter
                break
            z = z * z + c
        points.append((x_norm, escape))
    return points


# ── Post-processing ───────────────────────────────────────────────────
def _apply_vignette(img_arr, strength=0.4):
    h, w = img_arr.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    max_dist = math.sqrt(cx * cx + cy * cy)
    y_coords = np.arange(h).reshape(-1, 1)
    x_coords = np.arange(w).reshape(1, -1)
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    vignette = 1.0 - strength * (dist / max_dist) ** 1.8
    vignette = np.clip(vignette, 0, 1)
    result = img_arr.astype(np.float32)
    for c in range(3):
        result[:, :, c] *= vignette
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_chromatic_aberration(img_arr, offset=2):
    if offset == 0:
        return img_arr
    h, w = img_arr.shape[:2]
    result = img_arr.copy()
    result[:, offset:, 0] = img_arr[:, :-offset, 0]
    result[:, :w - offset, 2] = img_arr[:, offset:, 2]
    return result


def _apply_film_grain(img_arr, intensity=12, frame_idx=0):
    rng = np.random.RandomState(seed=(frame_idx * 7 + 31) & 0x7fffffff)
    noise = rng.randint(-intensity, intensity + 1,
                        size=img_arr.shape, dtype=np.int16)
    result = img_arr.astype(np.int16) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_bloom(img_arr, threshold=200, radius=8, intensity=0.3):
    bright = img_arr.astype(np.float32)
    mask = np.max(bright, axis=2) > threshold
    bloom_layer = np.zeros_like(bright)
    bloom_layer[mask] = bright[mask]
    from PIL import Image, ImageFilter
    bloom_img = Image.fromarray(bloom_layer.astype(np.uint8))
    bloom_img = bloom_img.filter(ImageFilter.GaussianBlur(radius=radius))
    bloom_arr = np.array(bloom_img, dtype=np.float32)
    result = img_arr.astype(np.float32) + bloom_arr * intensity
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Film reel decoration ──────────────────────────────────────────────
def _draw_film_sprockets(draw, w, h, frame, fps):
    """Draw film strip sprocket holes along edges — movie theme."""
    sprocket_h = max(4, int(h * 0.025))
    sprocket_w = max(6, int(w * 0.012))
    gap = max(12, int(h * 0.06))
    color = (35, 30, 20)
    border_color = (55, 50, 35)

    # Scroll offset for animation
    scroll = int(frame * 1.5) % gap

    # Left strip
    strip_w = int(w * 0.04)
    draw.rectangle([0, 0, strip_w, h], fill=(20, 18, 12))
    for y in range(-gap + scroll, h + gap, gap):
        sx = (strip_w - sprocket_w) // 2
        draw.rounded_rectangle(
            [sx, y, sx + sprocket_w, y + sprocket_h],
            radius=2, fill=color, outline=border_color
        )

    # Right strip
    rx = w - strip_w
    draw.rectangle([rx, 0, w, h], fill=(20, 18, 12))
    for y in range(-gap + scroll, h + gap, gap):
        sx = rx + (strip_w - sprocket_w) // 2
        draw.rounded_rectangle(
            [sx, y, sx + sprocket_w, y + sprocket_h],
            radius=2, fill=color, outline=border_color
        )


# ══════════════════════════════════════════════════════════════════════
#  Node
# ══════════════════════════════════════════════════════════════════════
class GooferProceduralClip:
    """Generate cinematic procedural goof-visualization motion graphics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("GOOFER_CONFIG",),
                "movie_data": ("GOOFER_MOVIE",),
                "goofs_data": ("GOOFER_GOOFS",),
                "width": ("INT", {"default": 768, "min": 128, "max": 3840, "step": 32}),
                "height": ("INT", {"default": 512, "min": 128, "max": 2160, "step": 32}),
                "duration_sec": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 60}),
                "style": (["goof_neon", "goof_minimal", "goof_retro"],
                          {"default": "goof_neon"}),
            },
            "optional": {
                "music": ("AUDIO", {
                    "tooltip": "Background music from GooferBackgroundMusic — "
                               "trimmed/looped to match clip duration automatically."
                }),
                "music_duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 300.0, "step": 0.5,
                    "tooltip": "Wire from GooferBackgroundMusic 'duration_sec' output "
                               "to sync this clip's length exactly to the music."
                }),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "Goofer"

    def generate(self, config, movie_data, goofs_data,
                 width, height, duration_sec, fps, style, music=None, music_duration=0.0):
        from PIL import Image
        import random

        # If the music node passed its duration, use that instead of the widget
        if music_duration is not None and float(music_duration) > 0.0:
            duration_sec = float(music_duration)
            log.info("[GooferProceduralClip] duration_sec overridden by music_duration=%.2f",
                     duration_sec)

        start_time = time.time()
        total_frames = int(duration_sec * fps)

        # Extract display data from goofs
        data = self._build_display_data(movie_data, goofs_data)

        # Scale fonts
        s = height / 512.0
        fonts = {
            "title":   _font(int(44 * s)),
            "heading": _font(int(26 * s)),
            "body":    _font(int(18 * s)),
            "small":   _font(int(13 * s)),
            "tiny":    _font(int(10 * s)),
            "goof_num": _font(int(60 * s)),
        }

        # Deterministic particles
        rng = random.Random(config.get("seed", 42))
        particles = [
            {
                "x": rng.random(), "y": rng.random(),
                "speed": rng.uniform(0.3, 1.5),
                "color": rng.choice([NEON["pink"], NEON["cyan"], NEON["amber"], NEON["purple"]]),
                "size": rng.uniform(0.5, 2.0),
                "phase": rng.random() * math.pi * 2,
            }
            for _ in range(40)
        ]

        # Cellular automata
        ca_cols = max(16, width // 24)
        ca_rows = max(12, height // 24)
        ca = _CellularAutomata(ca_cols, ca_rows, seed=config.get("seed", 42))

        # Fractal accents
        fractal_top = _mandelbrot_line(0.35, 0.0, 1.0, steps=width // 4)
        fractal_bot = _mandelbrot_line(0.65, 0.0, 1.0, steps=width // 4)

        # Render frames
        frames = []
        for f in range(total_frames):
            t = f / max(total_frames - 1, 1)

            if f % 3 == 0 and f > 0:
                ca.step()

            if style == "goof_neon":
                img = self._goof_neon(width, height, t, f, fps, data, fonts,
                                      particles, ca, fractal_top, fractal_bot)
            elif style == "goof_minimal":
                img = self._goof_minimal(width, height, t, f, fps, data, fonts)
            else:
                img = self._goof_retro(width, height, t, f, fps, data, fonts, ca)

            arr = np.array(img, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr))

        frame_tensor = torch.stack(frames, dim=0)

        # Audio — use supplied music or fall back to silence
        if music is not None:
            try:
                waveform = music["waveform"]          # [1, C, S]
                sample_rate = music.get("sample_rate", 48000)
                target = int(duration_sec * sample_rate)
                if waveform.shape[-1] == 0:
                    raise ValueError("empty waveform")
                if waveform.shape[-1] < target:
                    # loop until long enough, then trim
                    repeats = (target // waveform.shape[-1]) + 2
                    waveform = waveform.repeat(1, 1, repeats)[..., :target]
                else:
                    waveform = waveform[..., :target]
                audio = {"waveform": waveform, "sample_rate": sample_rate}
                log.info("[GooferProceduralClip] music attached (%d samples @ %dHz)",
                         target, sample_rate)
            except Exception as e:
                log.warning("[GooferProceduralClip] music attach failed (%s) — using silence", e)
                n_samples = int(duration_sec * 48000)
                audio = {"waveform": torch.zeros(1, 2, n_samples), "sample_rate": 48000}
        else:
            n_samples = int(duration_sec * 48000)
            audio = {"waveform": torch.zeros(1, 2, n_samples), "sample_rate": 48000}

        video = self._to_video(frame_tensor, audio, fps)

        elapsed = time.time() - start_time
        log.info("[GooferProceduralClip] %d frames %dx%d '%s' in %.1fs",
                 total_frames, width, height, style, elapsed)
        return (video,)

    def _to_video(self, images, audio, fps):
        from nodes import NODE_CLASS_MAPPINGS
        cls = NODE_CLASS_MAPPINGS.get("CreateVideo")
        if cls is None:
            raise RuntimeError("CreateVideo node not found — is ComfyUI-LTXVideo installed?")
        obj = cls()
        fn = getattr(cls, "FUNCTION", "execute")
        result = getattr(obj, fn)(images=images, audio=audio, fps=fps)
        return result.args[0] if hasattr(result, "args") else result[0]

    def _build_display_data(self, movie_data, goofs_data):
        """Build structured display data from movie + goofs."""
        title = movie_data.get("title", "UNKNOWN FILM").upper()
        year = movie_data.get("year", "????")
        genres = movie_data.get("genres", [])
        genre_str = " / ".join(genres[:3]) if genres else "FILM"
        rating = movie_data.get("rating", 0)

        goof_count = len(goofs_data)
        categories = {}
        goof_lines = []
        for g in goofs_data[:5]:
            cat = g.get("category", "Miscellaneous")
            desc = g.get("description", "")
            categories[cat] = categories.get(cat, 0) + 1
            # AI-summarize to 6 words for on-screen display
            desc = _ai_label(desc, max_words=6)
            goof_lines.append((cat, desc))

        return {
            "title": title,
            "year": str(year),
            "genre_str": genre_str,
            "rating": rating,
            "goof_count": goof_count,
            "categories": categories,
            "goof_lines": goof_lines,
        }

    # ══════════════════════════════════════════════════════════════════
    #  GOOF NEON style
    # ══════════════════════════════════════════════════════════════════
    def _goof_neon(self, w, h, t, frame, fps, data, fonts,
                   particles, ca, fractal_top, fractal_bot):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (w, h), NEON["bg"])
        draw = ImageDraw.Draw(img)
        time_sec = frame / fps

        # BG gradient
        ratio = np.linspace(0, 1, h, dtype=np.float32).reshape(-1, 1)
        r_ch = np.clip(8 + ratio * 6, 0, 255).astype(np.uint8)
        g_ch = np.clip(6 + ratio * 8, 0, 255).astype(np.uint8)
        b_ch = np.clip(22 + ratio * 18, 0, 255).astype(np.uint8)
        grad = np.concatenate([
            np.broadcast_to(r_ch[:, :, np.newaxis], (h, w, 1)),
            np.broadcast_to(g_ch[:, :, np.newaxis], (h, w, 1)),
            np.broadcast_to(b_ch[:, :, np.newaxis], (h, w, 1)),
        ], axis=2)
        img = Image.fromarray(grad)
        draw = ImageDraw.Draw(img)

        # Cellular automata background
        if ca is not None:
            ca_fade = _ease_out(min(t / 0.5, 1.0))
            if ca_fade > 0:
                ca_overlay = ca.render(w, h, NEON["cyan"], intensity=0.03 * ca_fade)
                img_arr = np.array(img, dtype=np.float32)
                img_arr += ca_overlay
                img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
                draw = ImageDraw.Draw(img)

        # Film strip sprocket borders
        _draw_film_sprockets(draw, w, h, frame, fps)

        # Content area (between sprocket strips)
        margin = int(w * 0.06)

        # ── Big goof count number ──
        count_raw = min(max(t - 0.05, 0) / 0.2, 1.0)
        count_p = _spring_overshoot(count_raw, damping=4.0, freq=10.0)
        if count_raw > 0:
            num_text = str(data["goof_count"])
            glow_i = _breathing(time_sec, freq=0.3, lo=0.2, hi=0.4)
            num_x = margin + int(w * 0.02)
            num_y = int(h * 0.06) + int((1.0 - count_p) * h * 0.03)
            _glow_text(draw, (num_x, num_y), num_text, fonts["goof_num"],
                       NEON["pink"], glow_radius=5, intensity=glow_i)
            # "GOOFS FOUND" label
            label_y = num_y + int(65 * h / 512)
            draw.text((num_x, label_y), "GOOFS FOUND",
                      font=fonts["small"], fill=_dim(NEON["dim"], count_p))

        # ── Title with spring ──
        title_raw = min(max(t - 0.12, 0) / 0.2, 1.0)
        title_p = _spring_overshoot(title_raw, damping=4.0, freq=10.0)
        if title_raw > 0:
            display = _typewriter(data["title"], min(title_raw * 1.5, 1.0))
            tx = margin + int(w * 0.20)
            base_ty = int(h * 0.07)
            spring_off = int((1.0 - title_p) * h * 0.03)
            ty = base_ty + spring_off
            glow_i = _breathing(time_sec, freq=0.4, lo=0.25, hi=0.45)
            _glow_text(draw, (tx, ty), display, fonts["title"],
                       NEON["cyan"], glow_radius=4, intensity=glow_i)

        # ── Year / Genre subtitle ──
        sub_raw = min(max(t - 0.22, 0) / 0.15, 1.0)
        if sub_raw > 0:
            sub = f"{data['year']}  ·  {data['genre_str']}"
            display = _typewriter(sub, min(sub_raw * 1.3, 1.0))
            sx = margin + int(w * 0.20)
            sy = int(h * 0.07) + int(50 * h / 512)
            draw.text((sx, sy), display, font=fonts["heading"],
                      fill=_dim(NEON["dim"], sub_raw))

        # ── Goof panels ──
        panel_start_t = 0.30
        px = margin + int(w * 0.02)
        py = int(h * 0.35)
        pw = int(w * 0.85)
        ph = int(h * 0.10)
        row_gap = int(h * 0.115)

        for idx, (cat, desc) in enumerate(data["goof_lines"][:5]):
            entry_t = panel_start_t + idx * 0.08
            pp = _ease_out(min(max(t - entry_t, 0) / 0.15, 1.0))
            if pp <= 0:
                continue

            by = py + idx * row_gap
            color = _CAT_COLORS.get(cat, NEON["dim"])

            # Panel background
            bg_c = _dim(color, 0.04 * pp)
            draw.rectangle([px, by, px + pw, by + ph], fill=bg_c)

            # Panel border with breathing
            border_b = _breathing(time_sec, freq=0.3 + idx * 0.1, lo=0.5, hi=1.0)
            bc = _dim(color, pp * border_b)
            draw.rectangle([px, by, px + pw, by + ph], outline=bc, width=1)

            # Corner accents
            cl = int(pw * 0.04)
            for cx, cy, ddx, ddy in [
                (px, by, 1, 1), (px + pw, by, -1, 1),
                (px, by + ph, 1, -1), (px + pw, by + ph, -1, -1),
            ]:
                draw.line([(cx, cy), (cx + cl * ddx, cy)], fill=bc, width=2)
                draw.line([(cx, cy), (cx, cy + cl * ddy)], fill=bc, width=2)

            # Goof number + category
            goof_label = f"GOOF #{idx + 1}  ·  {cat.upper()}"
            draw.text((px + 8, by + 3),
                      _typewriter(goof_label, pp),
                      font=fonts["small"], fill=color)

            # Description
            text_p = min(max((pp - 0.3) / 0.7, 0), 1.0)
            if text_p > 0:
                dy = by + int(ph * 0.45)
                draw.text((px + 8, dy),
                          _typewriter(desc, text_p),
                          font=fonts["body"], fill=NEON["white"])

        # ── Floating particles ──
        for p in particles:
            pt = (t * p["speed"]) % 1.0
            noise_dx = _fbm(p["x"] * 4, time_sec * 0.3 + p["phase"], octaves=2) * 0.06 - 0.03
            noise_dy = _fbm(p["y"] * 4 + 100, time_sec * 0.25 + p["phase"], octaves=2) * 0.04 - 0.02
            x = int((p["x"] + noise_dx) % 1.0 * w)
            y = int((p["y"] - pt * 0.3 + noise_dy) % 1.0 * h)
            alpha = 0.3 + 0.5 * math.sin(time_sec * 2.5 + p["phase"])
            dot_c = _dim(p["color"], alpha)
            sz = max(1, int(p["size"] * h / 512))
            draw.ellipse([x - sz, y - sz, x + sz, y + sz], fill=dot_c)

        # ── Corner brackets ──
        bf = min(t / 0.2, 1.0)
        if bf > 0:
            bl = int(min(w, h) * 0.04)
            bc = _dim(NEON["dim"], bf * _breathing(time_sec, freq=0.15))
            m = int(min(w, h) * 0.025)
            for cx, cy, ddx, ddy in [
                (m, m, 1, 1), (w - m, m, -1, 1),
                (m, h - m, 1, -1), (w - m, h - m, -1, -1),
            ]:
                draw.line([(cx, cy), (cx + bl * ddx, cy)], fill=bc, width=2)
                draw.line([(cx, cy), (cx, cy + bl * ddy)], fill=bc, width=2)

        # ── Scrolling ticker ──
        ticker_p = max(t - 0.55, 0) / 0.45
        if ticker_p > 0:
            ty = h - int(h * 0.055)
            ticker = (f"   GOOFER v1.0  ·  GOOF ANALYSIS  ·  "
                      f"{data['title']}  ({data['year']})  ·  "
                      f"{data['goof_count']} GOOFS DETECTED   ")
            offset = int(frame * 2.5) % max(len(ticker) * 10, 1)
            tc = _dim(NEON["dim"], ticker_p * 0.8)
            draw.text((w - offset, ty), ticker * 5, font=fonts["small"], fill=tc)
            draw.line([(0, ty - 3), (w, ty - 3)],
                      fill=_dim(NEON["dim"], ticker_p * 0.3), width=1)

        # ── Fractal border filigree ──
        fractal_fade = _ease_out(min(max(t - 0.15, 0) / 0.3, 1.0))
        if fractal_fade > 0 and fractal_top is not None:
            for x_norm, escape in fractal_top:
                if escape > 0:
                    px_x = int(x_norm * w)
                    frac_c = _dim(NEON["purple"], escape * fractal_fade *
                                  _breathing(time_sec, freq=0.2, lo=0.5, hi=1.0))
                    y_off = int(h * 0.015)
                    for dy in range(max(1, int(escape * 4))):
                        if y_off + dy < h:
                            draw.point((px_x, y_off + dy), fill=frac_c)
            if fractal_bot is not None:
                for x_norm, escape in fractal_bot:
                    if escape > 0:
                        px_x = int(x_norm * w)
                        frac_c = _dim(NEON["pink"], escape * fractal_fade *
                                      _breathing(time_sec, freq=0.2, lo=0.5, hi=1.0))
                        y_off = h - int(h * 0.07)
                        for dy in range(max(1, int(escape * 4))):
                            if y_off - dy >= 0:
                                draw.point((px_x, y_off - dy), fill=frac_c)

        # Post-processing
        img_arr = np.array(img)
        img_arr = _apply_vignette(img_arr, strength=0.45)
        img_arr = _apply_chromatic_aberration(img_arr, offset=max(1, int(2 * w / 768)))
        img_arr = _apply_film_grain(img_arr, intensity=8, frame_idx=frame)
        img_arr = _apply_bloom(img_arr, threshold=180, radius=6, intensity=0.2)

        return Image.fromarray(img_arr)

    # ══════════════════════════════════════════════════════════════════
    #  GOOF MINIMAL style
    # ══════════════════════════════════════════════════════════════════
    def _goof_minimal(self, w, h, t, frame, fps, data, fonts):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (w, h), (12, 12, 18))
        draw = ImageDraw.Draw(img)

        # Radial gradient background
        yc, xc = np.ogrid[:h, :w]
        dist = np.sqrt((xc - w / 2.0) ** 2 + (yc - h / 2.0) ** 2)
        max_dist = math.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)
        ratio = dist / max_dist
        c_val = np.clip(18 - ratio * 8, 0, 255).astype(np.uint8)
        bg_arr = np.stack([c_val, c_val, np.clip(c_val + 3, 0, 255).astype(np.uint8)], axis=2)
        img = Image.fromarray(bg_arr)
        draw = ImageDraw.Draw(img)

        # Breathing horizontal rule
        ry = int(h * 0.42)
        rule_breath = _breathing(frame / fps, freq=0.25, lo=0.5, hi=0.8)
        rule_w = int(w * 0.6 * _ease_out(min(t / 0.3, 1.0)))
        rx = (w - rule_w) // 2
        rule_c = _dim((80, 80, 110), rule_breath)
        draw.line([(rx, ry), (rx + rule_w, ry)], fill=rule_c, width=1)

        lines = [
            (data["title"], fonts["title"], (200, 200, 210), 0.08),
            (f"{data['year']}  ·  {data['genre_str']}", fonts["heading"],
             (120, 120, 140), 0.18),
            ("", None, None, 0),
            (f"{data['goof_count']} GOOFS DETECTED", fonts["heading"],
             (200, 80, 80), 0.30),
        ]

        # Add each goof as a line (desc already AI-summarized to 6 words)
        for i, (cat, desc) in enumerate(data["goof_lines"][:4]):
            lines.append((
                f"#{i+1} [{cat}] {desc}",
                fonts["body"], (160, 160, 170), 0.38 + i * 0.08
            ))

        y = int(h * 0.10)
        for text, font, color, start_t in lines:
            if font is None:
                y += int(h * 0.04)
                continue
            p = _ease_out(min(max(t - start_t, 0) / 0.2, 1.0))
            if p <= 0:
                y += int(h * 0.08)
                continue
            display = _typewriter(text, p)
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]
            x = (w - tw) // 2
            fc = _dim(color, p)
            draw.text((x, y), display, font=font, fill=fc)
            y += bbox[3] - bbox[1] + int(h * 0.02)

        # Watermark
        wm_p = max(t - 0.6, 0) / 0.4
        if wm_p > 0:
            wm_breath = _breathing(frame / fps, freq=0.2, lo=0.3, hi=0.6)
            draw.text((int(w * 0.05), h - int(h * 0.06)),
                      "GOOFER", font=fonts["small"],
                      fill=_dim((60, 60, 80), wm_p * wm_breath))

        # Post-processing
        img_arr = np.array(img)
        img_arr = _apply_vignette(img_arr, strength=0.3)
        img_arr = _apply_film_grain(img_arr, intensity=5, frame_idx=frame)

        return Image.fromarray(img_arr)

    # ══════════════════════════════════════════════════════════════════
    #  GOOF RETRO style
    # ══════════════════════════════════════════════════════════════════
    def _goof_retro(self, w, h, t, frame, fps, data, fonts, ca=None):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (w, h), (0, 4, 0))
        draw = ImageDraw.Draw(img)
        time_sec = frame / fps

        green = (0, 200, 0)
        dim_g = (0, 70, 0)
        bright_g = (0, 255, 0)

        # Cellular automata underlay
        if ca is not None:
            ca_fade = _ease_out(min(t / 0.4, 1.0)) * 0.5
            if ca_fade > 0:
                ca_overlay = ca.render(w, h, green, intensity=0.03 * ca_fade)
                img_arr = np.array(img, dtype=np.float32)
                img_arr += ca_overlay
                img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
                draw = ImageDraw.Draw(img)

        # CRT scan lines
        flicker = _breathing(time_sec, freq=30, lo=0.85, hi=1.0)
        c = int(8 * flicker)
        img_arr_crt = np.array(img)
        img_arr_crt[::2, :, 1] = np.clip(
            img_arr_crt[::2, :, 1].astype(np.int16) + c, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr_crt)
        draw = ImageDraw.Draw(img)

        # Header
        hdr_p = _ease_out(min(t / 0.2, 1.0))
        if hdr_p > 0:
            hdr = f">>> GOOF ANALYSIS SYSTEM v1.0 <<<"
            draw.text((int(w * 0.05), int(h * 0.04)),
                      _typewriter(hdr, hdr_p), font=fonts["heading"], fill=bright_g)
            draw.line([(int(w * 0.05), int(h * 0.10)),
                       (int(w * 0.95), int(h * 0.10))], fill=dim_g)

        # Target info
        tgt_p = _ease_out(min(max(t - 0.15, 0) / 0.15, 1.0))
        if tgt_p > 0:
            tgt = f"TARGET: {data['title']} ({data['year']})"
            draw.text((int(w * 0.05), int(h * 0.13)),
                      _typewriter(tgt, tgt_p), font=fonts["body"], fill=green)
            status = f"STATUS: {data['goof_count']} ANOMALIES DETECTED"
            draw.text((int(w * 0.05), int(h * 0.19)),
                      _typewriter(status, tgt_p), font=fonts["body"],
                      fill=(255, 50, 50) if data["goof_count"] > 0 else green)

        # Goof entries
        y = int(h * 0.28)
        for idx, (cat, desc) in enumerate(data["goof_lines"][:5]):
            entry_t = 0.25 + idx * 0.10
            ep = _ease_out(min(max(t - entry_t, 0) / 0.15, 1.0))
            if ep <= 0:
                continue

            # Blinking cursor effect
            cursor = "█" if int(time_sec * 3) % 2 == 0 and idx == len(data["goof_lines"]) - 1 else ""

            line1 = f"[{idx+1}] TYPE: {cat.upper()}"
            draw.text((int(w * 0.05), y),
                      _typewriter(line1, ep), font=fonts["small"], fill=bright_g)
            y += int(h * 0.05)

            line2 = f"    > {desc}{cursor}"
            draw.text((int(w * 0.05), y),
                      _typewriter(line2, ep), font=fonts["small"], fill=dim_g)
            y += int(h * 0.07)

        # Scrolling phosphor line
        scan_y = int((time_sec * 40) % h)
        scan_c = (0, 25, 0)
        for dy in range(3):
            if scan_y + dy < h:
                draw.line([(0, scan_y + dy), (w, scan_y + dy)], fill=scan_c)

        # Post-processing
        img_arr = np.array(img)
        img_arr = _apply_vignette(img_arr, strength=0.5)
        img_arr = _apply_film_grain(img_arr, intensity=10, frame_idx=frame)

        return Image.fromarray(img_arr)
