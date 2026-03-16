# ComfyUI-Goofer v1.0
## IMDb Goof-to-Video Pipeline

**Random IMDb movie goof → AI video prompt → LTX-2 video clips → procedural interstitial → MusicGen film score → final stitched video.**

Fully automated. Zero API keys. Drop into `custom_nodes/` and queue.

---

## Download

[![Download Goofer v1.0](https://img.shields.io/badge/Download-Goofer_v1.0-blue?style=for-the-badge)](https://github.com/jbrick2070/ComfyUI-Goofer/releases)

**[Click here to download the full package (v1.0)](https://github.com/jbrick2070/ComfyUI-Goofer/releases)** — includes workflow JSON + this guide.

---

## What It Does

Goofer picks a random film from a pool of 424 curated titles, fetches its IMDb continuity errors and goofs, sanitizes all copyrighted names and brands, generates a cinematic video prompt for each goof, renders an LTX-2 video clip per goof, creates a procedural data-viz interstitial with an AI-generated film score, then stitches everything into a single upscaled output video.

Every run is a different movie. Every queue is a new short film.

---

## New to ComfyUI? Start Here

ComfyUI is a free, node-based interface for running AI image and video models locally on your GPU.

> **Already have ComfyUI installed?** Skip to Step 2.

### Step 1 — Install ComfyUI

Use the official desktop installer — it handles Python, Git, and dependencies automatically:

**https://www.comfy.org/download**

Advanced users can also install manually from https://github.com/comfyanonymous/ComfyUI

### Step 2 — Install Required Models

> **☕ Grab a coffee — the main video model is ~9.5 GB.**

| Model | Download | Size | Save To |
|-------|----------|------|---------|
| **LTX-Video v0.9.5** (required) | [HuggingFace](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors) | ~9.5 GB | `ComfyUI/models/checkpoints/` |
| **LTX-Video 13B** (optional, higher quality) | [HuggingFace](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-distilled.safetensors) | ~26 GB | `ComfyUI/models/checkpoints/` |
| **MusicGen Large** | Auto-downloaded on first run | ~3.3 GB | HuggingFace cache |
| **Flan-T5-small** | Auto-downloaded on first run | ~80 MB | HuggingFace cache |

> MusicGen and Flan-T5 download automatically the first time you run the workflow — no manual steps needed.

### Step 3 — Install ComfyUI-Goofer

**Option A — ComfyUI Manager (recommended):**
1. Open ComfyUI Manager → Install Custom Nodes
2. Search "Goofer" → Install → Restart

**Option B — Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jbrick2070/ComfyUI-Goofer
```

Then install Python dependencies:
```bash
pip install cinemagoer requests transformers torch
```

### Step 4 — Load the Workflow

1. Open ComfyUI at `http://127.0.0.1:8000`
2. Click **Load** and select `example_workflows/goofer_basic.json`
3. If any nodes appear red: **Manager → Install Missing Custom Nodes → Restart**
4. Hit **Queue** — Goofer picks a random movie and starts generating

**Which GPU do I need?**

| Setup | GPU | VRAM |
|-------|-----|------|
| Recommended | RTX 5080 / 4090 | 16 GB+ |
| Minimum | RTX 4070 / 3060 | 8 GB |

> **Check your VRAM:** Windows: Task Manager (Ctrl+Shift+Esc) → Performance → GPU → Dedicated GPU Memory. Linux: `nvidia-smi`

---

## The Node Pipeline

```
GooferInit
    │
    ▼
GooferGoofFetch ──────────────────────────────────┐
    │  movie_data  goofs_data                      │
    ▼                                              │
GooferSanitizer                                   │
    │  sanitized_goofs                             │
    ▼                                              │
GooferPromptGen                                   │
    │  prompts (×5)                                │
    ▼                                              │
GooferBatchVideo                                  │
    │  video clips (×5)                            │
    ▼                                              ▼
 [final stitch] ◄── GooferProceduralClip ◄── GooferBackgroundMusic
```

### Node Reference

| Node | What It Does |
|------|-------------|
| **GooferInit** | Sets global config: output path, upscale mode, seed |
| **GooferGoofFetch** | Picks a random film from 424 titles, fetches goofs from IMDb via Cinemagoer + direct HTTP fallback. Seed-diversified so every run draws a fresh subset from the full goof pool |
| **GooferSanitizer** *(Copyright Cleaner)* | Strips copyrighted character names, actor names, studio names, brand names, and franchise references from all goof text before it reaches any AI model |
| **GooferPromptGen** | Converts sanitized goof descriptions into cinematic LTX-2 video prompts using category-aware templates (continuity errors → observational style, factual errors → documentary style, etc.) |
| **GooferBatchVideo** | Feeds each prompt to LTX-Video and renders one video clip per goof. Configurable resolution, frame count, and guidance strength per clip |
| **GooferBackgroundMusic** | Generates a film score with Meta MusicGen 3. Prompt is derived from actual goof keywords (explosions → percussive bass, chases → driving rhythm, basketball → upbeat brass, etc.) plus Flan-T5-small genre/mood inference from the film's plot. Falls back to additive-synthesis chord progression if MusicGen is unavailable |
| **GooferProceduralClip** | Renders a data-viz interstitial showing goof metadata as animated neon graphics. Duration syncs to the MusicGen output length via the `music_duration` input |
| **GooferAudioEnhance** | Optional EQ / loudness normalization pass on the final audio mix |

---

## Movie Pool

Goofer ships with **424 curated titles** spanning 1939–2024:

Action · Sci-Fi · Horror · Comedy · Drama · Western · War · Disaster · Spy/Espionage · Animation · Romance · Musical · Sports · Cult Classics

Each entry is verified against IMDb with a title-sanity check — if an ID returns the wrong film, Goofer skips it and picks another rather than generating content for the wrong movie.

To add your own films, edit `_RANDOM_MOVIE_POOL` in `goofer_goof_fetch.py`:
```python
("Your Movie Title", YEAR, "IMDB_ID_WITHOUT_TT"),
# e.g.
("The Thing",        1982, "0084787"),
```

---

## Copyright Safety

Goofer is designed to keep copyrighted names out of AI-generated video prompts. The **Copyright Cleaner** (GooferSanitizer) node runs several passes before any text reaches LTX-Video or MusicGen:

- Franchise names replaced with generic descriptions (`"James Bond"` → `"a spy thriller"`)
- Movie title as character name → `"the character"`
- Movie title as title reference → `"the film"`
- Actor and character names from the IMDb cast list stripped automatically
- Studio and production company names stripped
- Brand names including model names (`"Lamborghini Gallardo"`) → `"a well-known brand"`
- Standard First-Last name patterns detected and replaced

---

## Troubleshooting

<details>
<summary><strong>Nodes load with wrong widget values / "Failed to convert FLOAT"</strong></summary>

**Cause:** Stale `.pyc` bytecode in `__pycache__` loaded before the updated `.py` files.

**Fix:** Delete the cache folder and restart ComfyUI:
```powershell
Remove-Item "ComfyUI/custom_nodes/ComfyUI-Goofer/__pycache__" -Recurse -Force
```
</details>

<details>
<summary><strong>"No goofs found" / placeholder goofs appearing</strong></summary>

**Cause:** Cinemagoer's HTML parser is broken against IMDb's current page structure (known upstream issue). Goofer includes a direct HTTP fallback using `__NEXT_DATA__` JSON extraction, but some films legitimately have zero goofs listed.

**Fix:** The fallback is automatic. If a specific film keeps returning empty, verify its IMDb ID by checking `https://www.imdb.com/title/ttXXXXXXX/goofs/` in your browser.
</details>

<details>
<summary><strong>MusicGen prompt shows "cinematic atmospheric" every run</strong></summary>

**Cause:** No plot text from Cinemagoer (common on first fetch for some titles), so Flan-T5 skips inference and the genre keyword dict has no match.

**What to expect:** This is normal for some titles on first run. Goof keyword matching (explosions, chases, fights, sports, etc.) still enriches the prompt regardless of genre availability.
</details>

<details>
<summary><strong>MusicGen takes a long time on first run</strong></summary>

**Cause:** `facebook/musicgen-large` (~3.3 GB) downloads on first use and loads into VRAM alongside LTX-Video.

**Fix:** One-time only. After first run both models stay cached locally. If VRAM is tight, switch to `facebook/musicgen-small` (~300 MB) in the GooferBackgroundMusic node widget.
</details>

<details>
<summary><strong>"Title mismatch" warning in logs</strong></summary>

**Cause:** The IMDb ID in `_RANDOM_MOVIE_POOL` points to the wrong film (reused ID, sequel confusion, or regional title difference).

**Fix:** Look up the correct ID at `https://www.imdb.com/find?q=TITLE+YEAR` and update the 8-digit number in `goofer_goof_fetch.py`.
</details>

---

## Setting Up Continuous Output with OBS

Run Goofer as a live generative broadcast — each output video auto-loads into OBS as it finishes.

### Prerequisites

- [OBS Studio](https://obsproject.com/download)
- [Python 3.11.x](https://www.python.org/downloads/release/python-3119/)
- [Media Playlist Source Plugin](https://obsproject.com/forum/resources/media-playlist-source.1524/)
- [Directory Sorter for OBS Script](https://obsproject.com/forum/resources/directory-watch-media-sorter.1767/)

### Setup

1. Install OBS and the Media Playlist Source plugin
2. In OBS: **Tools → Scripts** → Python Settings → point to your Python 3.11 path
3. Load the `directory_sorter_for_obs` script and point it to `ComfyUI/output/`
4. Add a **Media Playlist Source** scene item pointed to the same folder
5. OBS automatically picks up each new Goofer output as it renders

> **Pro tip:** If your main GPU is maxed out on inference, set OBS to encode via your integrated GPU (**QSV AV1** or **HEVC**). This keeps the stream smooth while the RTX handles LTX-Video and MusicGen.

---

## Requirements

```
cinemagoer
requests
transformers>=4.40.0
torch>=2.0.0
numpy
```

Optional (for higher-quality audio resampling):
```
librosa
```

---

*Built by Jeffrey A. Brick — Los Angeles, 2026*
