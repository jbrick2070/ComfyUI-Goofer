"""
ComfyUI-Goofer â€” Movie Goof Video Generator

Finds IMDB-style goofs in a movie, generates copyright-safe video prompts,
creates cinematic video clips from each goof, stitches them with procedural
art interstitials, and outputs a final upscaled video.

No external API keys required. Uses Cinemagoer (MIT) for movie data.
All processing is open-source and self-contained.

Author: Jeffrey A. Brick
License: MIT
"""

from .goofer_init import GooferInit
from .goofer_goof_fetch import GooferGoofFetch
from .goofer_sanitizer import GooferSanitizer
from .goofer_prompt_gen import GooferPromptGen
from .goofer_procedural_clip import GooferProceduralClip
from .goofer_batch_video import GooferBatchVideo
from .goofer_background_music import GooferBackgroundMusic
from .goofer_audio_enhance import GooferAudioEnhance
from .goofer_video_concat import GooferVideoConcat


NODE_CLASS_MAPPINGS = {
    "GooferInit":            GooferInit,
    "GooferGoofFetch":       GooferGoofFetch,
    "GooferSanitizer":       GooferSanitizer,
    "GooferPromptGen":       GooferPromptGen,
    "GooferProceduralClip":  GooferProceduralClip,
    "GooferBatchVideo":      GooferBatchVideo,
    "GooferBackgroundMusic": GooferBackgroundMusic,
    "GooferAudioEnhance":    GooferAudioEnhance,
    "GooferVideoConcat":     GooferVideoConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GooferInit":            "Goofer: Init",
    "GooferGoofFetch":       "Goofer: Fetch Goofs",
    "GooferSanitizer":       "Goofer: Sanitize",
    "GooferPromptGen":       "Goofer: Generate Prompts",
    "GooferProceduralClip":  "Goofer: Procedural Clip",
    "GooferBatchVideo":      "Goofer: Batch Video",
    "GooferBackgroundMusic": "Goofer: Background Music",
    "GooferAudioEnhance":    "Goofer: Audio Enhance",
    "GooferVideoConcat":     "Goofer: Stitch + RTX Upscale",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None

