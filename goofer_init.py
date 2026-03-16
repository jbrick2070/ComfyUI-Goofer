"""
GooferInit — Central config for the Goofer pipeline.

No API keys needed. Takes a movie title and year as input,
produces a config dict that flows to all downstream nodes.

Author: Jeffrey A. Brick
"""

import time


class GooferInit:
    """Central config: movie title, year, visual style, seed."""

    CATEGORY = "Goofer"
    FUNCTION = "configure"
    RETURN_TYPES = ("GOOFER_CONFIG",)
    RETURN_NAMES = ("config",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "movie_title": ("STRING", {
                    "default": "Jurassic Park",
                    "multiline": False,
                    "tooltip": "Movie title to search for goofs"
                }),
                "movie_year": ("INT", {
                    "default": 1993,
                    "min": 1900,
                    "max": 2030,
                    "tooltip": "Release year (helps disambiguate titles)"
                }),
                "max_goofs": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "How many goofs to fetch (max 5, maps to batch video slots)"
                }),
                "visual_style": ([
                    "blockbuster",
                    "noir_cinematic",
                    "documentary_calm",
                    "golden_hour_beauty",
                    "dramatic_broadcast",
                    "retro_vhs",
                ], {
                    "default": "blockbuster",
                    "tooltip": "Visual style injected into all video prompts"
                }),
                "random_movie": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Pick a different movie from the built-in pool every run. "
                        "Overrides movie_title and movie_year. "
                        "Only selects movies known to have 3+ goofs on IMDb."
                    )
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Never cache — always re-run so seed rotation works."""
        return time.time()

    def configure(self, movie_title, movie_year, max_goofs,
                  visual_style, random_movie=True):
        config = {
            "movie_title": movie_title.strip(),
            "movie_year": int(movie_year),
            "max_goofs": int(max_goofs),
            "visual_style": visual_style,
            "seed": int(time.time()) % 2**32,
            "random_movie": bool(random_movie),
            "timestamp": time.time(),
        }
        return (config,)
