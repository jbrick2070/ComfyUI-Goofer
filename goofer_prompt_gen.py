"""
GooferPromptGen — Converts sanitized goofs into cinematic video prompts.

Takes up to 5 sanitized goofs and generates text-to-video prompts that
describe each goof as a short cinematic scene. Prompts are designed for
LTX-2 / Wan / CogVideo / any text-to-video model.

No copyrighted terms in output — all prompts use generic scene descriptions
derived from the goof's category and sanitized description.

Author: Jeffrey A. Brick
"""

import logging
import random
import time

log = logging.getLogger("Goofer.PromptGen")


# ── Camera moves ───────────────────────────────────────────────────────
_CAMERAS = [
    "slow pan left",
    "slow tracking shot forward",
    "static wide shot",
    "gentle dolly forward",
    "slow pan right",
    "slow zoom in",
    "overhead crane shot descending",
    "steady handheld medium shot",
]

# ── Style prefixes ─────────────────────────────────────────────────────
_STYLE_PREFIXES = {
    "blockbuster": (
        "Epic Hollywood blockbuster, IMAX-scale wide shot, bold dynamic lighting, "
        "rich saturated color grade, fast dramatic camera push, practical stunt energy, "
        "cinematic lens flare, massive production value."
    ),
    "noir_cinematic": (
        "Film noir style, high contrast lighting with deep shadows, "
        "35mm anamorphic lens, moody atmosphere."
    ),
    "documentary_calm": (
        "Documentary style, natural handheld camera, warm natural lighting, "
        "50mm lens, observational tone."
    ),
    "golden_hour_beauty": (
        "Cinematic 4K, golden hour lighting, warm color grading, "
        "shallow depth of field, 85mm lens."
    ),
    "dramatic_broadcast": (
        "Broadcast quality, crisp professional lighting, steady tripod shot, "
        "clean composition, news-style framing."
    ),
    "retro_vhs": (
        "VHS camcorder aesthetic, slight tracking distortion, warm oversaturated colors, "
        "4:3 aspect feel, tape hiss, 1990s home video look."
    ),
}

# ── Category-specific scene templates ──────────────────────────────────
# Each template takes {description} and builds a cinematic scene around it.
_CATEGORY_SCENES = {
    "Continuity": [
        "Close-up shot of a table with objects. {camera}, {description}, "
        "subtle change visible between cuts, film set atmosphere, {lighting}",
        "Medium shot of a room interior. {camera}, {description}, "
        "mismatched props between angles, cinematic tension, {lighting}",
    ],
    "Factual Error": [
        "Wide shot of a scene with historical details. {camera}, {description}, "
        "period setting with anachronistic elements visible, {lighting}",
        "Documentary-style recreation. {camera}, {description}, "
        "factual inconsistency highlighted by framing, {lighting}",
    ],
    "Revealing Mistake": [
        "Behind-the-scenes film set. {camera}, equipment edge visible in frame, "
        "{description}, studio lights and rigging partially exposed, {lighting}",
        "Movie set interior. {camera}, {description}, "
        "crew shadow on wall, boom mic tip entering frame, {lighting}",
    ],
    "Anachronism": [
        "Period piece setting. {camera}, {description}, "
        "modern object subtly visible among period-correct props, {lighting}",
        "Historical scene recreation. {camera}, {description}, "
        "time-displaced item catching the light, {lighting}",
    ],
    "Audio/Visual Unsync": [
        "Close-up of a person speaking. {camera}, {description}, "
        "lips slightly out of sync with audio, dubbing artifact, {lighting}",
        "Dialogue scene between two people. {camera}, {description}, "
        "sound timing slightly off, post-production atmosphere, {lighting}",
    ],
    "Crew Visible": [
        "Film set with camera reflections. {camera}, {description}, "
        "equipment reflection in glass, crew member glimpsed in background, {lighting}",
    ],
    "Boom Mic Visible": [
        "Interior scene, top of frame. {camera}, {description}, "
        "dark fuzzy boom microphone dipping into shot from above, {lighting}",
    ],
    "Plot Hole": [
        "Dramatic scene with characters. {camera}, {description}, "
        "narrative logic breaking down, confused expressions, {lighting}",
    ],
    "Character Error": [
        "Character interaction scene. {camera}, {description}, "
        "personality inconsistency visible in body language, {lighting}",
    ],
    "Geography Error": [
        "Establishing shot of a city or landscape. {camera}, {description}, "
        "location mismatch between establishing and interior shots, {lighting}",
    ],
}

# Default template for unknown categories
_DEFAULT_SCENES = [
    "Cinematic movie scene. {camera}, {description}, "
    "subtle filmmaking error visible to attentive viewers, {lighting}",
    "Film production set. {camera}, {description}, "
    "movie mistake caught on camera, {lighting}",
]

# ── Lighting options ───────────────────────────────────────────────────
_LIGHTING = [
    "warm practical lighting from table lamps",
    "cool blue moonlight through windows",
    "harsh overhead fluorescent lighting",
    "soft golden hour sunlight streaming in",
    "dramatic chiaroscuro side lighting",
    "naturalistic overcast daylight",
    "neon sign reflections in dim room",
    "candlelight flickering on faces",
]


class GooferPromptGen:
    """Converts sanitized goofs into 5 cinematic video prompts."""

    CATEGORY = "Goofer"
    FUNCTION = "generate_prompts"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5",
                    "live_seed")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("GOOFER_CONFIG",),
                "goofs_data": ("GOOFER_GOOFS",),
                "movie_data": ("GOOFER_MOVIE",),
            },
        }

    def generate_prompts(self, config, goofs_data, movie_data):

        base_seed = config["seed"]
        live_seed = base_seed ^ (int(time.time()) & 0xFFFFFFFF)

        style_key = config.get("visual_style", "noir_cinematic")
        style_prefix = _STYLE_PREFIXES.get(style_key, "")

        # Build prompts for each goof (up to 5)
        prompts = []
        for i, goof in enumerate(goofs_data[:5]):
            rng = random.Random(live_seed + i * 7919)

            category = goof.get("category", "Miscellaneous")
            description = goof.get("description", "a filmmaking error")

            # Pick scene template
            templates = _CATEGORY_SCENES.get(category, _DEFAULT_SCENES)
            template = rng.choice(templates)

            # Pick camera and lighting
            camera = rng.choice(_CAMERAS)
            lighting = rng.choice(_LIGHTING)

            # Build the prompt
            scene = template.format(
                camera=camera,
                description=description,
                lighting=lighting,
            )

            prompt = f"{style_prefix} {scene}".strip()
            prompts.append(prompt)

            log.info("[PromptGen] goof %d [%s]: %s...",
                     i + 1, category, prompt[:80])

        # Pad to 5 with empty strings — BatchVideo will skip empty slots
        while len(prompts) < 5:
            prompts.append("")

        log.info("[PromptGen] Generated %d prompts (seed=%d)", len(prompts), live_seed)
        return (*prompts, live_seed)
