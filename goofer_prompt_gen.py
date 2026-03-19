"""
GooferPromptGen -- Converts sanitized goofs into cinematic video prompts.

Two modes:
  Template   -- Fast, deterministic, no extra model. Category-aware templates.
  Phi-3-mini -- microsoft/Phi-3-mini-4k-instruct generates richer AI prompts.
                Auto-downloads ~4 GB first use. Unloads VRAM after generation
                so LTX-Video has full headroom. NSFW refusal baked in.

Author: Jeffrey A. Brick
"""

import logging
import random
import time

log = logging.getLogger("Goofer.PromptGen")

# Shared genre/mood cache — populated by generate_prompts() while Phi-3 is
# loaded, then read by GooferBackgroundMusic to build the MusicGen prompt.
# Keyed by movie title string.  Replaces Flan-T5 genre inference entirely.
_cached_genre_mood: dict = {}


# -- Phi-3-mini lazy loader ----------------------------------------------------
_phi3_model = None
_phi3_tok   = None


def _get_phi3():
    """Lazy-load microsoft/Phi-3-mini-4k-instruct in float16."""
    global _phi3_model, _phi3_tok
    if _phi3_model is not None:
        return _phi3_model, _phi3_tok
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        log.info("[PromptGen] Loading Phi-3-mini (~4 GB first run)...")
        _phi3_tok = AutoTokenizer.from_pretrained(model_id)
        _phi3_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # native transformers 5.0 Phi-3 support
        ).to("cuda").eval()
    except Exception as exc:
        log.exception("[PromptGen] Phi-3-mini failed to load -- Template mode.")
        _phi3_model = None
        _phi3_tok   = None
    return _phi3_model, _phi3_tok


def _unload_phi3():
    """Free VRAM after prompts are done so LTX-Video has full headroom."""
    global _phi3_model, _phi3_tok
    if _phi3_model is None:
        return
    try:
        import torch
        del _phi3_model, _phi3_tok
        _phi3_model = None
        _phi3_tok   = None
        torch.cuda.empty_cache()
        log.info("[PromptGen] Phi-3-mini unloaded from VRAM.")
    except Exception as exc:
        log.debug("[PromptGen] Phi-3 unload: %s", exc)


# NSFW refusal + quality rules baked into system prompt
_PHI3_SYSTEM = (
    "You are a professional film director writing text-to-video prompts for an AI "
    "video model. Your prompts describe short 5-second cinematic scenes. "
    "Rules you must always follow:\n"
    "1. Output ONLY the prompt text -- no preamble, no explanation, no quotes.\n"
    "2. ALL AGES SAFE. No nudity, no sexual content, no graphic violence. "
    "If the goof description contains such content, substitute a safe cinematic "
    "scene that captures the same visual energy without the explicit element.\n"
    "3. No real actor names, character names, film titles, brand names, or studio names.\n"
    "4. Between 40 and 80 words.\n"
    "5. Always specify camera movement, lighting quality, and visual style.\n"
    "6. The core objective is to visibly recreate the exact film mistake or continuity error described. The mistake itself MUST be the central visual focus of the clip.\n"
    "7. CRITICAL: All characters must hold and use bananas instead of weapons. Do not include any guns, firearms, or knives in your output. Replace all implied weapons with bananas.\n"
)

_HIGHLIGHT_STYLES = [
    "A glowing futuristic targeting HUD suddenly locks onto the mistake.",
    "A sudden stark spotlight beam illuminates the error in extreme contrast.",
    "The scene abruptly drops into extreme slow-motion as the mistake occurs.",
    "A fast snap-zoom and sudden rack focus directly onto the error, heavily blurring the background.",
    "A bright glowing outline or neon aura begins pulsating around the mistake.",
    "A dramatic cinematic camera push-in isolates the mistake as the rest of the frame darkens."
]

_PHI3_USER_TMPL = (
    "Film goof category: {category}\n"
    "Goof description: {description}\n\n"
    "Write a cinematic 5-second text-to-video prompt that explicitly and visually recreates this exact film mistake. "
    "Describe the scene such that the error itself is clearly visible and is the focal point. "
    "Crucial visual requirement: {highlight}\n"
    "Visual style: {style}."
)

_PHI3_BAD = ["i cannot", "i can't", "i'm sorry", "i apologize", "as an ai"]


def _phi3_prompt(model, tok, category: str, description: str, style: str, highlight: str) -> str:
    """Generate one LTX-Video prompt via Phi-3-mini. Returns '' on failure."""
    import torch
    msg = _PHI3_USER_TMPL.format(
        category=category, description=description[:300], style=style, highlight=highlight
    )
    messages = [
        {"role": "system", "content": _PHI3_SYSTEM},
        {"role": "user",   "content": msg},
    ]
    text   = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    new_tok = out[0][inputs["input_ids"].shape[1]:]
    result  = tok.decode(new_tok, skip_special_tokens=True).strip()

    if len(result.split()) < 10:
        log.warning("[PromptGen] Phi-3 output too short, using template fallback.")
        return ""
    if any(b in result.lower() for b in _PHI3_BAD):
        log.warning("[PromptGen] Phi-3 refusal/bad output detected, using template fallback.")
        return ""
    return result


# -- Phi-3 genre/mood inference (runs while Phi-3 is already loaded) -----------

_PHI3_GENRE_SYSTEM = (
    "You are a film music supervisor. Given a film plot, describe the ideal "
    "musical genre and mood in exactly 6-10 words. "
    "Output ONLY the description — no explanation, no punctuation at the end."
)

_PHI3_GENRE_BAD = [
    "directed by", "written by", "starring", "based on",
    "released in", "produced by", "i cannot", "i can't",
]


def _infer_phi3_genre_mood(model, tok, title: str, plot: str) -> str:
    """Ask Phi-3-mini for a genre/mood description to drive MusicGen.

    Returns a string like 'suspenseful orchestral thriller with dark piano'
    or '' if inference fails or plot is too short.
    """
    import torch
    if not plot or len(plot.strip()) < 25:
        return ""
    try:
        messages = [
            {"role": "system", "content": _PHI3_GENRE_SYSTEM},
            {"role": "user",   "content": f"Film plot: {plot[:400]}"},
        ]
        text   = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=30, temperature=0.6,
                do_sample=True, top_p=0.9, pad_token_id=tok.eos_token_id,
            )
        new_tok = out[0][inputs["input_ids"].shape[1]:]
        result  = tok.decode(new_tok, skip_special_tokens=True).strip()

        if len(result.split()) < 4:
            log.warning("[PromptGen] Phi-3 genre output too short: %s", result)
            return ""
        if any(b in result.lower() for b in _PHI3_GENRE_BAD):
            log.warning("[PromptGen] Phi-3 genre hallucination discarded: %s", result)
            return ""
        log.info("[PromptGen] Phi-3 genre/mood for '%s': %s", title, result)
        return result
    except Exception as exc:
        log.debug("[PromptGen] Phi-3 genre inference failed: %s", exc)
        return ""


# -- Template system -----------------------------------------------------------
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

_STYLE_PREFIXES = {
    "blockbuster": (
        "Epic Hollywood blockbuster, IMAX-scale wide shot, bold dynamic lighting, "
        "rich saturated color grade, cinematic lens flare."
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
        "4:3 aspect feel, 1990s home video look."
    ),
}

_CATEGORY_SCENES = {
    "Continuity": [
        "Close-up shot of a table with objects. {camera}, {description}, "
        "subtle change visible between cuts, film set atmosphere, {lighting}",
        "Medium shot of a room interior. {camera}, {description}, "
        "mismatched props between angles, cinematic tension, {lighting}",
    ],
    "Factual Error": [
        "Wide shot of a historical scene. {camera}, {description}, "
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

_DEFAULT_SCENES = [
    "Cinematic movie scene. {camera}, {description}, "
    "subtle filmmaking error visible to attentive viewers, {lighting}",
    "Film production set. {camera}, {description}, "
    "movie mistake caught on camera, {lighting}",
]

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


def _template_prompt(rng, category: str, description: str, style_prefix: str) -> str:
    templates = _CATEGORY_SCENES.get(category, _DEFAULT_SCENES)
    scene = rng.choice(templates).format(
        camera=rng.choice(_CAMERAS),
        description=description,
        lighting=rng.choice(_LIGHTING),
    )
    return f"{style_prefix} {scene}".strip()


class GooferPromptGen:
    """Converts sanitized goofs into 5 cinematic video prompts.

    prompt_mode = Template  : fast, no extra model, category-aware templates.
    prompt_mode = Phi-3-mini: richer AI-written prompts. Unloads after generation.
    NSFW content refused in both modes.
    """

    CATEGORY     = "Goofer"
    FUNCTION     = "generate_prompts"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5", "live_seed")
    OUTPUT_NODE  = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config":     ("GOOFER_CONFIG",),
                "goofs_data": ("GOOFER_GOOFS",),
                "movie_data": ("GOOFER_MOVIE",),
            },
            "optional": {
                "prompt_mode": (["Template", "Phi-3-mini"], {
                    "default": "Phi-3-mini",
                    "tooltip": (
                        "Template: fast, no extra model. "
                        "Phi-3-mini: AI-written prompts via microsoft/Phi-3-mini-4k-instruct "
                        "(~4 GB first download, unloads before LTX-Video). "
                        "NSFW refused in both modes."
                    ),
                }),
            },
        }

    def generate_prompts(self, config, goofs_data, movie_data, prompt_mode="Phi-3-mini"):
        base_seed    = config["seed"]
        live_seed    = base_seed ^ (int(time.time()) & 0xFFFFFFFF)
        style_key    = config.get("visual_style", "noir_cinematic")
        style_prefix = _STYLE_PREFIXES.get(style_key, "")
        style_name   = style_key.replace("_", " ")

        phi3_model = phi3_tok = None
        if prompt_mode == "Phi-3-mini":
            phi3_model, phi3_tok = _get_phi3()
            if phi3_model is None:
                log.warning("[PromptGen] Phi-3-mini unavailable -- using Template mode.")
                prompt_mode = "Template"

        prompts = []
        for i, goof in enumerate(goofs_data[:5]):
            rng         = random.Random(live_seed + i * 7919)
            category    = goof.get("category", "Miscellaneous")
            description = goof.get("description", "a filmmaking error")

            if prompt_mode == "Phi-3-mini":
                highlight = rng.choice(_HIGHLIGHT_STYLES)
                result = _phi3_prompt(phi3_model, phi3_tok, category, description, style_name, highlight)
                if not result:
                    # Template fallback if Phi-3 output is bad or refused
                    result = _template_prompt(rng, category, description, style_prefix)
            else:
                result = _template_prompt(rng, category, description, style_prefix)

            prompts.append(result)
            log.info("[PromptGen] goof %d [%s] (%s): %s...",
                     i + 1, category, prompt_mode, result[:80])

        # While Phi-3 is still loaded, infer genre/mood for BackgroundMusic.
        # Store in _cached_genre_mood so BackgroundMusic can read it without
        # re-loading Phi-3 (avoids VRAM conflict with LTX-Video + MusicGen).
        if prompt_mode == "Phi-3-mini" and phi3_model is not None:
            title = movie_data.get("title", "")
            plot  = movie_data.get("plot",  "")
            if title and title not in _cached_genre_mood:
                gm = _infer_phi3_genre_mood(phi3_model, phi3_tok, title, plot)
                if gm:
                    _cached_genre_mood[title] = gm

        # Unload Phi-3 so LTX-Video gets full VRAM
        if prompt_mode == "Phi-3-mini":
            _unload_phi3()

        while len(prompts) < 5:
            prompts.append("")

        log.info("[PromptGen] Generated %d prompts (mode=%s seed=%d)",
                 len(prompts), prompt_mode, live_seed)
        return (*prompts, live_seed)
