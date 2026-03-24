"""
GooferPromptGen -- Converts sanitized goofs into cinematic video prompts.

Three modes:
  Template        -- Fast, deterministic, no extra model. Category-aware templates.
  Qwen2.5-3B     -- Qwen/Qwen2.5-3B-Instruct generates richer AI prompts.
                    Auto-downloads ~6 GB first use. Unloads VRAM after generation
                    so LTX-Video has full headroom.
  Qwen2.5-7B     -- Qwen/Qwen2.5-7B-Instruct for higher quality prompts.
                    Auto-downloads ~14 GB. Better quality, more VRAM.

Uses the same Qwen lazy-load/unload pattern as ComfyUI-UCLA-News-Video.

Author: Jeffrey A. Brick
"""

import logging
import random
import time

log = logging.getLogger("Goofer.PromptGen")

# Shared genre/mood cache — populated by generate_prompts() while Qwen is
# loaded, then read by GooferBackgroundMusic to build the MusicGen prompt.
# Keyed by movie title string.
_cached_genre_mood: dict = {}


# -- Qwen lazy loader (same pattern as UCLA News Video) ------------------------
_qwen_model = None
_qwen_tok   = None
_qwen_loaded_id = None


def _get_qwen(model_id):
    """Lazy-load a Qwen2.5-Instruct model in float16 on CUDA."""
    global _qwen_model, _qwen_tok, _qwen_loaded_id

    # If the requested model is already loaded, reuse it
    if _qwen_model is not None and _qwen_loaded_id == model_id:
        return _qwen_model, _qwen_tok

    # If a different model is loaded, unload it first
    if _qwen_model is not None:
        _unload_qwen()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("[PromptGen] Loading %s ...", model_id)
        print(f"[Goofer] Loading {model_id} (first run downloads model)...")

        _qwen_tok = AutoTokenizer.from_pretrained(model_id)
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
        ).to("cuda").eval()

        _qwen_loaded_id = model_id
        log.info("[PromptGen] %s loaded on CUDA", model_id)
        print(f"[Goofer] {model_id} loaded on CUDA")
    except Exception as exc:
        log.exception("[PromptGen] %s failed to load — Template mode.", model_id)
        print(f"[Goofer] Qwen load failed: {exc}")
        _qwen_model = None
        _qwen_tok   = None
        _qwen_loaded_id = None

    return _qwen_model, _qwen_tok


def _unload_qwen():
    """Free VRAM after prompts are done so LTX-Video has full headroom."""
    global _qwen_model, _qwen_tok, _qwen_loaded_id
    if _qwen_model is None:
        return
    try:
        import torch
        del _qwen_model, _qwen_tok
        _qwen_model = None
        _qwen_tok   = None
        _qwen_loaded_id = None
        torch.cuda.empty_cache()
        log.info("[PromptGen] Qwen unloaded from VRAM.")
        print("[Goofer] Qwen unloaded from VRAM")
    except Exception as exc:
        log.debug("[PromptGen] Qwen unload: %s", exc)


# NSFW refusal + quality rules baked into system prompt
_QWEN_SYSTEM = (
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

_QWEN_USER_TMPL = (
    "Film goof category: {category}\n"
    "Goof description: {description}\n\n"
    "Write a cinematic 5-second text-to-video prompt that explicitly and visually recreates this exact film mistake. "
    "Describe the scene such that the error itself is clearly visible and is the focal point. "
    "Crucial visual requirement: {highlight}\n"
    "Visual style: {style}."
)

_BAD_OUTPUTS = ["i cannot", "i can't", "i'm sorry", "i apologize", "as an ai",
                "here is", "here's the", "sure,", "certainly", "of course"]


def _qwen_prompt(model, tok, category: str, description: str, style: str, highlight: str) -> str:
    """Generate one LTX-Video prompt via Qwen2.5. Returns '' on failure."""
    import torch
    msg = _QWEN_USER_TMPL.format(
        category=category, description=description[:300], style=style, highlight=highlight
    )
    messages = [
        {"role": "system", "content": _QWEN_SYSTEM},
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

    # Strip wrapping quotes
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1].strip()

    if len(result.split()) < 10:
        log.warning("[PromptGen] Qwen output too short, using template fallback.")
        return ""
    if any(b in result.lower()[:60] for b in _BAD_OUTPUTS):
        log.warning("[PromptGen] Qwen refusal/bad output detected, using template fallback.")
        return ""
    return result


# -- Qwen genre/mood inference (runs while Qwen is already loaded) -------------

_GENRE_SYSTEM = (
    "You are a film music supervisor. Given a film plot, describe the ideal "
    "musical genre and mood in exactly 6-10 words. "
    "Output ONLY the description — no explanation, no punctuation at the end."
)

_GENRE_BAD = [
    "directed by", "written by", "starring", "based on",
    "released in", "produced by", "i cannot", "i can't",
]


def _infer_genre_mood(model, tok, title: str, plot: str) -> str:
    """Ask Qwen2.5 for a genre/mood description to drive MusicGen.

    Returns a string like 'suspenseful orchestral thriller with dark piano'
    or '' if inference fails or plot is too short.
    """
    import torch
    if not plot or len(plot.strip()) < 25:
        return ""
    try:
        messages = [
            {"role": "system", "content": _GENRE_SYSTEM},
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
            log.warning("[PromptGen] Qwen genre output too short: %s", result)
            return ""
        if any(b in result.lower() for b in _GENRE_BAD):
            log.warning("[PromptGen] Qwen genre hallucination discarded: %s", result)
            return ""
        log.info("[PromptGen] Qwen genre/mood for '%s': %s", title, result)
        return result
    except Exception as exc:
        log.debug("[PromptGen] Qwen genre inference failed: %s", exc)
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


# ── Model size → HuggingFace ID mapping ──────────────────────────────

_MODEL_MAP = {
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}


class GooferPromptGen:
    """Converts sanitized goofs into 5 cinematic video prompts.

    prompt_mode = Template          : fast, no extra model, category-aware templates.
    prompt_mode = Qwen2.5-3B-Instruct : AI-written prompts (~6 GB VRAM).
    prompt_mode = Qwen2.5-7B-Instruct : higher quality (~14 GB VRAM).
    Unloads after generation. NSFW content refused in all modes.
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
                "prompt_mode": (["Template", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"], {
                    "default": "Qwen2.5-3B-Instruct",
                    "tooltip": (
                        "Template: fast, no extra model. "
                        "Qwen2.5-3B: AI-written prompts (~6 GB VRAM, faster). "
                        "Qwen2.5-7B: higher quality (~14 GB VRAM). "
                        "Models auto-download first use, unload before LTX-Video."
                    ),
                }),
                "unload_after": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Free VRAM after generation. Disable if chaining multiple prompt nodes.",
                }),
            },
        }

    def generate_prompts(self, config, goofs_data, movie_data,
                         prompt_mode="Qwen2.5-3B-Instruct", unload_after=True):
        base_seed    = config["seed"]
        live_seed    = base_seed ^ (int(time.time_ns()) & 0xFFFFFFFF)
        style_key    = config.get("visual_style", "noir_cinematic")
        style_prefix = _STYLE_PREFIXES.get(style_key, "")
        style_name   = style_key.replace("_", " ")

        qwen_model = qwen_tok = None
        using_ai = prompt_mode in _MODEL_MAP

        if using_ai:
            model_id = _MODEL_MAP[prompt_mode]
            qwen_model, qwen_tok = _get_qwen(model_id)
            if qwen_model is None:
                log.warning("[PromptGen] %s unavailable — using Template mode.", prompt_mode)
                using_ai = False

        prompts = []
        for i, goof in enumerate(goofs_data[:5]):
            rng         = random.Random(live_seed + i * 7919)
            category    = goof.get("category", "Miscellaneous")
            description = goof.get("description", "a filmmaking error")

            if using_ai:
                highlight = rng.choice(_HIGHLIGHT_STYLES)
                result = _qwen_prompt(qwen_model, qwen_tok, category, description, style_name, highlight)
                if not result:
                    result = _template_prompt(rng, category, description, style_prefix)
            else:
                result = _template_prompt(rng, category, description, style_prefix)

            prompts.append(result)
            log.info("[PromptGen] goof %d [%s] (%s): %s...",
                     i + 1, category, prompt_mode, result[:80])

        # While Qwen is still loaded, infer genre/mood for BackgroundMusic.
        # Store in _cached_genre_mood so BackgroundMusic can read it without
        # re-loading Qwen (avoids VRAM conflict with LTX-Video + MusicGen).
        if using_ai and qwen_model is not None:
            title = movie_data.get("title", "")
            plot  = movie_data.get("plot",  "")
            if title and title not in _cached_genre_mood:
                gm = _infer_genre_mood(qwen_model, qwen_tok, title, plot)
                if gm:
                    _cached_genre_mood[title] = gm

        # Unload Qwen so LTX-Video gets full VRAM
        if using_ai and unload_after:
            _unload_qwen()

        while len(prompts) < 5:
            prompts.append("")

        log.info("[PromptGen] Generated %d prompts (mode=%s seed=%d)",
                 len(prompts), prompt_mode, live_seed)
        return (*prompts, live_seed)
