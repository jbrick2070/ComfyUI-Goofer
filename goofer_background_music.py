"""
Goofer_BackgroundMusic — Generates AUDIO for the Goofer pipeline.

Primary path : Meta MusicGen 3 (facebook/musicgen-large)
               Builds a cinematic film-score prompt from movie genre +
               goof categories and runs inference.

Fallback     : Contextual chord-progression (additive synthesis, zero deps)
               Derives tempo / key / instrument from the same movie context.

Outputs:
  music        — AUDIO tensor (1, 2, samples) stereo
  duration_sec — FLOAT echoed back so it can wire to GooferProceduralClip's
                 'music_duration' input and keep both clips the same length.

v2.1  2026-03-15  Removed all MIDI / pretty_midi / mido references.
"""

import logging
import torch
import numpy as np

log = logging.getLogger("Goofer.BackgroundMusic")


# ─── Flan-T5 genre/mood inferencer (lazy, one-shot) ─────────────────────────

_FLAN_MODEL     = None
_FLAN_TOKENIZER = None
_FLAN_FAILED    = False
_FLAN_MODEL_ID  = "google/flan-t5-small"   # ~80 MB — fast, instruction-tuned


def _get_flan():
    """Lazy-load flan-t5-small once per session.  Returns (model, tokenizer) or (None, None)."""
    global _FLAN_MODEL, _FLAN_TOKENIZER, _FLAN_FAILED
    if _FLAN_FAILED:
        return None, None
    if _FLAN_MODEL is not None:
        return _FLAN_MODEL, _FLAN_TOKENIZER
    try:
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        log.info("[BackgroundMusic] Loading Flan-T5-small for genre inference...")
        tok   = AutoTokenizer.from_pretrained(_FLAN_MODEL_ID)
        model = T5ForConditionalGeneration.from_pretrained(_FLAN_MODEL_ID)
        model.eval()
        _FLAN_MODEL     = model
        _FLAN_TOKENIZER = tok
        log.info("[BackgroundMusic] Flan-T5-small ready")
        return model, tok
    except Exception as exc:
        log.warning("[BackgroundMusic] Flan-T5-small unavailable (%s) — using keyword fallback", exc)
        _FLAN_FAILED = True
        return None, None


def _infer_genre_mood(title: str, plot: str) -> str:
    """Ask Flan-T5-small to describe the film's genre and mood in ≤12 words.

    Returns a short string like 'uplifting sports drama with triumphant orchestral score'
    or empty string on failure.
    """
    model, tok = _get_flan()
    if model is None:
        return ""
    try:
        # Only run if we have actual plot text to ground the inference.
        # Title-only prompts produce hallucinations from tiny models — skip them.
        if not plot or len(plot.strip()) < 25:
            return ""
        prompt = (
            f"In 8 words, what musical mood and genre fits this film plot: {plot[:300]}"
        )
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20,
                                 num_beams=4, early_stopping=True)
        result = tok.decode(out[0], skip_special_tokens=True).strip()
        # Guard 1 — must be at least 4 words (single-word outputs like "sexy" are noise)
        if len(result.split()) < 4:
            log.warning("[BackgroundMusic] Flan-T5 output too short, discarding: %s", result)
            return ""
        # Guard 2 — reject factual hallucinations ("directed by", "based on", etc.)
        _BAD_PATTERNS = ["directed by", "written by", "is based on", "starring",
                         "stars ", "produced by", "released in", "film is"]
        if any(p in result.lower() for p in _BAD_PATTERNS):
            log.warning("[BackgroundMusic] Flan-T5 hallucinated factual response, discarding: %s", result)
            return ""
        log.info("[BackgroundMusic] Flan-T5 genre inference: %s", result)
        return result
    except Exception as exc:
        log.warning("[BackgroundMusic] Flan-T5 inference failed: %s", exc)
        return ""


# ─── MusicGen lazy loader ────────────────────────────────────────────────────

_MUSICGEN_MODEL       = None
_MUSICGEN_PROCESSOR   = None
_MUSICGEN_LOADED_NAME = None
_MUSICGEN_FAILED      = False   # set True permanently on first load failure


def _get_musicgen(model_name: str):
    """Lazy-load a MusicGen model once per session.

    Returns (model, processor) or (None, None) if unavailable.
    """
    global _MUSICGEN_MODEL, _MUSICGEN_PROCESSOR, _MUSICGEN_LOADED_NAME, _MUSICGEN_FAILED
    if _MUSICGEN_FAILED:
        return None, None
    if _MUSICGEN_MODEL is not None and _MUSICGEN_LOADED_NAME == model_name:
        return _MUSICGEN_MODEL, _MUSICGEN_PROCESSOR
    try:
        from transformers import MusicgenForConditionalGeneration, AutoProcessor
        log.info("[BackgroundMusic] Loading MusicGen: %s ...", model_name)
        proc  = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        model.eval()
        _MUSICGEN_MODEL       = model
        _MUSICGEN_PROCESSOR   = proc
        _MUSICGEN_LOADED_NAME = model_name
        log.info("[BackgroundMusic] MusicGen ready (%s)", model_name)
        return model, proc
    except Exception as exc:
        log.warning("[BackgroundMusic] MusicGen unavailable (%s) — using chord fallback", exc)
        _MUSICGEN_FAILED = True
        return None, None


# ─── MusicGen prompt builder ─────────────────────────────────────────────────

_GENRE_MOOD = {
    "action":      "energetic dramatic orchestral",
    "adventure":   "epic adventurous orchestral",
    "animation":   "playful whimsical orchestral",
    "biography":   "reflective gentle piano",
    "comedy":      "lighthearted playful jazz",
    "crime":       "dark tense jazz noir",
    "documentary": "calm atmospheric strings",
    "drama":       "emotional melancholic strings",
    "fantasy":     "magical ethereal orchestral",
    "history":     "grand majestic brass",
    "horror":      "eerie suspenseful dark ambient",
    "mystery":     "tense noir jazz piano",
    "romance":     "warm romantic strings",
    "sci-fi":      "futuristic atmospheric electronic",
    "sport":       "upbeat energetic brass",
    "thriller":    "tense suspenseful strings",
    "war":         "powerful dramatic orchestral",
    "western":     "dusty atmospheric guitar",
}


def _build_musicgen_prompt(movie_data: dict, goofs_data: list) -> str:
    """Derive a MusicGen prompt from actual goof content + movie metadata.

    Scans goof descriptions for action keywords (explosions, chases, fights,
    dialogue, space, water, etc.) and translates them into musical descriptors
    so the generated soundtrack *reflects* the goofs being shown on screen.
    """
    # ── Keyword → musical-texture mapping ────────────────────────────────
    _GOOF_MUSIC_CUES = {
        # physical action
        "explo":     "percussive hits, rumbling bass",
        "crash":     "percussive impacts, metallic textures",
        "chase":     "fast tempo, driving rhythm",
        "fight":     "aggressive staccato, pounding drums",
        "gun":       "sharp staccato brass, tense snare rolls",
        "shoot":     "sharp staccato brass, tense snare rolls",
        "battle":    "epic war drums, powerful brass fanfare",
        "punch":     "sharp percussive hits, aggressive rhythm",
        "run":       "fast tempo, urgent strings",
        # sports / competition
        "basketball": "upbeat driving brass, energetic crowd pulse",
        "football":   "powerful brass fanfare, driving stadium drums",
        "baseball":   "upbeat playful brass, light summer percussion",
        "soccer":     "building tension, rhythmic crowd chant",
        "boxing":     "hard-hitting percussion, tense brass stabs",
        "racing":     "fast driving rhythm, roaring engine pulse",
        "game":       "tense building momentum, rising energy",
        "team":       "driving ensemble brass, collective energy",
        "coach":      "determined steady rhythm, motivational brass",
        "player":     "energetic upbeat brass, driving beat",
        "crowd":      "swelling ensemble, collective triumphant brass",
        "score":      "triumphant brass fanfare, celebratory percussion",
        "champion":   "epic triumphant brass, soaring strings",
        "victory":    "triumphant fanfare, celebratory percussion",
        "stadium":    "epic reverberant brass, crowd ambience",
        "arena":      "powerful reverbrant brass, charged atmosphere",
        "court":      "upbeat driving energy, sharp percussive hits",
        "field":      "sweeping outdoor strings, open dynamic range",
        "tournament": "building competitive tension, martial percussion",
        # vehicles / movement
        "car":       "driving bass, rhythmic pulse",
        "plane":     "soaring strings, sweeping dynamics",
        "ship":      "deep swelling strings, nautical horn",
        "train":     "chugging rhythm, building momentum",
        "horse":     "galloping percussion, dusty guitar",
        "fly":       "airy ethereal pads, ascending melody",
        # environment / setting
        "space":     "ambient synth pads, cosmic reverb",
        "water":     "flowing arpeggios, rippling harp",
        "ocean":     "deep swelling waves of strings",
        "rain":      "soft patter textures, melancholic piano",
        "fire":      "crackling textures, rising intensity",
        "night":     "dark ambient, sparse piano notes",
        "desert":    "dusty slide guitar, sparse percussion",
        "forest":    "organic textures, woodwind melody",
        "city":      "urban groove, jazz-inflected rhythm",
        # emotion / tone
        "scream":    "dissonant stabs, rising tension",
        "cry":       "melancholic cello, gentle piano",
        "laugh":     "playful woodwinds, bouncy rhythm",
        "kiss":      "warm romantic strings, soft piano",
        "wedding":   "elegant harp, gentle strings",
        "funeral":   "somber low strings, mournful brass",
        "death":     "dark sustained chords, ominous pads",
        "ghost":     "eerie whispered textures, reversed reverb",
        "blood":     "dark pulsing bass, unsettling dissonance",
        # dialogue / character
        "phone":     "minimal piano, understated rhythm",
        "letter":    "gentle nostalgic strings",
        "child":     "innocent music box melody, soft chimes",
        "doctor":    "clinical minimal tones, steady pulse",
        "police":    "tense procedural rhythm, snare rolls",
    }

    # ── Scan goof text for matching cues ─────────────────────────────────
    import re as _re
    # Keywords ending in a truncated stem (like "explo", "crash") match as
    # prefixes.  Full words (like "space", "fire", "run") use word-boundary
    # matching to avoid false positives ("spacing", "firewall", "running").
    _PREFIX_KEYWORDS = {
        "explo", "crash", "shoot", "punch",  # intentional stems
    }
    matched_cues = []
    if goofs_data:
        goof_blob = " ".join(
            g.get("text", g.get("description", "")).lower()
            for g in goofs_data
        ).replace("'", "").replace('"', "")

        for keyword, cue in _GOOF_MUSIC_CUES.items():
            if keyword in _PREFIX_KEYWORDS:
                hit = keyword in goof_blob          # substring / prefix match
            else:
                hit = bool(_re.search(r'\b' + _re.escape(keyword) + r'\b',
                                      goof_blob))   # whole-word match
            if hit and cue not in matched_cues:
                matched_cues.append(cue)
                if len(matched_cues) >= 4:
                    break

    # ── AI genre/mood layer (Flan-T5-small) ──────────────────────────────
    # Ask a small instruction-tuned LLM to describe the film's genre and
    # mood from its title + plot.  Falls back to genre-dict lookup if the
    # model is unavailable.
    title = movie_data.get("title", "")
    plot  = movie_data.get("plot", "")
    ai_mood = _infer_genre_mood(title, plot) if title else ""

    if not ai_mood:
        # Keyword-dict fallback
        genres  = [g.lower() for g in (movie_data.get("genres") or [])]
        ai_mood = next((_GENRE_MOOD[g] for g in genres if g in _GENRE_MOOD),
                       "cinematic atmospheric")

    # ── Assemble prompt ──────────────────────────────────────────────────
    parts = ["cinematic film score"]
    if title:
        parts[0] = f"soundtrack inspired by {title}, cinematic film score"
    parts.append(ai_mood)
    if matched_cues:
        parts.append(", ".join(matched_cues))
    parts.append("no vocals, instrumental only, smooth transitions")

    prompt = ", ".join(parts)
    log.info("[BackgroundMusic] MusicGen prompt: %s", prompt)
    return prompt


# ─── MusicGen inference ──────────────────────────────────────────────────────

def _generate_musicgen_audio(model, processor, prompt: str,
                              duration_sec: float, sr_target: int) -> np.ndarray:
    """Run MusicGen and return (channels, samples) float32 at sr_target Hz.

    MusicGen natively generates at 32 kHz; result is resampled via librosa
    (preferred) or linear interpolation if librosa is not installed.
    """
    MUSICGEN_SR  = 32000
    max_new_tokens = int(duration_sec * 51.2) + 4   # ≈50 tokens/sec EnCodec

    log.info("[BackgroundMusic] MusicGen generating %.1fs (%d tokens)…",
             duration_sec, max_new_tokens)

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # audio_values: [batch, channels, samples] or [batch, samples]
    audio_np = audio_values[0].cpu().numpy().astype(np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]           # → [1, samples]

    # Trim to exact duration at native rate
    exact_32k = int(duration_sec * MUSICGEN_SR)
    if audio_np.shape[-1] > exact_32k:
        audio_np = audio_np[:, :exact_32k]

    # Resample to pipeline sample rate
    if sr_target != MUSICGEN_SR:
        try:
            import librosa
            audio_np = np.stack([
                librosa.resample(audio_np[ch], orig_sr=MUSICGEN_SR, target_sr=sr_target)
                for ch in range(audio_np.shape[0])
            ])
        except ImportError:
            n_out = int(audio_np.shape[-1] * sr_target / MUSICGEN_SR)
            x_old = np.linspace(0.0, 1.0, audio_np.shape[-1])
            x_new = np.linspace(0.0, 1.0, n_out)
            audio_np = np.stack([
                np.interp(x_new, x_old, audio_np[ch])
                for ch in range(audio_np.shape[0])
            ]).astype(np.float32)

    # Normalize
    peak = np.max(np.abs(audio_np))
    if peak > 1e-6:
        audio_np = audio_np / peak * 0.85

    log.info("[BackgroundMusic] MusicGen done — %d ch, %d samples @ %d Hz",
             audio_np.shape[0], audio_np.shape[-1], sr_target)
    return audio_np   # [channels, samples]


# ─── Chord-progression fallback (additive synthesis, zero extra deps) ────────

def _adsr(n_samples, sr, attack=0.02, decay=0.05, sustain=0.7, release=0.05):
    a = min(int(attack  * sr), n_samples)
    d = min(int(decay   * sr), max(0, n_samples - a))
    r = min(int(release * sr), max(0, n_samples - a - d))
    s = max(0, n_samples - a - d - r)
    parts = []
    if a > 0: parts.append(np.linspace(0, 1, a))
    if d > 0: parts.append(np.linspace(1, sustain, d))
    if s > 0: parts.append(np.full(s, sustain))
    if r > 0: parts.append(np.linspace(sustain, 0, r))
    if not parts: return np.zeros(n_samples)
    env = np.concatenate(parts)
    if len(env) < n_samples: env = np.pad(env, (0, n_samples - len(env)))
    return env[:n_samples]


def _note_to_freq(note):
    """Convert a MIDI note number to Hz (pure math, no library needed)."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _synth_brass(freq, duration, sr, velocity=0.8):
    n = int(duration * sr)
    if n < 1: return np.zeros(1)
    t = np.arange(n) / sr
    sig = (np.sin(2*np.pi*freq*t)*1.0 + np.sin(2*np.pi*freq*2*t)*0.6
           + np.sin(2*np.pi*freq*3*t)*0.4 + np.sin(2*np.pi*freq*4*t)*0.2
           + np.sin(2*np.pi*freq*5*t)*0.15 + np.sin(2*np.pi*freq*6*t)*0.08)
    return sig * _adsr(n, sr, attack=0.03, decay=0.08, sustain=0.75, release=0.08) * velocity


def _synth_piano(freq, duration, sr, velocity=0.8):
    n = int(duration * sr)
    if n < 1: return np.zeros(1)
    t = np.arange(n) / sr
    sig = (np.sin(2*np.pi*freq*t)*1.0 + np.sin(2*np.pi*freq*2*t)*0.5
           + np.sin(2*np.pi*freq*3*t)*0.25 + np.sin(2*np.pi*freq*4*t)*0.12
           + np.sin(2*np.pi*freq*5*t)*0.06)
    decay_time = min(duration * 0.8, 2.0)
    return sig * _adsr(n, sr, attack=0.005, decay=decay_time, sustain=0.3, release=0.1) * velocity


def _synth_strings(freq, duration, sr, velocity=0.8):
    n = int(duration * sr)
    if n < 1: return np.zeros(1)
    t = np.arange(n) / sr
    vib = 1.0 + 0.003 * np.sin(2*np.pi*5.5*t)
    sig = (np.sin(2*np.pi*freq*vib*t)*1.0 + np.sin(2*np.pi*freq*2*vib*t)*0.4
           + np.sin(2*np.pi*freq*3*vib*t)*0.2 + np.sin(2*np.pi*freq*4*vib*t)*0.1)
    return sig * _adsr(n, sr, attack=0.12, decay=0.1, sustain=0.85, release=0.15) * velocity


def _synth_generic(freq, duration, sr, velocity=0.8):
    n = int(duration * sr)
    if n < 1: return np.zeros(1)
    t = np.arange(n) / sr
    sig = (np.sin(2*np.pi*freq*t)*1.0 + np.sin(2*np.pi*freq*2*t)*0.3
           + np.sin(2*np.pi*freq*3*t)*0.1)
    return sig * _adsr(n, sr, attack=0.01, decay=0.1, sustain=0.6, release=0.05) * velocity


_GENRE_MUSIC = {
    "action":      {"tempo": 132, "root": 62, "scale": "minor",  "synth": "brass"},
    "adventure":   {"tempo": 115, "root": 60, "scale": "major",  "synth": "brass"},
    "animation":   {"tempo": 110, "root": 60, "scale": "major",  "synth": "piano"},
    "biography":   {"tempo":  80, "root": 57, "scale": "minor",  "synth": "strings"},
    "comedy":      {"tempo": 108, "root": 60, "scale": "major",  "synth": "piano"},
    "crime":       {"tempo":  88, "root": 57, "scale": "minor",  "synth": "piano"},
    "documentary": {"tempo":  85, "root": 55, "scale": "major",  "synth": "strings"},
    "drama":       {"tempo":  82, "root": 57, "scale": "minor",  "synth": "strings"},
    "fantasy":     {"tempo": 100, "root": 60, "scale": "major",  "synth": "strings"},
    "history":     {"tempo":  80, "root": 55, "scale": "major",  "synth": "brass"},
    "horror":      {"tempo":  66, "root": 57, "scale": "minor",  "synth": "strings"},
    "mystery":     {"tempo":  88, "root": 62, "scale": "minor",  "synth": "piano"},
    "romance":     {"tempo":  90, "root": 60, "scale": "major",  "synth": "strings"},
    "sci-fi":      {"tempo":  96, "root": 58, "scale": "minor",  "synth": "generic"},
    "sport":       {"tempo": 120, "root": 60, "scale": "major",  "synth": "brass"},
    "thriller":    {"tempo": 100, "root": 62, "scale": "minor",  "synth": "strings"},
    "war":         {"tempo":  90, "root": 57, "scale": "minor",  "synth": "brass"},
    "western":     {"tempo":  95, "root": 55, "scale": "major",  "synth": "brass"},
}
_DEFAULT_MUSIC = {"tempo": 100, "root": 60, "scale": "minor", "synth": "strings"}

_CHORD_PROGRESSIONS = {
    "major": [(0, 4, 7), (5, 9, 12), (7, 11, 14), (0, 4, 7)],
    "minor": [(0, 3, 7), (-1, 3, 6), (3, 7, 10), (2, 5, 9)],
}
_SYNTH_MAP = {
    "brass": _synth_brass, "piano": _synth_piano,
    "strings": _synth_strings, "generic": _synth_generic,
}


def _generate_chord_audio(movie_data: dict, goofs_data: list,
                           duration_sec: float, sr: int,
                           fade_in: float = 0.5, fade_out: float = 1.0) -> np.ndarray:
    """Contextual chord-progression — no external deps, instant generation."""
    genres = [g.lower() for g in (movie_data.get("genres") or [])]
    params = next((_GENRE_MUSIC[g] for g in genres if g in _GENRE_MUSIC), None)
    params = dict(params) if params else dict(_DEFAULT_MUSIC)

    if goofs_data:
        cats = [gf.get("category", "").lower() for gf in goofs_data]
        if any("continuity" in c for c in cats): params["tempo"] = max(60, params["tempo"] - 8)
        if any("factual"    in c for c in cats): params["scale"] = "minor"
        if any("anachronism" in c for c in cats):
            params["root"] = (params["root"] + 2) % 12 + 48

    root       = params["root"]
    scale      = params["scale"]
    synth      = _SYNTH_MAP.get(params["synth"], _synth_strings)
    progression = _CHORD_PROGRESSIONS[scale]
    chord_dur  = duration_sec / len(progression)
    total_samp = int(duration_sec * sr)
    audio      = np.zeros(total_samp, dtype=np.float64)

    for i, (r_off, t_off, f_off) in enumerate(progression):
        start = i * chord_dur
        for semitone, vel in [(root + r_off - 12, 0.55),
                               (root + t_off,      0.45),
                               (root + f_off,      0.40)]:
            rendered = synth(_note_to_freq(int(semitone)), chord_dur, sr, vel)
            si = int(start * sr)
            ei = si + len(rendered)
            if ei > total_samp:
                rendered = rendered[:total_samp - si]; ei = total_samp
            if si < total_samp:
                audio[si:ei] += rendered

    peak = np.max(np.abs(audio))
    if peak > 1e-6: audio = audio / peak * 0.85
    if fade_in  > 0:
        fi = min(int(fade_in  * sr), total_samp); audio[:fi] *= np.linspace(0, 1, fi)
    if fade_out > 0:
        fo = min(int(fade_out * sr), total_samp); audio[-fo:] *= np.linspace(1, 0, fo)
    return audio.astype(np.float32)


# ─── ComfyUI Node ────────────────────────────────────────────────────────────

class GooferBackgroundMusic:
    """
    Generates cinematic background music for the procedural clip.

    Path 1 — MusicGen 3 (facebook/musicgen-large): AI film-score prompt
              derived from movie genre + goof categories.
    Path 2 — Chord-progression fallback: instant, zero extra deps.

    Widget order (slot 0 = duration_sec):
      0  duration_sec   — master clock; also wired to ProceduralClip
      1  musicgen_model — model size / variant
      2  volume
      3  fade_in_sec
      4  fade_out_sec
      5  sample_rate
    """

    CATEGORY     = "Goofer"
    FUNCTION     = "render"
    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("music", "duration_sec")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration_sec": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 300.0, "step": 0.5,
                    "tooltip": (
                        "Duration of the music in seconds.  "
                        "Wire the 'duration_sec' OUTPUT → GooferProceduralClip "
                        "'music_duration' so both clips are exactly the same length."
                    )
                }),
            },
            "optional": {
                "movie_data": ("GOOFER_MOVIE", {
                    "tooltip": "Movie genre drives the MusicGen prompt and chord mood."
                }),
                "goofs_data": ("GOOFER_GOOFS", {
                    "tooltip": "Goof categories fine-tune the musical mood."
                }),
                "musicgen_model": ([
                    "facebook/musicgen-small",
                    "facebook/musicgen-medium",
                    "facebook/musicgen-large",
                    "facebook/musicgen-stereo-large",
                ], {
                    "default": "facebook/musicgen-large",
                    "tooltip": (
                        "MusicGen model variant.  "
                        "Falls back to chord-progression if the model cannot load."
                    )
                }),
                "volume": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Output volume (sole audio on procedural clip)"
                }),
                "fade_in_sec": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                }),
                "fade_out_sec": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5,
                }),
                "sample_rate": ("INT", {
                    "default": 48000, "min": 8000, "max": 96000, "step": 1000,
                    "tooltip": "Output sample rate — 48000 to match LTX-2 pipeline"
                }),
            },
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    def _to_comfy_audio(self, audio_np: np.ndarray, sample_rate: int, volume: float) -> dict:
        """(channels, samples) float32 → ComfyUI AUDIO dict (1, 2, samples) stereo."""
        audio_np = (audio_np * volume).astype(np.float32)
        if audio_np.shape[0] == 1:
            mono  = torch.from_numpy(audio_np[0])
            delay = int(0.0004 * sample_rate)            # 0.4 ms Haas delay
            right = torch.cat([torch.zeros(delay), mono[:-delay]])
            stereo = torch.stack([mono, right]).unsqueeze(0)
        else:
            left  = torch.from_numpy(audio_np[0])
            right = torch.from_numpy(audio_np[1])
            n     = max(left.shape[0], right.shape[0])
            if left.shape[0]  < n: left  = torch.nn.functional.pad(left,  (0, n - left.shape[0]))
            if right.shape[0] < n: right = torch.nn.functional.pad(right, (0, n - right.shape[0]))
            stereo = torch.stack([left, right]).unsqueeze(0)
        return {"waveform": stereo, "sample_rate": sample_rate}

    # ── main render ──────────────────────────────────────────────────────────

    def render(self,
               duration_sec: float = 5.0,
               movie_data=None,
               goofs_data=None,
               musicgen_model: str = "facebook/musicgen-large",
               volume: float = 0.85,
               fade_in_sec: float = 0.5,
               fade_out_sec: float = 2.0,
               sample_rate: int = 48000):

        target = float(duration_sec)

        # ── Path 1: MusicGen ─────────────────────────────────────────────
        if movie_data is not None:
            model, processor = _get_musicgen(musicgen_model)
            if model is not None:
                try:
                    prompt   = _build_musicgen_prompt(movie_data, goofs_data or [])
                    audio_np = _generate_musicgen_audio(model, processor, prompt,
                                                        target, sample_rate)
                    audio_dict = self._to_comfy_audio(audio_np, sample_rate, volume)
                    log.info("[BackgroundMusic] MusicGen → %.1fs @ vol %.0f%%",
                             target, volume * 100)
                    return (audio_dict, target)
                except Exception as exc:
                    log.warning("[BackgroundMusic] MusicGen inference failed (%s) "
                                "— falling back to chord-progression", exc)

        # ── Path 2: Chord-progression (always works) ─────────────────────
        ctx_movie = movie_data or {"genres": [], "title": "Unknown"}
        audio_np  = _generate_chord_audio(
            ctx_movie, goofs_data or [], target, sample_rate,
            fade_in=fade_in_sec, fade_out=fade_out_sec,
        )
        audio_dict = self._to_comfy_audio(audio_np[np.newaxis, :], sample_rate, volume)
        log.info("[BackgroundMusic] chord-progression → %.1fs @ vol %.0f%%",
                 target, volume * 100)
        return (audio_dict, target)
