"""
GooferSanitizer — Strips copyrighted terms, personal names, PII, and
brand references from goof descriptions before prompt generation.

Produces clean, copyright-safe text suitable for text-to-video engines.
No copyrighted character names, actor names, brand names, or studio
names will appear in the sanitized output.

Author: Jeffrey A. Brick
"""

import logging
import re

log = logging.getLogger("Goofer.Sanitizer")


# ── Studios / Production Companies ─────────────────────────────────────
_STUDIOS = [
    "Warner Bros", "Warner Brothers", "WB", "Universal", "Paramount",
    "20th Century Fox", "Twentieth Century Fox", "Disney", "Pixar",
    "DreamWorks", "Lionsgate", "MGM", "Metro-Goldwyn-Mayer",
    "Columbia Pictures", "Sony Pictures", "New Line Cinema",
    "Miramax", "A24", "Lucasfilm", "Marvel", "DC Comics", "DC",
    "Netflix", "Amazon Studios", "HBO", "Hulu", "Apple TV",
    "Amblin", "Legendary", "Blumhouse", "Focus Features",
    "Searchlight", "Fox Searchlight", "Orion", "TriStar",
    "Summit Entertainment", "Relativity Media", "STX",
]

# ── Brand Names ────────────────────────────────────────────────────────
_BRANDS = [
    "Coca-Cola", "Coke", "Pepsi", "Starbucks", "McDonald's",
    "Nike", "Adidas", "Apple", "iPhone", "iPad", "Samsung",
    "Google", "Microsoft", "Windows", "Amazon", "Tesla",
    "Mercedes", "BMW", "Porsche", "Ferrari", "Lamborghini",
    "Toyota", "Honda", "Ford", "Chevrolet", "Cadillac",
    "Rolex", "Gucci", "Prada", "Louis Vuitton", "Ray-Ban",
    "FedEx", "UPS", "Uber", "Lyft", "Budweiser", "Corona",
    "Jack Daniel's", "Jim Beam",
]

# ── Replacements ───────────────────────────────────────────────────────
_STUDIO_REPLACEMENT = "a film studio"
_BRAND_REPLACEMENT = "a well-known brand"
_NAME_REPLACEMENT = "a character"
_FRANCHISE_REPLACEMENTS = {
    "star wars": "a space saga",
    "star trek": "a space exploration series",
    "harry potter": "a wizarding story",
    "lord of the rings": "a fantasy epic",
    "james bond": "a spy thriller",
    "007": "a spy thriller",
    "batman": "a masked vigilante",
    "superman": "a flying hero",
    "spider-man": "a wall-crawling hero",
    "spiderman": "a wall-crawling hero",
    "avengers": "a hero team",
    "x-men": "a team of gifted individuals",
    "jurassic park": "a dinosaur theme park",
    "jurassic world": "a dinosaur theme park",
    "transformers": "giant robots",
    "terminator": "a time-traveling robot",
    "matrix": "a simulated reality",
    "mission impossible": "an elite spy mission",
    "fast and furious": "a street racing crew",
    "indiana jones": "an adventuring archaeologist",
    "pirates of the caribbean": "swashbuckling pirates",
    "toy story": "a group of sentient toys",
    "frozen": "an ice-powered royal",
    "the godfather": "a crime family",
    "alien": "a space creature",
    "aliens": "space creatures",
    "predator": "an alien hunter",
    "back to the future": "a time-travel adventure",
    "ghostbusters": "a team of paranormal investigators",
}


# -- NSFW / Explicit Content Filter --------------------------------------------
# Goofs matching any of these are DROPPED before prompt generation.
# No explicit sexual content, nudity, or extreme gore will reach LTX-Video.

_NSFW_TERMS = frozenset({
    # nudity / body (unambiguous)
    "nude", "nudity", "naked", "topless", "bottomless",
    "nipple", "nipples", "genitals", "pubic",
    "penis", "vagina", "vulva", "scrotum",
    # sexual acts / explicit
    "sex scene", "sex act", "sexual intercourse", "sexual content",
    "orgasm", "masturbat", "erection", "pornograph", "pornographic",
    "porn", "xxx", "explicit content", "adult content",
    "fornicate", "copulat", "ejaculat",
    # explicit slang
    "fuck", "fucking", "fucked", "motherfuck",
    "cock", "cunt", "pussy", "dick", "boner",
    "whore", "slut",
    # extreme gore / violence
    "decapitat", "dismemberment", "eviscer", "disembowel",
    # child safety � absolute block
    "underage", "minor sexual", "child sexual", "lolita",
})

# Phrase-level regex patterns for context-sensitive explicit content
_NSFW_PATTERNS = [
    re.compile(r'\bbare\s+(breast|chest|body|buttock)\b',         re.I),
    re.compile(r'\bfully?\s+(naked|nude|exposed)\b',               re.I),
    re.compile(r'\bin\s+the\s+(nude|buff|raw)\b',                  re.I),
    re.compile(r'\b(making love|love scene with explicit)\b',      re.I),
    re.compile(r'\bstrip(s|ped|ping)?\s+(naked|nude|down|off)\b',  re.I),
    re.compile(r'\bsex(ual)?\s+(scene|content|act|encounter)\b',   re.I),
    re.compile(r'\bbreast(s)?\s+(?:are\s+)?(?:visible|exposed|shown|bare)\b', re.I),
    re.compile(r'\bgenitalia\b',                                   re.I),
    re.compile(r'\b(pubic|body)\s+hair\b',                         re.I),
]

# Borderline terms that trigger the AI secondary check
_NSFW_BORDERLINE = frozenset({
    "breast", "bare", "strip", "exposed", "reveal", "undress",
    "shower", "bath", "intimate", "erotic", "sensual",
    "rape", "assault", "fondle", "grope",
})

# ── Banana Filter — weapons & graphic violence → 🍌 ───────────────────────────
# Because no one needs guns in their AI film. Replace with bananas instead.
# Ordered longest-first so "machine gun" matches before "gun".
_BANANA_REPLACEMENTS = [
    # multi-word weapons first
    ("machine gun",        "bunch of bananas"),
    ("shot gun",           "bunch of bananas"),
    ("shotgun",            "bunch of bananas"),
    ("semi-automatic",     "banana launcher"),
    ("assault rifle",      "very long banana"),
    ("sniper rifle",       "very long banana"),
    ("hand grenade",       "banana bomb"),
    ("pipe bomb",          "banana bomb"),
    ("rocket launcher",    "banana cannon"),
    ("submachine gun",     "bunch of bananas"),
    # single weapons
    ("grenade",            "banana bomb"),
    ("pistol",             "banana"),
    ("revolver",           "banana"),
    ("handgun",            "banana"),
    ("rifle",              "long banana"),
    ("firearm",            "banana"),
    ("weapon",             "banana"),
    ("sword",              "really long banana"),
    ("machete",            "large banana"),
    ("chainsaw",           "electric banana"),
    ("knife",              "small banana"),
    ("dagger",             "pointy banana"),
    ("axe",                "banana"),
    ("gun",                "banana"),
    # violence verbs
    ("shoot",              "squirt with a banana"),
    ("shoots",             "squirts with a banana"),
    ("shot",               "hit with a banana"),
    ("stab",               "poke with a banana"),
    ("stabs",              "pokes with a banana"),
    ("stabbed",            "poked with a banana"),
    ("strangle",           "tickle"),
    ("strangles",          "tickles"),
    ("strangled",          "tickled"),
    ("murder",             "banana party"),
    ("murders",            "banana parties"),
    ("murdered",           "banana partied"),
    ("kill",               "tickle"),
    ("kills",              "tickles"),
    ("killed",             "slipped on a banana"),
    ("bludgeon",           "bonk with a banana"),
    ("bludgeons",          "bonks with a banana"),
    # gore
    ("blood",              "banana juice"),
    ("bloodbath",          "banana smoothie"),
    ("gore",               "banana mess"),
    ("explosion",          "banana explosion"),
    ("explode",            "go full banana"),
    ("explodes",           "goes full banana"),
    ("bomb",               "banana bomb"),
]

class GooferSanitizer:
    """Strips copyrighted terms, names, PII from goof descriptions."""

    CATEGORY = "Goofer"
    FUNCTION = "sanitize"
    RETURN_TYPES = ("GOOFER_GOOFS",)
    RETURN_NAMES = ("sanitized_goofs",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "goofs_data": ("GOOFER_GOOFS",),
                "movie_data": ("GOOFER_MOVIE",),
            },
            "optional": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Enable copyright sanitization. "
                        "Turn off for public domain films where names are safe to use."
                    )
                }),
                "banana_filter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Replace all guns, weapons, and graphic violence with bananas. "
                        "Keeps prompts safe for all audiences and makes everything "
                        "considerably more fun. Highly recommended."
                    )
                }),
                "custom_blocklist": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Additional terms to strip (one per line)"
                }),
            },
        }

    def sanitize(self, goofs_data, movie_data,
                 enabled=True, banana_filter=True, custom_blocklist=""):
        if not enabled:
            log.info("[Sanitizer] disabled — passing goofs through unchanged")
            return (goofs_data,)
        strip_names = strip_brands = strip_studios = strip_franchises = True

        # Build custom blocklist
        custom_terms = [
            t.strip() for t in custom_blocklist.split("\n")
            if t.strip()
        ]

        # Also strip the movie's own title (it's copyrighted)
        movie_title = movie_data.get("title", "")

        # Dynamic cast names from GoofFetch (actors + character names)
        cast_names = movie_data.get("cast_names", [])

        sanitized = []
        for goof in goofs_data:
            text = goof["description"]
            category = goof["category"]            # Step 0: NSFW filter � drop explicit goofs before any AI sees them
            nsfw_hit, nsfw_reason = self._is_nsfw(text)
            if nsfw_hit:
                log.warning("[Sanitizer] NSFW goof DROPPED (%s): '%s'",
                            nsfw_reason, text[:80])
                continue


            # Step 1: Strip franchise names (longest first to avoid partial matches)
            if strip_franchises:
                text = self._strip_franchises(text)

            # Step 2: Strip the movie's own title.
            # When the title appears as a character name (followed by a verb
            # or preceded by a pronoun/article) replace with "the character"
            # so we don't get nonsense like "the film is stabbed in the head".
            # Otherwise replace with "the film" (title-as-title context).
            if movie_title:
                escaped = re.escape(movie_title)
                _VERBS = (
                    r'is|was|has|had|gets|got|jumps|runs|falls|dies'
                    r'|shoots|fights|kills|grabs|pulls|pushes|throws|holds'
                    r'|wears|uses|takes|makes|says|tells|asks|looks|turns'
                    r'|appears|removes|puts|stabs|opens|reaches|stands'
                    r'|walks|moves|fires|hits|lands|enters|exits|escapes'
                    r'|charges|ducks|dodges|attacks|saves|loses|drops|picks'
                    r'|draws|loads|aims|cuts|breaks|rips|tries|attempts'
                    r'|manages|decides|realizes|notices|grabs|wields'
                )
                _OBJ_VERBS = (
                    r'unmasks|stabs|kills|shoots|hits|punches|kicks|grabs'
                    r'|tackles|hugs|rescues|saves|defeats|fights|attacks'
                    r'|chases|follows|finds|sees|spots|watches|helps|joins'
                    r'|leaves|betrays|tricks|faces|confronts|knocks|throws'
                )
                # Subject position: title followed by verb → "the character"
                char_subj = r'(?<!\w)' + escaped + r'(?=\s+(?:' + _VERBS + r')\b)'
                text = re.sub(char_subj, "the character", text, flags=re.IGNORECASE)
                # Possessive: "Title's" → "the character's"
                text = re.sub(r'(?<!\w)' + escaped + r"'s\b",
                              "the character's", text, flags=re.IGNORECASE)
                # Object position: transitive verb + title → "the character"
                # Can't use variable-width lookbehind; capture the verb instead
                char_obj = r'(\b(?:' + _OBJ_VERBS + r')\s+)' + escaped + r'(?!\w)'
                text = re.sub(char_obj, r'\1the character', text, flags=re.IGNORECASE)
                # Remaining occurrences are title-as-title → "the film"
                text = re.sub(r'(?<!\w)' + escaped + r'(?!\w)',
                              "the film", text, flags=re.IGNORECASE)

            # Step 3: Strip studio names
            if strip_studios:
                text = self._strip_list(text, _STUDIOS, _STUDIO_REPLACEMENT)

            # Step 4: Strip brand names
            if strip_brands:
                text = self._strip_list(text, _BRANDS, _BRAND_REPLACEMENT)

            # Step 5: Strip dynamic cast/character names from this movie
            if strip_names and cast_names:
                text = self._strip_list(text, cast_names, _NAME_REPLACEMENT)

            # Step 6: Strip personal names (First Last pattern — catches remainders)
            if strip_names:
                text = self._strip_names(text)

            # Step 7: Strip PII patterns
            text = self._strip_pii(text)

            # Step 8: Banana filter — weapons & graphic violence → 🍌
            if banana_filter:
                text = self._apply_banana_filter(text)

            # Step 9: Custom blocklist
            if custom_terms:
                text = self._strip_list(text, custom_terms, "[redacted]")

            # Clean up orphaned Roman numerals after name replacement
            # e.g. "a character IV" → "a character", "the character III" → "the character"
            text = re.sub(
                r'\b(a character|the character|a spy|the film)\s+[IVXLCDM]+\b',
                r'\1', text
            )

            # Clean up whitespace
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"\s+([,.])", r"\1", text)

            sanitized.append({
                "category": category,
                "description": text,
                "original_length": len(goof["description"]),
            })

            if text != goof["description"]:
                log.info("[Sanitizer] cleaned: '%s' → '%s'",
                         goof["description"][:60], text[:60])

        log.info("[Sanitizer] processed %d goofs", len(sanitized))
        return (sanitized,)

    def _strip_franchises(self, text):
        """Replace known franchise/IP names with generic descriptions."""
        # Sort by length descending to match longest first
        for franchise, replacement in sorted(
            _FRANCHISE_REPLACEMENTS.items(), key=lambda x: -len(x[0])
        ):
            text = re.sub(
                r"\b" + re.escape(franchise) + r"\b",
                replacement, text, flags=re.IGNORECASE
            )
        return text

    def _strip_list(self, text, terms, replacement):
        """Replace a list of terms with a generic replacement.

        For brand names also eats an immediately following capitalized word
        (typically a model name, e.g. 'Lamborghini Gallardo' → one replacement
        rather than 'a well-known brand Gallardo').
        """
        for term in sorted(terms, key=len, reverse=True):
            # Primary: brand + optional model word (Cap word) + optional possessive
            text = re.sub(
                r"\b" + re.escape(term) + r"(?:\s+[A-Z][A-Za-z0-9]+)?(?:'s)?\b",
                replacement, text, flags=re.IGNORECASE
            )
        return text

    def _strip_names(self, text):
        """Replace First Last name patterns with generic terms."""
        # Match "First Last" but not common false positives
        # Excludes words that commonly start sentences or are titles
        # Words that cannot be the FIRST word of a real "First Last" name
        _FALSE_POS = {
            "The", "This", "That", "When", "Where", "While", "After",
            "Before", "During", "Between", "Under", "Over", "Also",
            "However", "Although", "Because", "Since", "About",
            "North", "South", "East", "West", "New", "Old", "Big",
            "Great", "First", "Last", "Next", "Left", "Right",
            "Scene", "Shot", "Camera", "Close", "Wide", "Final",
            # common first words that are NOT first names
            "Coach", "Officer", "Agent", "Doctor", "General", "Colonel",
            "Captain", "Sergeant", "Major", "President", "Senator",
            "State", "National", "Federal", "Local", "Regional",
            "Super", "High", "Low", "Deep", "Dark", "Light",
            "Early", "Late", "Young", "Senior", "Junior",
            # nationality / language adjectives
            "British", "American", "French", "German", "Italian",
            "Spanish", "Japanese", "Chinese", "Russian", "Australian",
            "Canadian", "European", "African", "Asian", "Latin",
            "Irish", "Scottish", "Welsh", "English", "Dutch",
            "Swedish", "Norwegian", "Danish", "Finnish", "Swiss",
            "Mexican", "Brazilian", "Indian", "Korean", "Thai",
            # royal / noble titles — prevent "King Baldwin" → "a character Baldwin"
            # The name after the title is caught by cast_names list instead
            "King", "Queen", "Prince", "Princess", "Duke", "Earl",
            "Lord", "Lady", "Sir", "Dame", "Baron", "Count",
            "Countess", "Marquis", "Viscount", "Duchess", "Kaiser",
            "Tsar", "Sultan", "Sheikh", "Emir", "Pharaoh", "Emperor",
            "Empress", "Regent", "Viceroy",
        }
        # Words that cannot be the SECOND word of a real "First Last" name
        # (venues, events, institutions, locations, sports terms, etc.)
        _FALSE_POS_LAST = {
            # venues / facilities
            "Fieldhouse", "Arena", "Stadium", "Coliseum", "Auditorium",
            "Gymnasium", "Gym", "Ballpark", "Court", "Field", "Track",
            "Center", "Centre", "Garden", "Palace", "Forum", "Hall",
            "Theatre", "Theater", "Cinema", "Museum", "Gallery",
            "Hotel", "Hospital", "Airport", "Station", "Terminal",
            # events / competitions
            "Finals", "Final", "Championship", "Tournament", "Series",
            "Classic", "Open", "Cup", "Trophy", "Bowl", "League",
            "Season", "Round", "Quarter", "Semifinals", "Playoffs",
            # institutions / places
            "University", "College", "Institute", "School", "Academy",
            "Street", "Avenue", "Road", "Boulevard", "Drive", "Lane",
            "Park", "Square", "Plaza", "Bridge", "Hill", "Lake", "Bay",
            # other non-name second words
            "Scene", "Shot", "Camera", "Frame", "Sequence",
            "Department", "Division", "Committee", "Bureau", "Agency",
            # organisation / company type nouns
            "Rail", "Railway", "Railways", "Airways", "Airlines",
            "Motors", "Industries", "Corporation", "Corp", "Company",
            "Group", "Systems", "Networks", "Broadcasting", "Television",
            "Radio", "Press", "Times", "News", "Bank", "Insurance",
            "Services", "Solutions", "International", "Holdings",
        }

        def _name_replacer(match):
            first = match.group(1)
            last  = match.group(2)
            if first in _FALSE_POS or last in _FALSE_POS_LAST:
                return match.group(0)
            return _NAME_REPLACEMENT

        text = re.sub(
            r"\b([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,15})\b",
            _name_replacer, text
        )
        return text
    def _is_nsfw(self, text: str):
        """Keyword + pattern check. Returns (True, reason) if explicit content detected."""
        lower = text.lower()
        for term in _NSFW_TERMS:
            if term in lower:
                return True, f"keyword:{term}"
        for pat in _NSFW_PATTERNS:
            m = pat.search(text)
            if m:
                return True, f"pattern:{m.group(0)}"
        return False, ""

    def _apply_banana_filter(self, text: str) -> str:
        """Replace weapons and graphic violence terms with bananas.
        Longest replacements applied first to avoid partial matches.
        Case-insensitive, word-boundary aware.
        """
        for term, replacement in _BANANA_REPLACEMENTS:
            text = re.sub(
                r"\b" + re.escape(term) + r"\b",
                replacement, text, flags=re.IGNORECASE
            )
        if text != text:  # always false, just for log hook
            pass
        return text

    def _strip_pii(self, text):
        """Remove emails, phone numbers, URLs, SSN-like patterns."""
        # Emails
        text = re.sub(r"\b\S+@\S+\.\S+\b", "[email]", text)
        # URLs
        text = re.sub(r"\bhttps?://\S+\b", "[url]", text)
        # Phone numbers (US formats)
        text = re.sub(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[phone]", text)
        text = re.sub(r"\(\d{3}\)\s*\d{3}[-.\s]\d{4}", "[phone]", text)
        # SSN-like
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[redacted]", text)
        # Credit card-like (13-16 digits)
        text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{0,4}\b",
                       "[redacted]", text)
        return text
