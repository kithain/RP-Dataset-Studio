"""Local, deterministic tag suggestions. No network calls or franchise database."""
from __future__ import annotations

import json
import re
import unicodedata
from html import unescape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core import Entry


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", unescape(text).casefold())
    return "".join(c for c in text if not unicodedata.combining(c)).replace("’", "'").replace("\xa0", " ")


# Canonical English labels, with French/English signals and word boundaries.
# These are suggestions from the record, not claims about a character's canon.
TAG_RULES = {
    "Genre": {
        "Comedy": r"comedy|comedie|humou?r(?:ous|istique)?|gags?|plaisanteries?",
        "Romance": r"romance|romantic|romantique|amoureu(?:x|se|ses)|in love|crush|sentiments pour",
        "Slice of Life": r"slice[ -]of[ -]life|tranche de vie|vie quotidienne|daily life|everyday life",
        "Fantasy": r"fantasy|fantastique|magic(?:al)?|magie|sorcier(?:e)?s?|dragons?",
        "Adventure": r"adventure|aventure|quete|quest",
        "Action": r"action|combat|battle|bataille",
        "Drama": r"drama|drame|dramatique",
        "Horror": r"horror|horreur|horrifique",
        "Sci-Fi": r"sci[ -]?fi|science[ -]fiction|spaceship|vaisseau spatial",
        "Mystery": r"mystery|mystere|enquete|detective",
    },
    "Personnage / source": {
        "Anime": r"anime|manga",
    },
    "Archétype / dynamique": {
        "Kouhai": r"kouhai|kohai",
        "Senpai dynamic": r"senpai|sempai",
        "Tsundere": r"tsundere",
        "Sadodere": r"sadodere",
        "Sadistic": r"sadistic|sadiques?",
        "Dominant": r"dominant(?:e)?|dominatrice|takes? (?:the )?lead|prend (?:toujours )?le lead",
        "Submissive": r"submissive|soumis(?:e)?",
        "Teasing": r"teas(?:e|es|ing)|taquin(?:e|es|s|er|erie)?|taquineries",
        "Playful": r"playful|joueu(?:r|rs|se|ses)|espiegles?",
        "Shy": r"shy|timide",
    },
    "Thèmes": {
        "Vore": r"vore|vorarephili[ae]|(?:swallows?|swallowing) (?:\w+ ){0,5}whole|"
                r"aval(?:e|er) (?:\w+ ){0,5}(?:tout entier|toute entiere)",
        "Predatory": r"predatory|predator|predatri(?:ce|ces)|predateur",
        "Voracious": r"voracious|voraces?|insatiables?",
        "Flirty": r"flirty|flirt(?:s|ing)?|seduct(?:eur|rice)|drague(?:r)?",
        "Violence": r"violence|violent(?:e)?|bloodshed|gore|torture",
    },
    "Traits": {
        "Athletic": r"athletic|athletique|sporti(?:f|ve)",
        "Energetic": r"energetic|energique|debordante? d'energie",
        "Popular": r"popular|populaire",
        "Petite": r"petite|menue",
        "Female": r"female|femme|fille|elle|she|her",
        "Male": r"male|homme|garcon|he|his",
        "High Schooler": r"high[ -]school(?:er| student)?|lyceen(?:ne)?|eleve du secondaire",
    },
    "Ton du bot": {
        "Roleplay": r"role[ -]?play|jeu de role|joue(?:r)? (?:le|son) role",
        "Immersive": r"immersive?|immersif|immersion",
        "Descriptive": r"descriptive?|descriptif|description detaillee|detailed description",
        "Stays in Character": r"stays? in character|never breaks? character|"
                              r"ne quitte jamais son personnage|reste (?:toujours )?dans (?:son|le) personnage",
    },
}

_RULES = [(tag, re.compile(r"(?<!\w)(?:" + pattern + r")(?!\w)"))
          for group in TAG_RULES.values() for tag, pattern in group.items()]
_EXPLICIT_NSFW = re.compile(
    r"(?<!\w)(?:nsfw|not safe for work|18\+|r[ -]?18|adult[ -]only|adults only|"
    r"sex(?:ual|uel(?:le)?|uelles|uels)?|sexually|sexe|erotic(?:a)?|erotique|porn\w*|"
    r"nudity|nudite|nude|naked|gore|gory|graphic violence|violence explicite|torture|"
    r"fetish(?:es)?|fetich\w*|bdsm|explicit content|contenu explicite)(?!\w)")
_NEGATION = re.compile(
    r"\b(?:no|not|without|sans|aucun(?:e)?|pas de|non|interdit(?:e)?s?)"
    r"(?:[ -]+[\w']+){0,2}[ -]*$")
_FORBIDDEN = re.compile(r"\s*(?:(?:is|are|est|sont)\s+)?"
                        r"(?:forbidden|prohibited|disabled|interdit(?:e)?s?)\b")
_ALIASES = {_fold(tag): tag for group in TAG_RULES.values() for tag in group}
_ALIASES.update({"sfw": "SFW", "nsfw": "NSFW", "comedie": "Comedy",
                 "tranche de vie": "Slice of Life", "manga": "Anime", "taquine": "Teasing"})


def normalize_tags(value: Any, *, resolve_classification: bool = True) -> list[str]:
    """Read source tag strings/lists without treating dictionaries as labels."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            decoded = None
        value = decoded if isinstance(decoded, list) else re.split(r"[;,\n]", unescape(value))
    if not isinstance(value, list):
        return []
    result, seen = [], set()
    for item in value:
        if not isinstance(item, str):
            continue
        tag = re.sub(r"\s+", " ", unescape(item)).strip(" #\t\r\n`")
        tag = _ALIASES.get(_fold(tag), tag)
        if tag and tag.casefold() not in seen:
            seen.add(tag.casefold())
            result.append(tag)
    # A record cannot have both ratings; a search may explicitly request/exclude both.
    return [tag for tag in result if not resolve_classification or tag != "SFW" or "NSFW" not in result]


def _containers(data: dict) -> list[dict]:
    containers, seen = [], set()
    pending = [data]
    for obj in pending:
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        containers.append(obj)
        pending.extend(obj[key] for key in ("data", "character", "metadata")
                       if isinstance(obj.get(key), dict))
    return containers


def source_tags(data: dict) -> list[str]:
    # Original exports write the effective tags at the root, including an explicit
    # empty list after a manual correction. Nested card metadata stays untouched.
    if isinstance(data.get("tags"), (list, str)):
        return normalize_tags(data["tags"])
    return normalize_tags([tag for obj in _containers(data) for tag in normalize_tags(obj.get("tags"))])


def entry_tags(row: Entry) -> list[str]:
    if row.tags_override is not None:
        return normalize_tags(row.tags_override)
    return normalize_tags(source_tags(row.data) + row.auto_tags)


def _text_parts(value: Any):
    """Only textual message/card fields; skip URLs, IDs, images and tool payloads."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value) if value.lstrip().startswith("[") else None
        except ValueError:
            decoded = None
        if isinstance(decoded, list):
            yield from _text_parts(decoded)
        else:
            yield value
    elif isinstance(value, list):
        for part in value:
            yield from _text_parts(part)
    elif isinstance(value, dict):
        if value.get("role", value.get("from")) in ("tool", "function"):
            return
        if value.get("type") not in (None, "text", "input_text", "output_text"):
            return
        for key in ("content", "value", "text"):
            if key in value:
                yield from _text_parts(value[key])
                break


_TEXT_KEYS = (
    "messages", "conversations", "conversation", "dialogue", "chat", "system", "system_prompt",
    "instruction", "input", "output", "prompt", "completion", "response", "question", "answer",
    "user", "assistant", "human", "gpt", "text", "content", "description", "personality",
    "scenario", "first_mes", "mes_example", "definition", "greeting", "creator_notes",
    "post_history_instructions", "category", "genre", "title",
)


def _affirmed(text: str, match: re.Match) -> bool:
    before = text[max(0, match.start() - 60):match.start()]
    after = text[match.end():match.end() + 35]
    return not _NEGATION.search(before) and not _FORBIDDEN.match(after)


def _names_from_text(text: str) -> list[str]:
    """Read explicitly labelled names and quoted/emphasized work titles."""
    names = []
    # Keep Markdown emphasis long enough to delimit a title inside a sentence.
    source_prefix = (r"(?:anime(?:\s*/\s*manga)?|manga|series|série|franchise)\s+"
                     r"(?:intitul[ée]e?\s+)?")
    for pattern in (
        source_prefix + r"\*{1,2}([^*\n]+)\*{1,2}(?:\s*\(([^()\n]+)\))?",
        source_prefix + r'[«"“]([^»"”\n]+)[»"”](?:\s*\(([^()\n]+)\))?',
    ):
        for match in re.finditer(pattern, unescape(text), re.I):
            names.extend(value.strip() for value in match.groups() if value and value.strip())
    plain = re.sub(r"[*_`]", "", unescape(text))
    for pattern in (
        r"^\s*(?:le personnage|personnage|character(?: name)?|nom du personnage)\s*:\s*([^,.;\n]+)",
        r"^\s*(?:source|series|série|serie|franchise|univers|fandom)\s*:\s*([^;\n]+)",
    ):
        names.extend(match.group(1).strip() for match in re.finditer(pattern, plain, re.I | re.M))
    return [name for name in names if 1 <= len(name) <= 100]


def detect_tags(data: dict[str, Any]) -> list[str]:
    """Suggest tags from explicit metadata and FR/EN clues across the full text.

    SFW means no supported NSFW signal was found, never a safety guarantee.
    An unrecognized/empty record is left unrated. No character lore is inferred.
    """
    containers = _containers(data)
    text = "\n".join(part for obj in containers for key in _TEXT_KEYS
                     for part in _text_parts(obj.get(key)))
    explicit = source_tags(data)
    corpus = _fold(text + "\n" + "\n".join(explicit))
    tags = []
    for tag, pattern in _RULES:
        if any(tag == "Stays in Character" or _affirmed(corpus, match)
               for match in pattern.finditer(corpus)):
            tags.append(tag)
    # Keep names/titles supplied by the card; don't maintain a guessed franchise database.
    for obj in containers:
        for key in ("name", "char_name", "character_name", "character", "characters",
                    "source_title", "series", "franchise", "universe", "fandom"):
            value = obj.get(key)
            names = value if isinstance(value, list) else [value]
            tags.extend(name.strip() for name in names
                        if isinstance(name, str) and 1 <= len(name.strip()) <= 100)
    tags.extend(_names_from_text(text))
    nsfw = ("NSFW" in explicit or "Vore" in tags or
            any(_affirmed(corpus, match) for match in _EXPLICIT_NSFW.finditer(corpus)))
    if nsfw:
        tags.insert(0, "NSFW")
    elif any(char.isalpha() for char in text) or "SFW" in explicit:
        tags.insert(0, "SFW")
    return normalize_tags(tags)


def tag_entries(rows: list[Entry]) -> list[tuple[int, list[str]]]:
    """Recompute automatic annotations, preserving source data and manual overrides.

    Return the previous annotations for undo. Deleted records remain untouched.
    """
    before = []
    for row in rows:
        if row.deleted:
            continue
        tags = detect_tags(row.data)
        if tags != row.auto_tags:
            before.append((row.uid, row.auto_tags))
            row.auto_tags = tags
    return before
