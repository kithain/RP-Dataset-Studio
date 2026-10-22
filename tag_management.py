"""Tag catalogue and reversible bulk corrections; no inference is needed here."""
from __future__ import annotations

from collections import Counter

from tagging import TAG_RULES, entry_tags, normalize_tags


def default_filters():
    return {"contains": "", "excludes": "", "languages": [], "regex": False,
            "unconvertible": False, "tags": [], "excluded_tags": [],
            "tag_mode": "all", "tag_status": "all"}


def tag_family(tag):
    if tag.casefold() in {"sfw", "nsfw", "needs review"}:
        return "Classification"
    for family, members in TAG_RULES.items():
        if tag.casefold() in {name.casefold() for name in members}:
            return family
    return "Personnage / source / tags libres"


def tag_catalogue(rows):
    """One row per case-insensitive tag, counting active entries, not occurrences."""
    names, counts = {}, Counter()
    for row in rows:
        if row.deleted:
            continue
        for tag in entry_tags(row):
            key = tag.casefold()
            names.setdefault(key, tag)
            counts[key] += 1
    return [{"Tag": names[key], "Famille": tag_family(names[key]), "Entrées": count}
            for key, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]


def plan_tag_change(rows, tag, replacement=None):
    """Preview replacement/merge or removal, using effective tags and keeping source data."""
    target = normalize_tags(replacement) if replacement is not None else []
    if replacement is not None and len(target) != 1:
        raise ValueError("Indique un seul tag de remplacement.")
    changes = []
    for row in rows:
        if row.deleted:
            continue
        current = entry_tags(row)
        if not any(value.casefold() == tag.casefold() for value in current):
            continue
        updated = normalize_tags([value for item in current
                                  for value in (target if item.casefold() == tag.casefold() else [item])])
        if updated != current:
            changes.append((row, updated))
    return changes


def apply_tag_change(changes):
    """Return the same undo snapshot format as the per-entry tag editor."""
    before = [(row.uid, list(row.auto_tags),
               None if row.tags_override is None else list(row.tags_override))
              for row, _ in changes]
    for row, updated in changes:
        row.tags_override = list(updated)
    return before
