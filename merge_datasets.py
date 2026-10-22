"""Analyse and prepare multi-file datasets without modifying their sources."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import BinaryIO, Iterable

from core import Entry, as_messages, fingerprint, load_dataset
from conversation_conversion import ConversationConversion, convert_conversation, partial_conversion as _partial_conversion


@dataclass
class MergeAnalysis:
    rows: list[Entry] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    origins: dict[int, str] = field(default_factory=dict)
    incompatible: set[int] = field(default_factory=set)
    partial: set[int] = field(default_factory=set)
    fingerprints: dict[int, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    recoveries: dict[int, ConversationConversion] = field(default_factory=dict)
    recovered_fingerprints: dict[int, str] = field(default_factory=dict)


def analyze_merge(sources: Iterable[tuple[str, BinaryIO | Path]], *, repair_jsonl: bool = True) -> MergeAnalysis:
    result = MergeAnalysis()
    for name, source in sources:
        log = []
        origins = {}
        try:
            context = source.open("rb") if isinstance(source, Path) else nullcontext(source)
            with context as stream:
                rows, errors = load_dataset(stream, name, repair_jsonl=repair_jsonl,
                                            repairs=log, entry_sources=origins)
        except Exception as exc:
            rows, errors = [], [f"Impossible de lire ce fichier : {exc}"]
        active = compatible = partial = recoverable = 0
        for row in rows:
            origin = origins.get(row.uid, name)
            row.uid = len(result.rows) + 1
            result.rows.append(row)
            result.origins[row.uid] = origin
            if row.deleted:
                continue
            active += 1
            messages = as_messages(row.data)
            if messages is None:
                result.incompatible.add(row.uid)
            elif _partial_conversion(row.data, messages):
                result.partial.add(row.uid)
                partial += 1
            else:
                compatible += 1
            if row.uid in result.incompatible or row.uid in result.partial:
                conversion = convert_conversation(row.data)
                if conversion.data is not None:
                    result.recoveries[row.uid] = conversion
                    result.recovered_fingerprints[row.uid] = fingerprint(conversion.data)
                    recoverable += 1
            result.fingerprints[row.uid] = fingerprint(row.data)
        result.sources.append({"Fichier": name, "Entrées lues": len(rows), "Actives": active,
                               "Compatibles Unsloth": compatible, "Conversions incomplètes": partial,
                               "Récupérables": recoverable,
                               "Réparations": len(log), "Erreurs signalées": len(errors)})
        result.errors.extend(f"{name} — {error}" for error in errors)
        result.repairs.extend(f"{name} — {repair}" for repair in log)
    return result


@dataclass
class MergeSelection:
    rows: list[Entry]
    duplicates: int
    excluded: int
    already_deleted: int
    active: int
    convertible: int
    converted: int = 0


def prepare_merge(analysis: MergeAnalysis, *, remove_duplicates: bool = True,
                  exclude_incompatible: bool = True, shuffle: bool = True, seed: int = 42,
                  auto_convert: bool = True) -> MergeSelection:
    rows = []
    seen = set()
    duplicates = excluded = already_deleted = active = convertible = converted = 0
    for original in analysis.rows:
        deleted = original.deleted
        recovery = analysis.recoveries.get(original.uid) if auto_convert and not deleted else None
        data = recovery.data if recovery else original.data
        invalid = original.uid in analysis.incompatible or original.uid in analysis.partial
        if recovery:
            invalid = False
            converted += 1
        if deleted:
            already_deleted += 1
        elif exclude_incompatible and invalid:
            deleted = True
            excluded += 1
        else:
            signature = (analysis.recovered_fingerprints[original.uid] if recovery else
                         analysis.fingerprints[original.uid])
            if remove_duplicates and signature in seen:
                deleted = True
                duplicates += 1
            else:
                seen.add(signature)
                active += 1
                convertible += recovery is not None or original.uid not in analysis.incompatible
        rows.append(Entry(original.uid, data, original.source_line, deleted,
                          original.language_override, list(original.auto_tags),
                          list(original.tags_override) if original.tags_override is not None else None))
    if shuffle:
        retained = [row for row in rows if not row.deleted]
        Random(seed).shuffle(retained)
        rows = retained + [row for row in rows if row.deleted]
    return MergeSelection(rows, duplicates, excluded, already_deleted, active, convertible, converted)
