"""Pure, dependency-free dataset helpers for RP Dataset Studio.

All modifications occur on in-memory rows. The original files are never modified.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from tagging import entry_tags, normalize_tags


@dataclass
class Entry:
    uid: int
    data: dict[str, Any]
    source_line: int
    deleted: bool = False
    language_override: str | None = None
    auto_tags: list[str] = field(default_factory=list)
    tags_override: list[str] | None = None


LANG_CODES = {
    "fr": "Français", "en": "Anglais", "es": "Espagnol", "de": "Allemand",
    "it": "Italien", "pt": "Portugais", "nl": "Néerlandais", "ja": "Japonais",
    "zh": "Chinois", "ko": "Coréen", "ru": "Russe", "ar": "Arabe",
    "?": "Non détectée",
}


ROLE_NAMES = {
    "user": "user", "human": "user", "utilisateur": "user", "joueur": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant", "model": "assistant",
    "character": "assistant", "char": "assistant", "narrator": "assistant",
    "system": "system", "système": "system",
}


@dataclass(frozen=True)
class TextField:
    path: tuple[str | int, ...]
    role: str
    label: str
    text: str


def conversation_fields(data: dict[str, Any]) -> list[TextField]:
    """Locate original text fields without flattening or converting the record."""
    fields = []
    role_labels = {"system": "Système", "user": "Utilisateur", "assistant": "Assistant",
                   "tool": "Outil", "function": "Fonction"}
    sequence_key = next((key for key in ("messages", "conversations")
                         if isinstance(data.get(key), list)), None)
    if sequence_key:
        for index, message in enumerate(data[sequence_key]):
            if not isinstance(message, dict):
                continue
            raw_role = str(message.get("role", message.get("from", message.get("speaker", ""))))
            role = ROLE_NAMES.get(raw_role.strip().lower(), raw_role.strip().lower())
            label = f"Message {index + 1} · {role_labels.get(role, raw_role or 'Sans rôle')}"
            key = next((key for key in ("content", "value", "text") if key in message), None)
            if key is None:
                continue
            value = message[key]
            path = (sequence_key, index, key)
            if isinstance(value, str):
                fields.append(TextField(path, role, label, value))
            elif isinstance(value, list):
                for part_index, part in enumerate(value):
                    part_path = (*path, part_index)
                    if isinstance(part, str):
                        fields.append(TextField(part_path, role, f"{label} · Texte {part_index + 1}", part))
                    elif (isinstance(part, dict) and part.get("type") in
                          ("text", "input_text", "output_text") and isinstance(part.get("text"), str)):
                        fields.append(TextField((*part_path, "text"), role,
                                                f"{label} · Texte {part_index + 1}", part["text"]))
        return fields
    labels = {"system": ("system", "Consigne système"),
              "instruction": ("user", "Instruction"), "input": ("user", "Entrée / contexte"),
              "output": ("assistant", "Réponse"), "prompt": ("user", "Demande"),
              "completion": ("assistant", "Réponse"), "response": ("assistant", "Réponse"),
              "question": ("user", "Question"), "answer": ("assistant", "Réponse"),
              "text": ("", "Texte"), "content": ("", "Contenu")}
    for key, (role, label) in labels.items():
        if isinstance(data.get(key), str):
            fields.append(TextField((key,), role, label, data[key]))
    return fields


def update_conversation_text(data: dict[str, Any], edits: dict[tuple[str | int, ...], str]) -> dict[str, Any]:
    """Apply text edits to a copy, keeping roles, metadata and non-text blocks intact."""
    allowed = {field.path for field in conversation_fields(data)}
    if any(path not in allowed or not isinstance(value, str) for path, value in edits.items()):
        raise ValueError("Champ de texte invalide")
    updated = deepcopy(data)
    for path, value in edits.items():
        target = updated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return updated


def _unpack_cell(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except (ValueError, TypeError):
                pass
    return value


def _make_entries(objects: Iterable[Any], errors: list[str], source_first_line: int = 1) -> list[Entry]:
    entries: list[Entry] = []
    for line_no, item in enumerate(objects, start=source_first_line):
        if not isinstance(item, dict):
            if len(errors) < 30:
                errors.append(f"Entrée {line_no} : objet JSON attendu, obtenu {type(item).__name__}.")
            continue
        entries.append(Entry(uid=len(entries) + 1, data=item, source_line=line_no))
    return entries


def _remove_trailing_commas(text: str) -> str:
    """Remove commas before } or ], preserving every character inside strings."""
    result = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
            result.append(char)
        elif char == ",":
            following = index + 1
            while following < len(text) and text[following] in " \t\r\n":
                following += 1
            if following == len(text) or text[following] not in "}]":
                result.append(char)
        else:
            result.append(char)
    return "".join(result)


def _repair_jsonl_line(text: str) -> tuple[list[dict[str, Any]], str]:
    """Recover complete objects only; reject the whole line if any suffix is invalid."""
    cleaned = _remove_trailing_commas(text)
    decoder = json.JSONDecoder()
    objects = []
    position = 0
    separator_removed = False
    while position < len(cleaned):
        while position < len(cleaned) and cleaned[position] in " \t\r\n":
            position += 1
        if position == len(cleaned):
            break
        item, position = decoder.raw_decode(cleaned, position)
        if not isinstance(item, dict):
            raise ValueError("objet JSON attendu")
        objects.append(item)
        while position < len(cleaned) and cleaned[position] in " \t\r\n":
            position += 1
        if position < len(cleaned) and cleaned[position] == ",":
            separator_removed = True
            position += 1
    if not objects:
        raise ValueError("aucun objet JSON complet")
    reasons = []
    if len(objects) > 1:
        reasons.append(f"{len(objects)} objets JSON séparés")
    if cleaned != text or separator_removed:
        reasons.append("virgules superflues supprimées hors du texte")
    return objects, "; ".join(reasons)


def load_dataset(stream: BinaryIO, filename: str, *, repair_jsonl: bool = True,
                 repairs: list[str] | None = None,
                 entry_sources: dict[int, str] | None = None) -> tuple[list[Entry], list[str]]:
    """Import a dataset; fill entry_sources only after a project is fully validated."""
    suffix = Path(filename).suffix.lower()
    errors: list[str] = []
    if filename.lower().endswith(".rpwork.json"):
        payload = json.loads(stream.read().decode("utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("Le projet doit contenir un objet JSON")
        if payload.get("format") != "rp-dataset-studio-v1":
            raise ValueError("Format de projet RP non reconnu")
        if not isinstance(payload.get("entries"), list):
            raise ValueError("Le champ entries du projet doit être une liste")
        entries = []
        uids = set()
        for index, item in enumerate(payload["entries"], 1):
            if not isinstance(item, dict) or not isinstance(item.get("data"), dict):
                raise ValueError(f"Entrée {index} du projet : un objet data est requis")
            uid = item.get("uid")
            source_line = item.get("source_line", 0)
            deleted = item.get("deleted", False)
            language = item.get("language_override")
            auto_tags = item.get("auto_tags", [])
            tags_override = item.get("tags_override")
            if type(uid) is not int or uid < 1:
                raise ValueError(f"Entrée {index} du projet : identifiant entier positif requis")
            if uid in uids:
                raise ValueError("Identifiants du projet en double")
            if type(source_line) is not int or source_line < 0:
                raise ValueError(f"Entrée {index} du projet : numéro de ligne invalide")
            if not isinstance(deleted, bool):
                raise ValueError(f"Entrée {index} du projet : deleted doit être un booléen")
            if language is not None and (not isinstance(language, str) or language not in LANG_CODES):
                raise ValueError(f"Entrée {index} du projet : langue invalide")
            for name, value in (("auto_tags", auto_tags), ("tags_override", tags_override)):
                if name == "tags_override" and value is None:
                    continue
                if not isinstance(value, list) or any(not isinstance(tag, str) or not tag.strip()
                                                      for tag in value):
                    raise ValueError(f"Entrée {index} du projet : {name} doit être une liste de tags textuels")
            uids.add(uid)
            entries.append(Entry(uid, item["data"], source_line, deleted, language,
                                 auto_tags, tags_override))
        raw_sources = payload.get("entry_sources", {})
        if not isinstance(raw_sources, dict):
            raise ValueError("Le champ entry_sources du projet doit être un objet")
        origins = {}
        for key, name in raw_sources.items():
            try:
                uid = int(key)
            except ValueError:
                raise ValueError("Identifiant de provenance du projet invalide") from None
            if str(uid) != key or uid not in uids or not isinstance(name, str):
                raise ValueError("Provenance du projet invalide")
            origins[uid] = name
        if entry_sources is not None:
            entry_sources.clear()
            entry_sources.update(origins)
        return entries, []
    if suffix == ".jsonl":
        entries = []
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                text = raw.decode("utf-8-sig")
                repair_note = None
                try:
                    items = [json.loads(text)]
                except json.JSONDecodeError as original_error:
                    if not repair_jsonl:
                        raise
                    try:
                        items, repair_note = _repair_jsonl_line(text)
                    except ValueError:
                        raise original_error from None
                if not all(isinstance(item, dict) for item in items):
                    raise ValueError("la ligne ne contient pas un objet JSON")
                for item in items:
                    entries.append(Entry(uid=len(entries) + 1, data=item, source_line=line_no))
                if repair_note and repairs is not None:
                    repairs.append(f"Ligne {line_no} : {repair_note}.")
            except (ValueError, UnicodeDecodeError) as exc:
                if len(errors) < 30:
                    errors.append(f"Ligne {line_no} : {exc}")
        return entries, errors
    if suffix == ".json":
        doc = json.load(io.TextIOWrapper(stream, encoding="utf-8-sig"))
        if isinstance(doc, list):
            objects = doc
        elif isinstance(doc, dict):
            objects = next((doc[key] for key in ("data", "train", "rows", "records")
                            if isinstance(doc.get(key), list)), [doc])
        else:
            raise ValueError("Le JSON doit contenir un objet ou une liste")
        return _make_entries(objects, errors), errors
    if suffix in (".csv", ".tsv"):
        text = io.TextIOWrapper(stream, encoding="utf-8-sig", errors="replace", newline="")
        if suffix == ".tsv":
            delimiter = "\t"
        else:
            sample = text.read(8192)
            text.seek(0)
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
            except csv.Error:
                delimiter = ","
        reader = csv.DictReader(text, delimiter=delimiter)
        objects = ({k: _unpack_cell(v) for k, v in line.items() if k is not None}
                   for line in reader)
        return _make_entries(objects, errors, source_first_line=2), errors
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Installe pyarrow pour ouvrir un fichier Parquet") from exc
        table = pq.read_table(stream)
        return _make_entries(table.to_pylist(), errors), errors
    raise ValueError("Format inconnu (acceptés : JSONL, JSON, CSV, TSV, Parquet)")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") in ("text", "input_text", "output_text"):
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(value) if isinstance(value, (int, float, bool)) else ""


def as_messages(data: dict[str, Any]) -> list[dict[str, str]] | None:
    """Normalize common chat/ShareGPT/Alpaca/prompt-completion formats."""
    candidates = data.get("messages")
    if not isinstance(candidates, list):
        candidates = data.get("conversations")
    if isinstance(candidates, list):
        result = []
        for msg in candidates:
            if not isinstance(msg, dict):
                continue
            role_raw = msg.get("role", msg.get("from", msg.get("speaker", "")))
            role = ROLE_NAMES.get(str(role_raw).strip().lower())
            value = _as_text(msg.get("content", msg.get("value", msg.get("text"))))
            if role and value.strip():
                result.append({"role": role, "content": value})
        if any(m["role"] == "user" for m in result) and any(m["role"] == "assistant" for m in result):
            return result
        return None
    if "instruction" in data and "output" in data:
        user = _as_text(data.get("instruction"))
        detail = _as_text(data.get("input"))
        answer = _as_text(data.get("output"))
        if user.strip() and answer.strip():
            prompt = user + ("\n\n" + detail if detail.strip() else "")
            messages = []
            if _as_text(data.get("system")).strip():
                messages.append({"role": "system", "content": _as_text(data["system"])})
            messages.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])
            return messages
    for prompt_key, response_key in (("prompt", "completion"), ("prompt", "response"),
                                     ("question", "answer"), ("input", "output")):
        if prompt_key in data and response_key in data:
            question = _as_text(data.get(prompt_key))
            answer = _as_text(data.get(response_key))
            if question.strip() and answer.strip():
                messages = []
                if _as_text(data.get("system")).strip():
                    messages.append({"role": "system", "content": _as_text(data["system"])})
                messages.extend([{"role": "user", "content": question},
                                 {"role": "assistant", "content": answer}])
                return messages
    return None


def content_text(data: dict[str, Any]) -> str:
    """Text visible in searching (including incomplete records)."""
    msgs = as_messages(data)
    if msgs is not None:
        body = "\n".join(m["content"] for m in msgs)
    else:
        seq = data.get("messages", data.get("conversations"))
        if isinstance(seq, list):
            body = "\n".join(_as_text(m.get("content", m.get("value", m.get("text"))))
                             for m in seq if isinstance(m, dict))
        else:
            body = "\n".join(_as_text(data.get(field)) for field in (
                "instruction", "input", "output", "prompt", "completion", "response",
                "question", "answer", "text", "content"))
    meta = " ".join(_as_text(data.get(field)) for field in ("category", "genre", "title", "tags"))
    return body + ("\n" + meta if meta else "")


def text_for_language(data: dict[str, Any]) -> str:
    msgs = as_messages(data)
    if msgs:
        target = "\n".join(m["content"] for m in msgs if m["role"] == "assistant")
        if target:
            return target[:7000]
    return content_text(data)[:7000]


def normalize_language(value: Any) -> str | None:
    if value is None or value == "":
        return None
    lower = str(value).strip().lower().replace("_", "-")
    replacements = {"french": "fr", "français": "fr", "francais": "fr", "english": "en",
                    "anglais": "en", "spanish": "es", "español": "es", "german": "de",
                    "italian": "it", "portuguese": "pt", "japanese": "ja", "chinese": "zh"}
    lower = replacements.get(lower, lower)
    if lower.split("-")[0] in LANG_CODES:
        return lower.split("-")[0]
    return "?"


def effective_language(row: Entry) -> str:
    if row.language_override is not None:
        return row.language_override
    for field in ("language", "lang", "locale"):
        if field in row.data:
            return normalize_language(row.data.get(field)) or "?"
    return "?"


def label(data: dict[str, Any]) -> str:
    for key in ("category", "genre", "type"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value[:80]
    return "—"


def message_count(data: dict[str, Any]) -> int:
    if isinstance(data.get("messages"), list):
        return len(data["messages"])
    if isinstance(data.get("conversations"), list):
        return len(data["conversations"])
    return 2 if as_messages(data) else 0


def preview(data: dict[str, Any], limit: int = 165) -> str:
    text = re.sub(r"\s+", " ", content_text(data)).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def terms(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace("\n", ";").split(";") if part.strip()]


def apply_filters(rows: list[Entry], *, languages: list[str] | None = None,
                  contains: str = "", excludes: str = "", use_regex: bool = False,
                  include_deleted: bool = False, only_unconvertible: bool = False,
                  tags: list[str] | None = None, excluded_tags: list[str] | None = None) -> list[Entry]:
    """OR for include terms; exclusion if ANY excluded term matches."""
    positive = terms(contains)
    negative = terms(excludes)
    required = {tag.casefold() for tag in normalize_tags(tags)}
    forbidden = {tag.casefold() for tag in normalize_tags(excluded_tags)}
    if use_regex:
        yes = [re.compile(t, flags=re.IGNORECASE) for t in positive]
        no = [re.compile(t, flags=re.IGNORECASE) for t in negative]
    else:
        yes = [t.casefold() for t in positive]
        no = [t.casefold() for t in negative]
    matches = []
    for row in rows:
        if row.deleted and not include_deleted:
            continue
        if languages and effective_language(row) not in languages:
            continue
        if only_unconvertible and as_messages(row.data) is not None:
            continue
        row_tags = entry_tags(row)
        actual_tags = {tag.casefold() for tag in row_tags}
        if not required.issubset(actual_tags) or forbidden.intersection(actual_tags):
            continue
        if not yes and not no:
            matches.append(row)
            continue
        corpus = content_text(row.data) + "\n" + "\n".join(row_tags)
        if use_regex:
            if yes and not any(r.search(corpus) for r in yes):
                continue
            if any(r.search(corpus) for r in no):
                continue
        else:
            low = corpus.casefold()
            if yes and not any(t in low for t in yes):
                continue
            if any(t in low for t in no):
                continue
        matches.append(row)
    return matches


def fingerprint(data: dict[str, Any]) -> str:
    msgs = as_messages(data)
    if msgs:
        normalized = "\n".join(f"{m['role']}:{' '.join(m['content'].casefold().split())}" for m in msgs)
    else:
        normalized = " ".join(content_text(data).casefold().split())
        if not normalized:
            normalized = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def duplicate_ids(rows: list[Entry]) -> list[int]:
    seen = set()
    duplicates = []
    for row in rows:
        if row.deleted:
            continue
        mark = fingerprint(row.data)
        if mark in seen:
            duplicates.append(row.uid)
        else:
            seen.add(mark)
    return duplicates


def delete_ids(rows: list[Entry], ids: set[int]) -> list[int]:
    changed = []
    for row in rows:
        if row.uid in ids and not row.deleted:
            row.deleted = True
            changed.append(row.uid)
    return changed


def restore_ids(rows: list[Entry], ids: set[int]) -> list[int]:
    changed = []
    for row in rows:
        if row.uid in ids and row.deleted:
            row.deleted = False
            changed.append(row.uid)
    return changed


def export_jsonl(rows: list[Entry], chatml: bool = False) -> tuple[bytes, int]:
    output = io.StringIO()
    count = 0
    for row in rows:
        if row.deleted:
            continue
        item = {"messages": as_messages(row.data)} if chatml else dict(row.data)
        if not chatml and row.language_override and row.language_override != "?":
            item["language"] = row.language_override
        if not chatml and (row.auto_tags or row.tags_override is not None):
            item["tags"] = entry_tags(row)
        if chatml and item["messages"] is None:
            continue
        output.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        count += 1
    return output.getvalue().encode("utf-8"), count


def export_project(rows: list[Entry], source: str,
                   entry_sources: dict[int, str] | None = None) -> bytes:
    payload = {"format": "rp-dataset-studio-v1", "source": source,
               "entries": [{"uid": r.uid, "data": r.data, "source_line": r.source_line,
                            "deleted": r.deleted, "language_override": r.language_override,
                            "auto_tags": r.auto_tags, "tags_override": r.tags_override}
                           for r in rows],
               "entry_sources": {str(uid): name for uid, name in (entry_sources or {}).items()}}
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def counts_by_language(rows: list[Entry]) -> dict[str, int]:
    return dict(Counter(effective_language(r) for r in rows if not r.deleted))
