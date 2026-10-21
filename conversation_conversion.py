"""Conservative recovery of text conversations for Unsloth exports."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any

from core import ROLE_NAMES, Entry, as_messages


def _has_non_text_fields(message: dict) -> bool:
    return any(message.get(key) for key in (
        "tool_calls", "function_call", "image", "images", "audio", "video", "attachments"))


def partial_conversion(data: dict, messages: list[dict]) -> bool:
    sequence = data.get("messages")
    if not isinstance(sequence, list):
        sequence = data.get("conversations")
    if not isinstance(sequence, list):
        return False
    if len(sequence) != len(messages):
        return True
    for message in sequence:
        if not isinstance(message, dict) or _has_non_text_fields(message):
            return True
        value = message.get("content", message.get("value", message.get("text")))
        if isinstance(value, list) and any(
            not isinstance(part, str) and not (
                isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text")
                and isinstance(part.get("text"), str)) for part in value
        ):
            return True
        if not isinstance(value, (str, list)):
            return True
    return False


@dataclass
class ConversationConversion:
    data: dict[str, Any] | None
    notes: list[str] = field(default_factory=list)
    reason: str = ""


def convert_conversation(data: dict[str, Any]) -> ConversationConversion:
    """Never invent roles/replies or remove nonempty text, images or tool calls."""
    sequence_key = None
    sequence = None
    notes = []
    for key in ("messages", "conversations", "conversation", "dialogue", "chat"):
        value = data.get(key)
        if isinstance(value, str) and value.lstrip().startswith("["):
            try:
                value = json.loads(value)
            except ValueError:
                return ConversationConversion(None, reason="Le tableau de messages contient un JSON invalide.")
            notes.append("tableau de messages décodé depuis du JSON")
        if isinstance(value, list):
            sequence_key, sequence = key, value
            break
    if sequence is None:
        for user_key, assistant_key in (("user", "assistant"), ("human", "gpt"),
                                         ("instruction", "response"), ("question", "response")):
            if user_key not in data or assistant_key not in data:
                continue
            if not all(isinstance(data.get(key), str) and data[key].strip()
                       for key in (user_key, assistant_key)):
                return ConversationConversion(None, reason="La demande ou la réponse est vide ou non textuelle.")
            messages = []
            system = data.get("system")
            if isinstance(system, str) and system.strip():
                messages.append({"role": "system", "content": system})
            prompt = data[user_key]
            if user_key == "instruction" and "input" in data:
                context = data["input"]
                if not isinstance(context, str):
                    return ConversationConversion(None, reason="Le contexte input est non textuel ; correction manuelle nécessaire.")
                if context.strip():
                    prompt += "\n\n" + context
            messages.extend([{"role": "user", "content": prompt},
                             {"role": "assistant", "content": data[assistant_key]}])
            updated = deepcopy(data)
            updated["messages"] = messages
            return ConversationConversion(updated, [f"champs {user_key}/{assistant_key} convertis en messages"])
        return ConversationConversion(None, reason="Aucun échange complet reconnu ; il faut identifier la demande et la réponse.")
    cleaned = []
    empty_count = 0
    normalized_count = 0
    for message in sequence:
        if not isinstance(message, dict):
            return ConversationConversion(None, reason="Un message n'a pas de rôle explicite ; correction manuelle nécessaire.")
        if _has_non_text_fields(message):
            return ConversationConversion(None, reason="Présence d'images, d'audio ou d'appels d'outils ; conversion texte automatique refusée.")
        role_key = next((key for key in ("role", "from", "speaker") if key in message), None)
        role = ROLE_NAMES.get(str(message.get(role_key, "")).strip().lower())
        if role is None:
            return ConversationConversion(None, reason="Rôle non reconnu ; aucun rôle n'est attribué automatiquement.")
        text_key = next((key for key in ("content", "value", "text") if key in message), None)
        if text_key is None:
            return ConversationConversion(None, reason="Un message n'a pas de champ de texte reconnu.")
        value = message[text_key]
        if value is None or (isinstance(value, str) and not value.strip()):
            empty_count += 1
            continue
        if isinstance(value, list):
            if any(not isinstance(part, str) and not (
                isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text")
                and isinstance(part.get("text"), str)) for part in value):
                return ConversationConversion(None, reason="Un message contient un bloc non textuel qui serait perdu.")
            text_parts = [part if isinstance(part, str) else part["text"] for part in value]
            if not any(part.strip() for part in text_parts):
                empty_count += 1
                continue
        elif not isinstance(value, str):
            return ConversationConversion(None, reason="Contenu non textuel ; correction manuelle nécessaire.")
        updated_message = deepcopy(message)
        if role_key != "role":
            updated_message.pop(role_key)
        if text_key != "content":
            updated_message.pop(text_key)
        normalized_count += role_key != "role" or text_key != "content" or message.get("role") != role
        updated_message["role"] = role
        updated_message["content"] = value
        cleaned.append(updated_message)
    if not any(message["role"] == "user" for message in cleaned) or not any(
            message["role"] == "assistant" for message in cleaned):
        return ConversationConversion(None, reason="Il manque une demande utilisateur ou une réponse assistant non vide.")
    updated = deepcopy(data)
    if sequence_key != "messages":
        updated.pop(sequence_key)
    updated["messages"] = cleaned
    if empty_count:
        notes.append(f"{empty_count} message(s) vide(s) retiré(s)")
    if normalized_count:
        notes.append(f"{normalized_count} rôle(s) ou champ(s) normalisé(s)")
    if sequence_key != "messages":
        notes.append(f"{sequence_key} converti en messages")
    if not notes:
        notes.append("structure des messages normalisée")
    return ConversationConversion(updated, notes)


@dataclass
class ConversionCandidate:
    uid: int
    source_line: int
    deleted: bool
    result: ConversationConversion


def analyze_conversions(rows: list[Entry], *, include_deleted: bool = False) -> list[ConversionCandidate]:
    candidates = []
    for row in rows:
        if row.deleted and not include_deleted:
            continue
        messages = as_messages(row.data)
        if messages is not None and not partial_conversion(row.data, messages):
            continue
        candidates.append(ConversionCandidate(row.uid, row.source_line, row.deleted,
                                              convert_conversation(row.data)))
    return candidates
