"""Recover incompatible local modules retained by a running Streamlit session."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent
# Dependency order matters: refresh providers before their UI consumers.
REQUIRED_API = (
    ("tagging", {"normalize_tags": ("resolve_classification",)}),
    ("core", {"purge_deleted": (), "apply_filters": ("tag_mode", "tag_status")}),
    ("tag_management", {"untagged_entries": ()}),
    ("tagging_ui", {"analyze_tag_batch": ()}),
    ("tag_browser_ui", {"render_filters": ("button_width",), "render_tag_browser": ("button_width",)}),
    ("llm_tagging", {"generate_json": (), "QwenConfig": ("low_memory",)}),
    ("qwen_ui", {"generate_book": (), "generate_for_config": ()}),
    ("book_generation", {"validate_narration": ()}),
    ("book_ui", {"render_book_page": ("generator_version",)}),
    ("finetuning", {"source_parts": (), "exclusion_report": (), "validate_instruction_text": ()}),
    ("finetuning_ui", {"render_finetuning_page": ("generator_version",), "render_exclusion_details": ("width",),
                       "render_preparation_report": ()}),
)


def _compatible(module, api):
    for name, parameters in api.items():
        value = getattr(module, name, None)
        if not callable(value):
            return False
        if parameters and not set(parameters).issubset(inspect.signature(value).parameters):
            return False
    return True


def refresh_cached_modules():
    """Reload only stale local APIs, keeping session state and Qwen resources intact.

    Fresh imports use Python's normal loader. Compatible modules are left alone;
    no Streamlit caches, user rows or model clients are cleared here.
    """
    refreshed = []
    for name, api in REQUIRED_API:
        module = sys.modules.get(name)
        if module is None:
            continue
        source = getattr(module, "__file__", None)
        if source is None or Path(source).resolve() != BASE / f"{name}.py":
            continue
        if _compatible(module, api):
            continue
        importlib.invalidate_caches()
        module = importlib.reload(module)
        if not _compatible(module, api):
            raise ImportError(f"Le module {name} sur disque ne fournit pas l'API attendue. "
                              "Vérifie que tous les fichiers de l'application sont à jour.")
        refreshed.append(name)
    return refreshed
