"""Streamlit settings and shared entry points for semantic tag analysis."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from llm_tagging import (BASE, DEFAULT_COMFY, DEFAULT_FP8, DEFAULT_GGUF, DEFAULT_PYTHON,
                         DEFAULT_SERVER, LocalQwen, QwenConfig, QwenError, analyze_record,
                         generate_json, load_with_activity, tag_with_qwen)
from activity_log import Activity, event, shared_lock
from tagging import detect_tags, tag_entries
from install_assets import resolve_setting


QWEN_2B = "Qwen 2B Heretic · rapide (GGUF)"
QWEN_4B = "Qwen 4B Heretic · modèle fourni (FP8)"
RULES = "Règles rapides (sans LLM)"
SETTINGS = BASE / "qwen_settings.json"


class ClientPool:
    def __init__(self):
        self.client = None
        self.lock = threading.RLock()

    def get(self, config):
        return configured_client(self, config)

    def unload(self):
        with self.lock:
            if self.client:
                self.client.close()
                self.client = None

    def generate(self, config, system, prompt):
        with self.lock:
            return self.get(config).generate(system, prompt)

    def load(self, config):
        with self.lock:
            load_with_activity(self.get(config))

@st.cache_resource(show_spinner=False)
def clients():
    return ClientPool()


def configured_client(pool, config):
    """Compare configuration values across hot reloads without unloading a warm model."""
    with shared_lock(pool.lock):
        if pool.client is None or asdict(pool.client.config) != asdict(config):
            if pool.client:
                event("Changement de modèle : déchargement de l'ancien moteur",
                      modele=Path(pool.client.config.model).name)
                pool.client.close()
            pool.client = LocalQwen(config)
        return pool.client


def load_config(config):
    pool = clients()
    with shared_lock(pool.lock):
        load_with_activity(configured_client(pool, config))


def unload_model():
    pool = clients()
    with shared_lock(pool.lock):
        if pool.client:
            with Activity("Déchargement du modèle", heartbeat=False,
                          modele=Path(pool.client.config.model).name):
                pool.unload()
        else:
            event("Aucun modèle à décharger")


def generate_tags(config, system, prompt, *, pool):
    with shared_lock(pool.lock), Activity("Analyse des tags", heartbeat=False):
        return configured_client(pool, config).generate(system, prompt)


def initialize_qwen_settings():
    defaults = {"tagging_engine": QWEN_2B, "qwen_model_2b": DEFAULT_GGUF,
                "qwen_server": DEFAULT_SERVER, "qwen_model_4b": DEFAULT_FP8,
                "qwen_python": DEFAULT_PYTHON, "qwen_comfy": DEFAULT_COMFY}
    try:
        saved = json.loads(SETTINGS.read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            defaults.update({key: value for key, value in saved.items()
                             if key in defaults and isinstance(value, str)})
    except (OSError, ValueError):
        pass
    if defaults["tagging_engine"] not in (QWEN_2B, QWEN_4B, RULES):
        defaults["tagging_engine"] = QWEN_2B
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_config():
    if st.session_state.tagging_engine == QWEN_4B:
        return QwenConfig(backend="comfy", model=resolve_setting(st.session_state.qwen_model_4b, BASE),
                          python=resolve_setting(st.session_state.qwen_python, BASE),
                          comfy=resolve_setting(st.session_state.qwen_comfy, BASE))
    return QwenConfig(model=resolve_setting(st.session_state.qwen_model_2b, BASE),
                      server=resolve_setting(st.session_state.qwen_server, BASE))


def render_qwen_settings():
    initialize_qwen_settings()
    with st.expander("🧠 Analyse des tags", expanded=False):
        st.selectbox("Moteur de tags", options=[QWEN_2B, QWEN_4B, RULES], key="tagging_engine")
        if st.session_state.tagging_engine == QWEN_2B:
            st.text_input("Modèle Qwen GGUF", key="qwen_model_2b")
            st.text_input("Chemin de llama-server.exe", key="qwen_server")
            st.caption("Le serveur démarre automatiquement sur ce PC, puis conserve le modèle en mémoire. "
                       "Choisir une variante Heretic adaptée aux données NSFW.")
        elif st.session_state.tagging_engine == QWEN_4B:
            st.text_input("Modèle Qwen FP8 / safetensors", key="qwen_model_4b")
            st.text_input("Python de ComfyUI", key="qwen_python")
            st.text_input("Dossier ComfyUI", key="qwen_comfy")
            st.caption("Réutilise le Python, le tokenizer et le moteur de ComfyUI. "
                       "ComfyUI peut rester fermé. Aucun téléchargement automatique.")
        else:
            st.caption("Détection par mots-clés uniquement, sans compréhension sémantique.")
        if st.button("Enregistrer la configuration des tags"):
            keys = ("tagging_engine", "qwen_model_2b", "qwen_server", "qwen_model_4b",
                    "qwen_python", "qwen_comfy")
            try:
                SETTINGS.write_text(json.dumps({key: st.session_state[key] for key in keys},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
                st.success("Configuration enregistrée sur ce PC.")
            except OSError as exc:
                st.error(f"Configuration non enregistrée : {exc}")
        if st.session_state.tagging_engine != RULES:
            if st.button("Charger / vérifier Qwen"):
                try:
                    with st.spinner("Chargement du modèle local..."):
                        load_config(current_config())
                    st.success("Qwen chargé et prêt.")
                except QwenError as exc:
                    st.error(str(exc))
            if st.button("Décharger Qwen de la mémoire"):
                unload_model()
                st.success("Modèle déchargé.")
        st.caption("Les données restent sur ce PC. Les annotations générées restent à vérifier.")


def analyze_tags(data):
    if st.session_state.get("tagging_engine", QWEN_2B) == RULES:
        return detect_tags(data)
    config = current_config()
    pool = clients()
    return analyze_record(data, lambda system, prompt: generate_tags(config, system, prompt, pool=pool))


def generate_book(system, prompt, *, schema, max_tokens):
    if st.session_state.get("tagging_engine", QWEN_2B) == RULES:
        raise QwenError("Choisis Qwen 2B ou 4B dans la barre latérale pour générer le dataset.")
    pool = clients()
    with shared_lock(pool.lock), Activity("Exemple depuis un livre", heartbeat=False):
        return generate_json(configured_client(pool, current_config()), system, prompt,
                             schema=schema, max_tokens=max_tokens)


def generate_for_config(config, system, prompt, *, schema, max_tokens, _pool=None):
    """Explicit configuration, usable by a worker without Streamlit session access."""
    pool = _pool if _pool is not None else clients()
    with shared_lock(pool.lock):
        return generate_json(configured_client(pool, config), system, prompt, schema=schema, max_tokens=max_tokens)


def annotate_rows(rows):
    if st.session_state.get("tagging_engine", QWEN_2B) == RULES:
        return tag_entries(rows), []
    progress = st.progress(0, text="Analyse sémantique Qwen...")
    config = current_config()
    try:
        pool = clients()
        def update(index, total):
            event("Progression des tags", traites=index, total=total)
            progress.progress(index / total, text=f"Qwen : {index} / {total} entrées analysées")
        with Activity("Lot d'analyse des tags", heartbeat=False, total=sum(not row.deleted for row in rows)):
            result = tag_with_qwen(rows, lambda system, prompt: generate_tags(config, system, prompt, pool=pool), update)
            event("Bilan des tags", modifies=len(result[0]), erreurs=len(result[1]))
            return result
    finally:
        progress.empty()


def report_errors(errors):
    if errors:
        st.warning(f"Qwen : {len(errors)} erreur(s). Les annotations des entrées en échec sont conservées. "
                   "Une erreur de chargement ou de connexion interrompt le lot ; les entrées suivantes "
                   "restent à analyser. Aucun remplacement automatique par les règles.")
        with st.expander("Détail des erreurs Qwen"):
            for uid, message in errors[:20]:
                st.text(f"Entrée {uid} : {message}")
