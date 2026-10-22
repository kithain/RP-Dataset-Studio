"""Streamlit settings and shared entry points for semantic tag analysis."""
from __future__ import annotations

import json
import threading

import streamlit as st

from llm_tagging import (BASE, DEFAULT_COMFY, DEFAULT_FP8, DEFAULT_GGUF, DEFAULT_PYTHON,
                         DEFAULT_SERVER, LocalQwen, QwenConfig, QwenError, analyze_record,
                         tag_with_qwen)
from tagging import detect_tags, tag_entries


QWEN_2B = "Qwen 2B Heretic · rapide (GGUF)"
QWEN_4B = "Qwen 4B Heretic · modèle fourni (FP8)"
RULES = "Règles rapides (sans LLM)"
SETTINGS = BASE / "qwen_settings.json"


class ClientPool:
    def __init__(self):
        self.client = None
        self.lock = threading.RLock()

    def get(self, config):
        with self.lock:
            if self.client is None or self.client.config != config:
                if self.client:
                    self.client.close()
                self.client = LocalQwen(config)
            return self.client

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
            self.get(config).load()


@st.cache_resource(show_spinner=False)
def clients():
    return ClientPool()


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
        return QwenConfig(backend="comfy", model=st.session_state.qwen_model_4b,
                          python=st.session_state.qwen_python, comfy=st.session_state.qwen_comfy)
    return QwenConfig(model=st.session_state.qwen_model_2b, server=st.session_state.qwen_server)


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
                        clients().load(current_config())
                    st.success("Qwen chargé et prêt.")
                except QwenError as exc:
                    st.error(str(exc))
            if st.button("Décharger Qwen de la mémoire"):
                clients().unload()
                st.success("Modèle déchargé.")
        st.caption("Les données restent sur ce PC. Les annotations générées restent à vérifier.")


def analyze_tags(data):
    if st.session_state.get("tagging_engine", QWEN_2B) == RULES:
        return detect_tags(data)
    config = current_config()
    return analyze_record(data, lambda system, prompt: clients().generate(config, system, prompt))


def annotate_rows(rows):
    if st.session_state.get("tagging_engine", QWEN_2B) == RULES:
        return tag_entries(rows), []
    progress = st.progress(0, text="Analyse sémantique Qwen...")
    config = current_config()
    try:
        return tag_with_qwen(rows, lambda system, prompt: clients().generate(config, system, prompt),
                             lambda index, total: progress.progress(index / total,
                                 text=f"Qwen : {index} / {total} entrées analysées"))
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
