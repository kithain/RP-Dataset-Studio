"""Local TXT import, generation preview, checkpoints and exports."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from book_generation import (MODES, VERSION, create_job, decode_book, job_rows,
                             restore_checkpoint, run_job, save_checkpoint, tasks)
from core import export_jsonl, export_project
from llm_tagging import BASE
from qwen_ui import RULES, current_config, generate_book


def render_book_page(on_load, *, generator_version=VERSION):
    st.title("📖 Créer un dataset depuis un livre")
    st.write("Transforme un livre TXT en scènes de RP et en exemples de narration avec ton Qwen local.")
    upload = st.file_uploader("Livre au format TXT", type=["txt"], key="book_upload")
    local = st.text_input("Ou chemin local du livre TXT", key="book_path",
                          placeholder=r"D:\Livres\mon_livre.txt")
    title = st.text_input("Titre du livre (facultatif)", key="book_title")
    labels = st.multiselect("Exemples à créer", list(MODES.values()), default=list(MODES.values()),
                            key="book_modes")
    modes = [mode for mode, label in MODES.items() if label in labels]
    with st.expander("Lecture et découpage"):
        encoding_label = st.selectbox("Encodage du TXT", ["UTF-8", "UTF-16", "Windows-1252"],
                                      key="book_encoding")
        size = st.number_input("Taille maximale d'un passage (caractères)", 800, 4000, 2400, 100,
                                key="book_size")
        restart = st.checkbox("Recommencer la génération depuis zéro", key="book_restart",
                               help="Au prochain clic sur Préparer, remplace la progression de ce livre et de ces réglages.")
    encoding = {"UTF-8": "utf-8-sig", "UTF-16": "utf-16", "Windows-1252": "cp1252"}[encoding_label]
    engine = st.session_state.tagging_engine
    config = current_config()
    model = f"{engine} / {Path(config.model).name}"
    st.caption(f"Modèle : {model}. Pour changer, ouvre Analyse des tags dans la barre latérale, "
               "sélectionne Qwen 4B ou modifie le chemin du GGUF, puis prépare à nouveau le livre.")
    if engine == RULES:
        st.info("La génération nécessite Qwen ; sélectionne le moteur 2B ou 4B dans la barre latérale.")
    path = Path(local.strip().strip('"')) if local.strip() else None
    try:
        stat = path.stat() if path else None
        source_key = (upload.name, upload.size, upload.file_id) if upload else (
            str(path), stat.st_size if stat else None, stat.st_mtime_ns if stat else None)
    except OSError:
        source_key = (str(path), None, None)
    signature = (source_key, title, tuple(modes), size, encoding, model)
    if st.button("Préparer le livre / reprendre la sauvegarde", type="primary",
                 disabled=(upload is None and path is None) or not modes or engine == RULES):
        try:
            if upload:
                filename, blob = upload.name, upload.getvalue()
            else:
                if path.suffix.lower() != ".txt":
                    raise ValueError("Choisis un fichier .txt.")
                filename, blob = path.name, path.read_bytes()
            text = decode_book(blob, encoding)
            job = create_job(text, filename, title, modes, int(size), model)
            checkpoint = BASE / "data" / "book_generation" / f"{job['id']}.bookwork.json"
            if checkpoint.is_file() and not restart:
                restore_checkpoint(job, checkpoint.read_bytes())
            save_checkpoint(job, checkpoint)
            st.session_state.book_job = job
            st.session_state.book_signature = signature
            st.session_state.book_checkpoint = str(checkpoint)
        except (OSError, ValueError) as exc:
            st.error(f"Livre non préparé : {exc}")
    job = st.session_state.get("book_job")
    if job is None:
        st.caption("Un exemple de chaque type sélectionné sera créé par passage exploitable. "
                   "Commence par un aperçu pour contrôler le résultat.")
        return
    params = job["parameters"]
    st.subheader(f"Livre préparé : {params['title']}")
    st.caption(f"{params['filename']} · {len(job['passages'])} passage(s) · "
               f"{', '.join(MODES[mode] for mode in params['modes'])}")
    legacy = params.get("version") != generator_version
    stale = legacy or signature != st.session_state.get("book_signature")
    if legacy:
        st.warning("Cette préparation utilise l'ancien générateur. Clique sur Préparer pour appliquer les "
                   "corrections ; la nouvelle progression sera séparée de l'ancienne sauvegarde.")
    if stale:
        st.info("Le fichier ou les réglages ont changé. Clique sur Préparer pour les utiliser. "
                "Les résultats affichés ci-dessous proviennent encore du livre préparé.")
    st.write("**Narration et style** : Qwen rédige la consigne ; la réponse conserve exactement le passage "
             "original. **RP et personnages** : Qwen sélectionne des répliques originales et identifie "
             "le personnage joué. Leur texte et leur ordre sont vérifiés ; relis l'attribution des voix. "
             "Les sommaires, mentions d'édition et playlists détectés sont écartés automatiquement.")
    with st.expander("Premier passage source"):
        st.text(job["passages"][0]["text"])
    total = len(tasks(job))
    done = len(job["results"])
    a, b = st.columns(2)
    preview = a.button("Générer un aperçu", disabled=stale or done == total or engine == RULES)
    full = b.button("Générer tout le livre / reprendre", type="primary",
                     disabled=stale or done == total or engine == RULES)
    if preview or full:
        bar = st.progress(done / total, text="Génération locale en cours...")
        try:
            run_job(job, generate_book, limit=len(params["modes"]) if preview else None,
                    save=lambda current: save_checkpoint(current, st.session_state.book_checkpoint),
                    progress=lambda count, amount: bar.progress(count / amount,
                        text=f"{count} / {amount} exemples traités"))
        except OSError as exc:
            st.error(f"Sauvegarde impossible : {exc}. Les résultats restent en mémoire ; télécharge-les ci-dessous.")
        finally:
            bar.empty()
    rows, sources = job_rows(job)
    done = len(job["results"])
    skipped = [(key, item["skipped"]) for key, item in job["results"].items() if "skipped" in item]
    a, b, c = st.columns(3)
    a.metric("Exemples créés", len(rows))
    b.metric("Passages/types traités", f"{done} / {total}")
    c.metric("Exemples écartés", len(skipped))
    if done == total:
        st.success("Traitement terminé. Relis les exemples avant l'entraînement.")
    st.caption("La progression est sauvegardée localement après chaque résultat. Après fermeture, "
               "prépare le même fichier avec les mêmes réglages pour reprendre. Les erreurs restent à réessayer.")
    if job["errors"]:
        st.warning(f"{len(job['errors'])} erreur(s). Les réussites sont conservées. Une erreur de moteur "
                   "arrête la passe ; utilise Reprendre après correction.")
        with st.expander("Erreurs de génération", expanded=True):
            for key, message in list(job["errors"].items())[:30]:
                index, mode = key.split(":")
                st.text(f"Passage {int(index) + 1}, {MODES[mode]} : {message}")
    if skipped:
        with st.expander("Exemples écartés et motifs"):
            for key, reason in skipped[:50]:
                index, mode = key.split(":")
                st.text(f"Passage {int(index) + 1}, {MODES[mode]} : {reason}")
    if not rows:
        return
    st.subheader("Relire les exemples")
    selected = st.selectbox("Exemple généré", range(len(rows)),
                            format_func=lambda i: sources[rows[i].uid], key=f"book_preview_{job['id']}")
    row = rows[selected]
    for message in row.data["messages"]:
        st.markdown(f"**{message['role']}**")
        st.text(message["content"])
    with st.expander("Passage source et provenance"):
        st.json(row.data["book_source"])
    st.caption("Tous les exemples portent Needs Review. Le contrôle automatique vérifie la structure, "
               "la présence du nom du personnage et les répliques originales. Un contrôle de langue et de "
               "vocabulaire écarte certaines consignes hors sujet ; vérifie encore la consigne et les rôles.")
    name = Path(params["filename"]).stem
    original, _ = export_jsonl(rows)
    chatml, _ = export_jsonl(rows, chatml=True)
    st.download_button("Télécharger le JSONL Unsloth", chatml, f"{name}_unsloth.jsonl", "application/x-ndjson")
    st.download_button("Télécharger le JSONL avec les sources", original, f"{name}_sources.jsonl", "application/x-ndjson")
    st.download_button("Télécharger le projet modifiable", export_project(rows, f"{name}_livre.jsonl", sources),
                       f"{name}.rpwork.json", "application/json")
    if st.session_state.get("dataset_name"):
        st.info("Ouvrir ces exemples remplacera le dataset en mémoire. Exporte le projet actuel pour "
                "conserver ses changements avant de continuer.")
    if st.button("Ouvrir les exemples dans l'éditeur"):
        on_load(rows, f"{name}_livre.jsonl", [], entry_sources=sources)
        st.session_state.navigate_after_action = "🔎 Explorer"
        st.rerun()
