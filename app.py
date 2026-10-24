"""Local browser UI for reviewing and cleaning RP training datasets."""
from __future__ import annotations

import io
import json
import math
import re
from html import escape
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from runtime_imports import refresh_cached_modules

# A running Streamlit process can retain pre-update modules after app.py changes.
refresh_cached_modules()

from split_export import export_split_zip
from merge_ui import render_merge_page
from book_ui import render_book_page
from finetuning_ui import render_finetuning_page, PAGE as FINETUNING_PAGE
from conversion_ui import render_conversion_panel
from tagging import entry_tags
from tagging_ui import render_tag_editor, render_tag_panel
from tag_management import default_filters
from tag_browser_ui import render_filters, render_filtered_export, render_tag_browser
from qwen_ui import analyze_tags, annotate_rows, render_qwen_settings, report_errors
from llm_tagging import QwenError
from core import (Entry, LANG_CODES, apply_filters, as_messages, content_text,
                  conversation_fields, update_conversation_text,
                  counts_by_language, delete_ids, duplicate_ids, effective_language,
                  export_jsonl, export_project, label, load_dataset, message_count,
                  preview, purge_deleted, restore_ids, text_for_language)

st.set_page_config(page_title="RP Dataset Studio", page_icon="🎭", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container {padding-top: 1.3rem; max-width: 1550px;}
div[data-testid='stMetric'] {border:1px solid rgba(122,122,122,.18); border-radius:12px;
 padding:12px 16px; background:rgba(120,120,120,.055)}
.stButton button {border-radius:9px;}
.conversation-body {white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.75;
 font-family:inherit; font-size:1rem;}
div[data-testid='stTextArea'] textarea {line-height:1.65; unicode-bidi:plaintext;}
</style>""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent


def initialize():
    defaults = {"rows": [], "dataset_name": None, "errors": [], "repairs": [], "history": [],
                "revision": 0, "filters": default_filters(), "page": 1}
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def clear_exports():
    for key in ("export_original", "export_chatml", "export_project_data", "export_parts", "export_filtered"):
        st.session_state.pop(key, None)


def loaded(rows: list[Entry], name: str, errs: list[str], repairs: list[str] | None = None,
           entry_sources: dict[int, str] | None = None):
    clear_exports()
    st.session_state.rows = rows
    st.session_state.dataset_name = name
    st.session_state.errors = errs
    st.session_state.repairs = repairs or []
    st.session_state.history = []
    st.session_state.revision += 1
    st.session_state.filters = default_filters()
    st.session_state.filter_revision = st.session_state.get("filter_revision", 0) + 1
    st.session_state.explorer_page = 1
    st.session_state.page = 1
    st.session_state.pop("merge_report", None)
    st.session_state.entry_sources = dict(entry_sources or {})
    for key in ("conversion_preview", "conversion_signature", "conversion_notice", "tagging_notice", "tagging_errors",
                "tag_manager_notice", "purge_notice"):
        st.session_state.pop(key, None)


def mutation(action: dict):
    clear_exports()
    st.session_state.history.append(action)
    if len(st.session_state.history) > 30:
        st.session_state.history.pop(0)
    st.session_state.revision += 1


def delete_some(ids: set[int]):
    changed = delete_ids(st.session_state.rows, ids)
    if changed:
        mutation({"kind": "delete", "ids": changed})
    return len(changed)


def empty_trash():
    removed_ids = purge_deleted(st.session_state.rows)
    if not removed_ids:
        return
    clear_exports()
    # Undo snapshots and conversion previews may retain full copies of deleted records.
    st.session_state.history = []
    for key in ("conversion_preview", "conversion_signature", "conversion_notice",
                "merge_analysis", "merge_signature"):
        st.session_state.pop(key, None)
    st.session_state.entry_sources = {uid: source for uid, source in
        st.session_state.get("entry_sources", {}).items() if uid not in removed_ids}
    st.session_state.tagging_errors = [(uid, message) for uid, message in
        st.session_state.get("tagging_errors", []) if uid not in removed_ids]
    st.session_state.revision += 1
    st.session_state.filter_revision = st.session_state.get("filter_revision", 0) + 1
    st.session_state.explorer_page = 1
    st.session_state.purge_notice = (
        f"{len(removed_ids)} entrée(s) retirée(s) définitivement du projet en mémoire. "
        "L'historique d'annulation a été effacé. Sauvegarde le projet pour conserver cette purge.")


def undo():
    if not st.session_state.history:
        return
    clear_exports()
    action = st.session_state.history.pop()
    if action["kind"] == "delete":
        restore_ids(st.session_state.rows, set(action["ids"]))
    elif action["kind"] == "restore":
        delete_ids(st.session_state.rows, set(action["ids"]))
    elif action["kind"] == "edit":
        item = next(e for e in st.session_state.rows if e.uid == action["id"])
        item.data = action["before"]
        if "before_auto_tags" in action:
            item.auto_tags = action["before_auto_tags"]
    elif action["kind"] == "tags":
        before = {uid: (tags, override) for uid, tags, override in action["before"]}
        for row in st.session_state.rows:
            if row.uid in before:
                row.auto_tags, row.tags_override = before[row.uid]
    elif action["kind"] == "lang":
        before = dict(action["before"])
        for row in st.session_state.rows:
            if row.uid in before:
                row.language_override = before[row.uid]
    elif action["kind"] == "convert":
        before = {uid: (data, deleted) for uid, data, deleted in action["before"]}
        for row in st.session_state.rows:
            if row.uid in before:
                row.data, row.deleted = before[row.uid]
    st.session_state.revision += 1


def import_file(source, name: str):
    try:
        repairs = []
        entry_sources = {}
        with st.spinner(f"Lecture de {name}..."):
            rows, errors = load_dataset(
                source, name, repair_jsonl=st.session_state.get("repair_jsonl_import", True),
                repairs=repairs, entry_sources=entry_sources)
        if not rows:
            st.error("Aucune entrée valide n'a été trouvée dans ce fichier.")
            for issue in errors:
                st.error(issue)
            return
        tagging_errors = []
        if st.session_state.get("auto_tags_import", True) and not name.lower().endswith(".rpwork.json"):
            with st.spinner("Attribution des tags automatiques..."):
                _, tagging_errors = annotate_rows(rows)
        loaded(rows, name, errors, repairs, entry_sources)
        st.session_state.tagging_errors = tagging_errors
        st.rerun()
    except Exception as exc:
        st.error(f"Impossible d'ouvrir le fichier : {exc}")


def language_detection():
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError:
        st.error("Le détecteur est absent. Relance install.bat pour installer les dépendances.")
        return
    languages = {
        Language.FRENCH: "fr", Language.ENGLISH: "en", Language.SPANISH: "es",
        Language.GERMAN: "de", Language.ITALIAN: "it", Language.PORTUGUESE: "pt",
        Language.DUTCH: "nl", Language.JAPANESE: "ja", Language.CHINESE: "zh",
        Language.KOREAN: "ko", Language.RUSSIAN: "ru", Language.ARABIC: "ar"}
    with st.spinner("Chargement du détecteur local..."):
        detector = LanguageDetectorBuilder.from_languages(*languages.keys()).build()
    targets = [row for row in st.session_state.rows
               if not row.deleted and effective_language(row) == "?"]
    if not targets:
        st.info("Toutes les entrées actives ont déjà une langue renseignée.")
        return
    bar = st.progress(0, text="Détection en cours...")
    detected = Counter()
    before = [(row.uid, row.language_override) for row in targets]
    for i, row in enumerate(targets, 1):
        text = text_for_language(row.data)
        letters = sum(c.isalpha() for c in text)
        result = detector.detect_language_of(text) if letters >= 35 else None
        code = languages.get(result, "?")
        row.language_override = code
        detected[code] += 1
        if i % max(1, len(targets) // 100) == 0 or i == len(targets):
            bar.progress(i / len(targets), text=f"Analysé {i:,} / {len(targets):,}".replace(",", " "))
    mutation({"kind": "lang", "before": before})
    st.session_state.language_detection_summary = (
        f"Détection terminée : {len(targets)} conversations analysées, "
        f"{len(targets) - detected['?']} langues attribuées, "
        f"{detected['?']} entrées encore inconnues. "
        "Vérifie les résultats sur des textes courts ou multilingues.")
    return True


def row_table(rows: list[Entry], *, prefix: str, page_size: int = 50) -> set[int]:
    if not rows:
        st.info("Aucune entrée à afficher.")
        return set()
    n_pages = max(1, math.ceil(len(rows) / page_size))
    page_key = f"{prefix}_page"
    if page_key not in st.session_state or st.session_state[page_key] > n_pages:
        st.session_state[page_key] = 1
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.number_input(f"Page (sur {n_pages})", min_value=1, max_value=n_pages,
                        key=page_key, step=1)
    with col_b:
        st.caption(f"{len(rows):,} entrées · {page_size} par page".replace(",", " "))
    page = st.session_state[page_key]
    shown = rows[(page - 1) * page_size:page * page_size]
    data = [{"Cocher": False, "ID": r.uid, "Langue": effective_language(r),
             "Messages": message_count(r.data), "Catégorie": label(r.data),
             "Tags": " · ".join(entry_tags(r)),
             "Extrait": preview(r.data), "Ligne": r.source_line} for r in shown]
    df = pd.DataFrame(data)
    page_ids = tuple(row.uid for row in shown)
    selection_key = f"{prefix}_grid_{st.session_state.revision}_{page}_{page_ids}"
    editor = st.data_editor(
        df, key=selection_key, hide_index=True, width="stretch", height=min(650, 60 + 36 * len(shown)),
        disabled=["ID", "Langue", "Messages", "Catégorie", "Tags", "Extrait", "Ligne"],
        column_config={"Cocher": st.column_config.CheckboxColumn("Sélectionner", width="small"),
                       "ID": st.column_config.NumberColumn("ID", width="small"),
                       "Langue": st.column_config.TextColumn("Langue", width="small"),
                       "Messages": st.column_config.NumberColumn("Tours", width="small"),
                       "Catégorie": st.column_config.TextColumn("Type", width="medium"),
                       "Tags": st.column_config.TextColumn("Tags", width="medium"),
                       "Extrait": st.column_config.TextColumn("Conversation", width="large"),
                       "Ligne": st.column_config.NumberColumn("Ligne", width="small")})
    return set(int(uid) for uid in editor.loc[editor["Cocher"] == True, "ID"].tolist())


def save_conversation(entry: Entry, data: dict):
    if data == entry.data:
        st.info("Aucune modification à enregistrer.")
        return
    tags = entry.auto_tags
    if entry.auto_tags or st.session_state.get("auto_tags_import", True):
        try:
            with st.spinner("Actualisation des tags..."):
                tags = analyze_tags(data)
            st.session_state.tagging_errors = [(uid, message) for uid, message in
                st.session_state.get("tagging_errors", []) if uid != entry.uid]
        except QwenError as exc:
            st.session_state.tagging_errors = [(entry.uid, str(exc))]
    mutation({"kind": "edit", "id": entry.uid, "before": entry.data,
              "before_auto_tags": entry.auto_tags})
    entry.data = data
    entry.auto_tags = tags
    st.session_state.editor_notice = entry.uid
    st.rerun()


def inspector(rows: list[Entry], *, prefix: str):
    if not rows:
        return
    st.subheader("Inspecter / modifier une conversation")
    # Inspect the current visible page; avoid sending huge dropdowns to the browser.
    page_no = st.session_state.get(f"{prefix}_page", 1)
    candidates = rows[(page_no - 1) * 50:page_no * 50] or rows[:50]
    choices = {r.uid: r for r in candidates}
    if len(rows) > len(candidates):
        st.caption("Inspecteur : entrées de la page affichée. Change de page pour éditer les autres.")
    uid = st.selectbox("Entrée", options=list(choices.keys()),
                       format_func=lambda x: f"#{x} · {preview(choices[x].data, 95)}",
                       key=f"{prefix}_inspector")
    entry = choices[uid]
    if st.session_state.pop("editor_notice", None) == uid:
        st.success("Conversation enregistrée en mémoire. Exporte pour conserver les modifications.")
    fields = conversation_fields(entry.data)
    st.caption(f"Entrée #{uid} · ligne {entry.source_line} · "
               f"{sum(len(field.text) for field in fields):,} caractères".replace(",", " "))
    if uid in st.session_state.get("entry_sources", {}):
        st.caption(f"Fichier source : {st.session_state.entry_sources[uid]}")
    mode = st.radio("Affichage de l'entrée", ["📖 Lecture", "✏️ Modifier les messages", "{ } JSON avancé"],
                    horizontal=True, key=f"{prefix}_view")
    if mode == "📖 Lecture":
        if fields:
            avatars = {"system": "⚙️", "user": "👤", "assistant": "🎭"}
            for field in fields:
                with st.chat_message(field.role or "texte", avatar=avatars.get(field.role, "💬")):
                    st.caption(field.label)
                    # Escape dataset content: display literal text, including markup and URLs.
                    st.markdown(f'<div class="conversation-body" dir="auto">{escape(field.text)}</div>',
                                unsafe_allow_html=True)
        else:
            st.info("Aucun texte reconnu dans cette entrée. Utilise « JSON avancé » pour la consulter.")
    elif mode == "✏️ Modifier les messages":
        if fields:
            st.caption("Modifie les textes, puis enregistre la conversation avant de changer d'entrée ou de vue.")
            with st.form(key=f"{prefix}_messages_{uid}_{st.session_state.revision}"):
                save_at_top = len(fields) > 8 and st.form_submit_button(
                    "Enregistrer la conversation", type="primary",
                    key=f"{prefix}_save_top_{uid}_{st.session_state.revision}")
                edits = {}
                for index, field in enumerate(fields):
                    height = min(420, max(130, 70 + 22 * (field.text.count("\n") + len(field.text) // 90 + 1)))
                    edits[field.path] = st.text_area(
                        field.label, value=field.text, height=height,
                        key=f"{prefix}_message_{uid}_{st.session_state.revision}_{index}")
                save_at_bottom = st.form_submit_button(
                    "Enregistrer la conversation", type="primary",
                    key=f"{prefix}_save_bottom_{uid}_{st.session_state.revision}")
                if save_at_top or save_at_bottom:
                    save_conversation(entry, update_conversation_text(entry.data, edits))
        else:
            st.info("Ce format se modifie dans « JSON avancé ».")
    else:
        st.caption("Modification du JSON original complet, y compris les rôles et les métadonnées.")
        with st.form(key=f"{prefix}_edit_{uid}_{st.session_state.revision}"):
            edited = st.text_area("Contenu JSON", value=json.dumps(entry.data, ensure_ascii=False, indent=2),
                                  height=550, key=f"{prefix}_edit_area_{uid}_{st.session_state.revision}")
            submitted = st.form_submit_button("Enregistrer les modifications", type="primary")
            if submitted:
                try:
                    obj = json.loads(edited)
                    if not isinstance(obj, dict):
                        raise ValueError("Le JSON doit être un objet { ... }")
                except (ValueError, json.JSONDecodeError) as exc:
                    st.error(f"JSON non valide : {exc}")
                else:
                    save_conversation(entry, obj)
    st.caption("Les images, appels d'outils et autres champs restent accessibles dans « JSON avancé ».")
    render_tag_editor(entry, mutation, prefix=prefix)
    with st.expander("Langue et aperçu de l'export Unsloth"):
        st.caption("Aperçu normalisé pour Unsloth")
        messages = as_messages(entry.data)
        if messages:
            st.json({"messages": messages}, expanded=False)
        else:
            st.warning("Ce format n'est pas directement convertible en messages ChatML.")
            st.json(entry.data, expanded=False)
        lang_choice = st.selectbox("Corriger la langue de cette entrée", options=list(LANG_CODES),
                                   index=list(LANG_CODES).index(effective_language(entry)),
                                   format_func=lambda code: f"{code} — {LANG_CODES[code]}",
                                   key=f"lang_{uid}_{st.session_state.revision}")
        if st.button("Appliquer la langue", key=f"set_lang_{uid}_{st.session_state.revision}"):
            mutation({"kind": "lang", "before": [(uid, entry.language_override)]})
            entry.language_override = lang_choice
            st.rerun()


initialize()
if "navigate_after_action" in st.session_state:
    st.session_state.navigation_page = st.session_state.pop("navigate_after_action")
rows = st.session_state.rows

with st.sidebar:
    st.markdown("# 🎭 RP Dataset Studio")
    st.caption("Gestion locale de datasets JDR · v0.09b")
    render_qwen_settings()
    st.divider()
    st.markdown("**Ouvrir un dataset**")
    st.checkbox("Réparer les erreurs JSONL simples", value=True, key="repair_jsonl_import",
                help="Sépare les objets collés et retire les virgules superflues. "
                     "Le fichier d'origine reste intact. Réimporte le fichier pour appliquer ce choix.")
    st.checkbox("Ajouter les tags automatiquement à l'import", value=True, key="auto_tags_import",
                help="SFW/NSFW, genres, thèmes et traits avec le moteur de tags choisi. "
                     "Les projets .rpwork.json retrouvent leurs annotations sans nouveau classement.")
    upload = st.file_uploader("Importer un fichier", type=["jsonl", "json", "csv", "tsv", "parquet"],
                              help="JSONL, JSON, CSV, TSV, Parquet, projets .rpwork.json")
    if st.button("Importer le fichier sélectionné", disabled=upload is None, use_container_width=True):
        import_file(io.BytesIO(upload.getvalue()), upload.name)
    st.caption("Pour les gros fichiers, ouvre directement un chemin local :")
    local_path = st.text_input("Chemin du fichier sur ce PC", placeholder=r"C:\\Datasets\\rp_train.jsonl")
    if st.button("Ouvrir le chemin local", disabled=not local_path.strip(), use_container_width=True):
        path = Path(local_path.strip().strip('"'))
        if not path.is_file():
            st.error("Ce fichier n'existe pas.")
        else:
            with path.open("rb") as handle:
                import_file(handle, path.name)
    if st.button("Charger le dataset de démonstration", use_container_width=True):
        sample = BASE / "examples" / "exemple_rp.jsonl"
        with sample.open("rb") as handle:
            import_file(handle, sample.name)
    st.divider()
    if st.session_state.dataset_name:
        st.caption(f"Dataset : {st.session_state.dataset_name}")
        if st.button("↶ Annuler la dernière action", disabled=not st.session_state.history,
                     use_container_width=True):
            undo()
            st.rerun()
    page = st.radio("Navigation", ["🔎 Explorer", "📖 Depuis un livre", FINETUNING_PAGE, "🏷️ Tags", "🧩 Fusionner", "🧹 Qualité", "📦 Exporter", "🗑️ Corbeille", "❔ Aide"],
                    key="navigation_page", label_visibility="collapsed")
    st.divider()
    st.caption("🔒 Fichiers traités localement. Aucune donnée envoyée à un service externe."
               " Le navigateur communique uniquement avec l'application sur ce PC.")

if page == "🧩 Fusionner":
    render_merge_page(loaded)
    st.stop()

if page == "📖 Depuis un livre":
    render_book_page(loaded)
    st.stop()

if page == FINETUNING_PAGE:
    render_finetuning_page(rows, st.session_state.dataset_name, st.session_state.get("entry_sources"))
    st.stop()

if not st.session_state.dataset_name:
    st.title("Éditeur local de datasets RP")
    st.write("Parcours tes conversations, filtre par langue ou mots-clés, supprime des lots, édite les"
             " entrées, ajoute des tags et exporte un fichier JSONL prêt pour Unsloth.")
    a, b, c = st.columns(3)
    a.info("**1. Importe**\n\nJSONL · JSON · CSV · TSV · Parquet")
    b.info("**2. Nettoie**\n\nLangues · mots-clés · doublons · édition")
    c.info("**3. Exporte**\n\nOriginal filtré · ChatML · projet")
    st.subheader("Commencer")
    st.write("Utilise la colonne de gauche pour charger un dataset, ou clique sur « Charger le dataset de démonstration ». "
             "Pour rassembler plusieurs fichiers, ouvre « 🧩 Fusionner ». "
             "Pour créer des exemples RP et narration depuis un TXT, ouvre « 📖 Depuis un livre ».")
    st.stop()

live = [r for r in rows if not r.deleted]
removed = [r for r in rows if r.deleted]
convertible = sum(as_messages(r.data) is not None for r in live)
unknown = sum(effective_language(r) == "?" for r in live)
st.title("🎭 RP Dataset Studio")
st.caption(f"{st.session_state.dataset_name} · travail en mémoire, fichier source préservé")
a, b, c, d = st.columns(4)
a.metric("Conversations actives", f"{len(live):,}".replace(",", " "))
b.metric("Dans la corbeille", f"{len(removed):,}".replace(",", " "))
c.metric("Convertibles ChatML", f"{convertible:,}".replace(",", " "))
d.metric("Langue inconnue", f"{unknown:,}".replace(",", " "))

if st.session_state.repairs:
    st.success(f"{len(st.session_state.repairs)} ligne(s) JSONL réparée(s) automatiquement. "
               "Exporte le dataset pour enregistrer une copie corrigée.")
    with st.expander("Détail des réparations à l'import", expanded=False):
        for repair in st.session_state.repairs[:30]:
            st.write(repair)
        if len(st.session_state.repairs) > 30:
            st.caption("Les 30 premières réparations sont affichées.")

if st.session_state.errors:
    st.warning("Certaines entrées n'ont pas pu être importées. Consulte les erreurs ci-dessous.")
    with st.expander(f"⚠️ {len(st.session_state.errors)} erreur(s) d'import affichables", expanded=False):
        for issue in st.session_state.errors:
            st.write(issue)

report_errors(st.session_state.get("tagging_errors", []))

if page == "🔎 Explorer":
    st.subheader("Rechercher et filtrer")
    render_filters(live)
    ff = st.session_state.filters
    try:
        filtered = apply_filters(rows, contains=ff["contains"], excludes=ff["excludes"],
                                 languages=ff["languages"], use_regex=ff["regex"],
                                 only_unconvertible=ff["unconvertible"], tags=ff.get("tags"),
                                 excluded_tags=ff.get("excluded_tags"),
                                 tag_mode=ff.get("tag_mode", "all"), tag_status=ff.get("tag_status", "all"))
    except re.error as exc:
        st.error(f"Expression régulière invalide : {exc}")
        filtered = []
    st.write(f"**{len(filtered):,} résultat(s)** sur {len(live):,} actifs".replace(",", " "))
    render_filtered_export(filtered)
    chosen = row_table(filtered, prefix="explorer")
    st.markdown("#### Actions sur les résultats")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.4, 1.4, 2])
        with c1:
            if st.button(f"🗑️ Supprimer {len(chosen)} cochée(s)", disabled=not chosen,
                         use_container_width=True):
                delete_some(chosen)
                st.rerun()
        with c2:
            lang_selected = st.selectbox("Langue des entrées cochées", options=list(LANG_CODES),
                                         format_func=lambda x: f"{x} — {LANG_CODES[x]}")
            if st.button("Attribuer cette langue", disabled=not chosen,
                         use_container_width=True):
                targets = [r for r in live if r.uid in chosen]
                before = [(r.uid, r.language_override) for r in targets]
                for r in targets:
                    r.language_override = lang_selected
                mutation({"kind": "lang", "before": before})
                st.rerun()
        with c3:
            st.warning("La suppression par lot concerne **tous les résultats filtrés**,"
                       " pas seulement la page visible. Elle est annulable.", icon="⚠️")
            confirm = st.checkbox(f"Confirmer la suppression des {len(filtered)} résultats",
                                  key=f"confirm_{st.session_state.revision}")
            if st.button("Supprimer TOUS les résultats filtrés", type="primary",
                         disabled=not filtered or not confirm, use_container_width=True):
                delete_some({r.uid for r in filtered})
                st.rerun()
    st.divider()
    inspector(filtered, prefix="explorer")

elif page == "🏷️ Tags":
    render_tag_browser(rows, mutation)

elif page == "🧹 Qualité":
    st.subheader("Contrôle qualité")
    render_tag_panel(rows, mutation)
    st.divider()
    count = counts_by_language(rows)
    if count:
        chart = pd.DataFrame([{"Langue": f"{code} · {LANG_CODES.get(code, code)}", "Entrées": n}
                              for code, n in sorted(count.items(), key=lambda it: -it[1])])
        st.bar_chart(chart, x="Langue", y="Entrées", horizontal=True, height=min(400, 55 * len(count) + 80))
    st.markdown("#### Détection des langues")
    if "language_detection_summary" in st.session_state:
        st.success(st.session_state.pop("language_detection_summary"))
    st.write("Les colonnes `language`, `lang` ou `locale` sont réutilisées quand elles existent. "
             "Sinon, le détecteur analyse surtout les réponses de l'assistant. "
             "Pour les dialogues très courts, le résultat peut rester inconnu.")
    if st.button(f"🌍 Détecter la langue des {unknown:,} entrées inconnues".replace(",", " "),
                 disabled=unknown == 0):
        if language_detection():
            st.rerun()
    st.divider()
    render_conversion_panel(rows, mutation)
    st.divider()
    st.markdown("#### Doublons exacts / quasi-identiques (espaces et casse ignorés)")
    duplicates = duplicate_ids(rows)
    st.metric("Doublons", len(duplicates))
    st.write("Conserve la première occurrence active et identifie les suivantes."
             " Ce contrôle n'est pas une comparaison sémantique.")
    if duplicates:
        duplicate_set = set(duplicates)
        displayed = [row for row in live if row.uid in duplicate_set][:100]
        st.dataframe(pd.DataFrame([{"ID": r.uid, "Langue": effective_language(r),
                                   "Conversation": preview(r.data)}
                                  for r in displayed]),
                     hide_index=True, width="stretch")
        if st.button(f"🗑️ Placer les {len(duplicates)} doublons dans la corbeille"):
            delete_some(duplicate_set)
            st.rerun()
    st.divider()
    st.markdown("#### Formats de données")
    st.write(f"**{convertible:,}** entrées convertibles / **{len(live) - convertible:,}** "
             "non convertibles en `messages`.".replace(",", " "))
    if len(live) - convertible:
        st.info("Pour afficher ces entrées, coche « Seulement les formats non convertibles »"
                " dans l'onglet Explorer. Elles ne seront pas incluses dans l'export ChatML.")

elif page == "📦 Exporter":
    st.subheader("Exporter ton dataset")
    if "merge_report" in st.session_state:
        report = st.session_state.merge_report
        if "export_chatml" in st.session_state:
            st.success(f"Fusion terminée : {report['conversations_exportees']} conversations prêtes pour Unsloth. "
                       "Le JSONL est disponible ci-dessous.")
        with st.expander("Bilan de la fusion"):
            st.dataframe(pd.DataFrame(report["fichiers"]), hide_index=True, width="stretch")
            st.write(f"Doublons écartés : {report['doublons_ecartes']} · "
                     f"Entrées incompatibles écartées : {report['incompatibles_ecartees']} · "
                     f"Entrées converties : {report.get('conversions_automatiques', 0)}")
            st.download_button("Télécharger le bilan de fusion", json.dumps(report, ensure_ascii=False, indent=2),
                               file_name="fusion_bilan.json", mime="application/json")
    st.write("**Aucun export ne modifie le dataset d'origine.** Les entrées de la corbeille"
             " sont exclues des fichiers d'entraînement.")
    base = Path(st.session_state.dataset_name).stem.replace(".rpwork", "") or "dataset"
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("#### 1. JSONL nettoyé (structure d'origine)")
            st.write("Conserve les champs originaux, à l'exception des entrées supprimées."
                     " Les corrections de langue sont ajoutées au champ `language`, les tags au champ `tags`.")
            if st.button("Préparer le JSONL d'origine", key="prepare_original"):
                blob, amount = export_jsonl(rows)
                st.session_state.export_original = (blob, amount)
            if "export_original" in st.session_state:
                blob, amount = st.session_state.export_original
                st.download_button(f"⬇️ Télécharger {amount} entrées", blob,
                                   file_name=f"{base}_nettoye.jsonl", mime="application/x-ndjson")
    with c2:
        with st.container(border=True):
            st.markdown("#### 2. JSONL Unsloth / ChatML")
            st.write("Convertit les formats compatibles en objets `messages` (system/user/assistant)."
                     " Les entrées incompatibles sont ignorées.")
            if st.button("Préparer le JSONL ChatML", key="prepare_chatml"):
                blob, amount = export_jsonl(rows, chatml=True)
                st.session_state.export_chatml = (blob, amount)
            if "export_chatml" in st.session_state:
                blob, amount = st.session_state.export_chatml
                st.download_button(f"⬇️ Télécharger {amount} conversations", blob,
                                   file_name=f"{base}_unsloth.jsonl", mime="application/x-ndjson")
    st.divider()
    with st.container(border=True):
        st.markdown("#### 3. Projet modifiable (.rpwork.json)")
        st.write("Sauvegarde aussi la corbeille, les langues, les tags automatiques et manuels, et les fichiers sources."
                 " Réimporte ce fichier dans l'application pour poursuivre plus tard.")
        if st.button("Préparer la sauvegarde du projet"):
            st.session_state.export_project_data = export_project(
                rows, st.session_state.dataset_name, st.session_state.get("entry_sources", {}))
        if "export_project_data" in st.session_state:
            st.download_button("💾 Télécharger la sauvegarde complète",
                               st.session_state.export_project_data,
                               file_name=f"{base}.rpwork.json", mime="application/json")
    st.divider()
    with st.container(border=True):
        st.markdown("#### 4. Découper en plusieurs fichiers")
        st.write("Crée des fichiers JSONL numérotés, regroupés dans un ZIP. "
                 "L'ordre des conversations est conservé et chaque entrée reste entière.")
        size_col, format_col = st.columns(2)
        with size_col:
            part_size = int(st.number_input("Entrées par fichier", min_value=1, value=2000,
                                            step=500, key="split_size"))
        with format_col:
            split_format = st.radio("Format des fichiers découpés",
                                    ["Structure d'origine", "Unsloth / ChatML"],
                                    horizontal=True, key="split_format")
        split_chatml = split_format == "Unsloth / ChatML"
        split_total = convertible if split_chatml else len(live)
        part_count = math.ceil(split_total / part_size)
        if split_total:
            last_size = split_total - (part_count - 1) * part_size
            total_label = f"{split_total:,}".replace(",", " ")
            size_label = f"{part_size:,}".replace(",", " ")
            last_label = f"{last_size:,}".replace(",", " ")
            st.info(f"{total_label} entrée{'s' if split_total > 1 else ''} → "
                    f"{part_count} fichier{'s' if part_count > 1 else ''}, "
                    f"jusqu'à {size_label} entrées par fichier. "
                    f"Dernier fichier : {last_label} entrée{'s' if last_size > 1 else ''}.")
        else:
            st.info("Aucune entrée à exporter dans ce format.")
        if split_chatml and len(live) > convertible:
            st.caption(f"{len(live) - convertible} entrée(s) incompatible(s) avec ChatML seront exclues.")
        settings = (part_size, split_chatml)
        prepared = st.session_state.get("export_parts")
        if prepared and prepared[2] != settings:
            st.session_state.pop("export_parts", None)
        if st.button("Préparer les fichiers découpés", disabled=split_total == 0):
            with st.spinner("Création des fichiers et compression du ZIP..."):
                blob, parts = export_split_zip(rows, base, entries_per_file=part_size, chatml=split_chatml)
            st.session_state.export_parts = (blob, parts, settings)
        if "export_parts" in st.session_state:
            blob, parts, _ = st.session_state.export_parts
            format_name = "unsloth" if split_chatml else "nettoye"
            st.download_button(f"⬇️ Télécharger le ZIP ({len(parts)} fichier{'s' if len(parts) > 1 else ''})", blob,
                               file_name=f"{base}_{format_name}_par_{part_size}.zip", mime="application/zip")
            with st.expander("Contenu du ZIP"):
                st.dataframe(pd.DataFrame(parts, columns=["Fichier", "Entrées"]),
                             hide_index=True, width="stretch")
    st.caption("Après toute modification, clique de nouveau sur « Préparer » pour régénérer les fichiers." )

elif page == "🗑️ Corbeille":
    st.subheader("Corbeille")
    st.write("Les entrées de la corbeille sont exclues des exports JSONL, mais conservées dans le projet. "
             "Tu peux les restaurer ou vider définitivement la corbeille.")
    if "purge_notice" in st.session_state:
        st.success(st.session_state.pop("purge_notice"))
    selected = row_table(removed, prefix="trash")
    a, b = st.columns(2)
    if a.button("Restaurer les entrées cochées", disabled=not selected):
        changed = restore_ids(rows, selected)
        if changed:
            mutation({"kind": "restore", "ids": changed})
        st.rerun()
    if b.button("Restaurer TOUTE la corbeille", disabled=not removed):
        changed = restore_ids(rows, {r.uid for r in removed})
        if changed:
            mutation({"kind": "restore", "ids": changed})
        st.rerun()
    with st.expander("Vider définitivement la corbeille"):
        st.warning(f"Retire les {len(removed)} entrées de la corbeille du projet et efface tout "
                   "l'historique d'annulation. Cette action est irréversible dans l'application. "
                   "Les fichiers déjà présents sur le disque restent inchangés.")
        confirm_purge = st.checkbox(f"Confirmer la purge définitive des {len(removed)} entrées",
                                   key=f"confirm_purge_{st.session_state.revision}", disabled=not removed)
        if st.button("Vider définitivement la corbeille", disabled=not removed or not confirm_purge):
            empty_trash()
            st.rerun()
    inspector(removed, prefix="trash")

else:
    st.subheader("Aide rapide")
    st.markdown("""
**Import** : `JSONL` (un objet par ligne), `JSON` (liste d'objets), `CSV`, `TSV` ou `Parquet`.
Les conversations sous `messages`, `conversations` (ShareGPT), `instruction/output`
(Alpaca) et `prompt/completion` sont reconnues.

**Fusion** : ouvre « 🧩 Fusionner », sélectionne plusieurs fichiers ou indique leurs chemins,
puis analyse le bilan. La fusion peut écarter les doublons et les conversions incomplètes,
et mélanger les conversations. Le JSONL Unsloth est préparé dans Exporter ; les entrées
écartées restent dans la corbeille. Sauvegarde le projet pour les conserver.

**Livre TXT** : ouvre « 📖 Depuis un livre », sélectionne ton fichier et les exemples RP et/ou
narration, puis prépare le livre. Génère un aperçu avant de lancer tout le livre. Les réponses
d'entraînement conservent le texte original ; Qwen sélectionne les dialogues RP et rédige les
consignes de narration. Vérifie les rôles et les consignes. La progression est sauvegardée dans
`data/book_generation/` et retrouvée avec le même livre et les mêmes réglages. Ouvre les exemples
dans l'éditeur pour les corriger, puis sauvegarde un projet : les corrections de l'éditeur ne
modifient pas la progression du générateur. Le TXT reste intact et l'export Unsloth contient
seulement les messages. Qwen 2B ou 4B est nécessaire ; les règles sans LLM ne génèrent pas d'exemples.

**Filtre** : `romance ; bataille` conserve les entrées contenant *au moins un*
de ces termes. Le filtre « exclure » rejette toute entrée contenant un terme interdit.
Les expressions régulières sont facultatives, sans distinction majuscule/minuscule.

**Tags** : ajoutés automatiquement à l'import (désactivable) et pendant la fusion. Dans Qualité,
« Détecter / actualiser les tags » analyse les entrées actives. Dans Explorer, choisis tous les
tags requis (ET) ou au moins un (OU), puis exclus les tags indésirables. Les raccourcis SFW,
NSFW, Needs Review et Sans tags remplacent les critères de tags. Le catalogue « 🏷️ Tags » permet
de rechercher par nom ou famille, voir les fréquences, ouvrir un filtre et renommer/fusionner/retirer
un tag par lot après aperçu et confirmation. Les tags correspondent à des noms exacts, sans
distinction de casse. « Exporter uniquement les résultats filtrés » conserve le sous-ensemble
sur toutes les pages en JSONL original avec tags ou ChatML. Les exports de la page Exporter
continuent à concerner toutes les entrées actives. L'inspecteur permet de corriger les tags
ou de revenir aux suggestions automatiques. Les corrections sont annulables et conservées
dans les projets ; le JSONL d'origine reçoit un champ `tags`. ChatML conserve seulement les messages.
Dans la barre latérale → Analyse des tags, choisis Qwen 2B Heretic (GGUF), Qwen 4B Heretic
(le modèle FP8 fourni), ou les règles sans LLM. Qwen analyse les genres implicites et les thèmes
NSFW ; ses suggestions peuvent contenir des erreurs. Needs Review signale une incertitude.
Un échec conserve les annotations précédentes et affiche une erreur, sans basculer sur les règles.
Les longs textes sont analysés en segments avec recouvrement. Le modèle reste en mémoire
jusqu'au déchargement ou à la fermeture de l'application. Les noms et œuvres doivent être explicites.
Pour reprendre une analyse sans tout recalculer : Qualité → « Analyser uniquement les entrées sans tags ».
Seules les entrées actives sans tag sont analysées ; les tags existants et les listes vidées manuellement
sont conservés. Le bouton se désactive quand il ne reste aucune entrée éligible.

**Langues** : utilise les métadonnées de langue existantes ou l'onglet Qualité pour détecter
le français, l'anglais, l'espagnol et d'autres langues localement. Une détection automatique
reste indicative ; vérifie les cas ambigus.

**Conversion des incompatibles** : dans Qualité, analyse les entrées incompatibles et applique
les conversions récupérables. Coche « Inclure les entrées de la corbeille » pour récupérer
des conversations écartées pendant une fusion. Les conversions sont annulables ; exporte
ensuite le JSONL Unsloth. La fusion peut aussi appliquer ces conversions automatiquement.

**Suppression** : les cases sélectionnent uniquement les entrées de la page affichée.
« Supprimer tous les résultats filtrés » applique le filtre à l'ensemble du fichier.
Les suppressions sont réversibles via la corbeille ou le bouton Annuler.
Dans Corbeille, « Vider définitivement la corbeille » retire ses entrées du projet après confirmation
et efface tout l'historique d'annulation. Sauvegarde ensuite un nouveau projet pour conserver la purge.
Les anciens fichiers sur disque ne sont pas modifiés.

**Édition** : dans Explorer, choisis une entrée. « Lecture » affiche la conversation en entier,
message par message. « Modifier les messages » permet d'éditer les textes sans écrire du JSON.
Enregistre avant de changer d'entrée ou de vue. Les rôles et les autres champs sont conservés.
« JSON avancé » donne accès à l'objet complet et refuse un JSON invalide.

**Sauvegarde** : télécharge un fichier `.rpwork.json` pour conserver le travail en cours,
ou un `.jsonl` nettoyé à utiliser pour l'entraînement. Aucune sauvegarde automatique.
Dans Exporter, « Découper en plusieurs fichiers » produit un ZIP de fichiers JSONL,
avec 2 000 entrées par fichier par défaut. Le dernier contient les entrées restantes.

**Sécurité et confidentialité** : l'application fonctionne sur `127.0.0.1` et ne transmet
pas les conversations au cloud. Les bibliothèques sont installées au premier lancement.
    """)
