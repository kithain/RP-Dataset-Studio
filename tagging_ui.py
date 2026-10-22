"""Tag review and editing controls shared by Explorer and Quality."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from tagging import TAG_RULES, entry_tags, normalize_tags
from tag_management import tag_catalogue
from qwen_ui import analyze_tags, annotate_rows
from llm_tagging import QwenError


TAG_HELP = ("Qwen analyse le sens du texte, les genres implicites, les dynamiques et les thèmes, "
            "y compris NSFW. Le moteur se choisit dans la barre latérale → Analyse des tags. "
            "Les résultats restent des suggestions à vérifier ; Needs Review signale une classification "
            "incertaine. Les règles sans LLM restent disponibles en option explicite.")


def render_tag_editor(entry, mutation, *, prefix):
    with st.expander("🏷️ Tags de cette entrée"):
        tags = entry_tags(entry)
        known = set()
        for group, members in {"Classification": ("SFW", "NSFW", "Needs Review"), **TAG_RULES}.items():
            found = [tag for tag in tags if tag in members]
            known.update(found)
            if found:
                st.caption(group)
                st.text(" · ".join(found))
        other = [tag for tag in tags if tag not in known]
        if other:
            st.caption("Personnage / source / tags libres")
            st.text(" · ".join(other))
        if not tags:
            st.caption("Aucun tag renseigné.")
        st.caption(TAG_HELP)
        if entry.tags_override is not None:
            st.caption("Correction manuelle active : les détections par lot la conservent.")
        with st.form(f"{prefix}_tags_{entry.uid}_{st.session_state.revision}"):
            raw = st.text_input("Corriger les tags (séparés par ;)", value=" ; ".join(tags))
            submitted = st.form_submit_button("Enregistrer les tags")
        if submitted:
            updated = normalize_tags(raw)
            if entry.tags_override != updated:
                mutation({"kind": "tags", "before": [(entry.uid, entry.auto_tags, entry.tags_override)]})
                entry.tags_override = updated
                st.rerun()
        if st.button("Utiliser les tags automatiques", key=f"{prefix}_auto_tags_{entry.uid}"):
            try:
                with st.spinner("Analyse de cette entrée..."):
                    updated = analyze_tags(entry.data)
                st.session_state.tagging_errors = [(uid, message) for uid, message in
                    st.session_state.get("tagging_errors", []) if uid != entry.uid]
                if entry.auto_tags != updated or entry.tags_override is not None:
                    mutation({"kind": "tags", "before": [(entry.uid, entry.auto_tags, entry.tags_override)]})
                    entry.auto_tags = updated
                    entry.tags_override = None
                    st.rerun()
            except QwenError as exc:
                st.error(f"Tags conservés : {exc}")
        st.caption("Les tags sont conservés dans le projet et le JSONL d'origine. "
                   "L'export Unsloth contient uniquement les messages.")


def render_tag_panel(rows, mutation):
    st.markdown("#### 🏷️ Tags automatiques")
    st.write(TAG_HELP)
    st.caption("Genres : Comedy, Romance, Slice of Life, Fantasy, Adventure, Action, Drama, Horror, "
               "Sci-Fi, Mystery. Autres familles : source, archétypes, thèmes, traits et ton du bot.")
    live = [row for row in rows if not row.deleted]
    if "tagging_notice" in st.session_state:
        st.success(st.session_state.pop("tagging_notice"))
    if st.button("Détecter / actualiser les tags", disabled=not live):
        with st.spinner("Analyse locale des conversations et fiches..."):
            before, errors = annotate_rows(live)
        if before:
            originals = {uid: tags for uid, tags in before}
            mutation({"kind": "tags", "before": [(row.uid, originals[row.uid], row.tags_override)
                                                  for row in live if row.uid in originals]})
        st.session_state.tagging_notice = (
            f"Lot de {len(live)} entrées · {len(before)} annotations automatiques actualisées. "
            "Les corrections manuelles sont conservées. L'action est annulable.")
        st.session_state.tagging_errors = errors
        st.rerun()
    catalogue = tag_catalogue(live)
    st.caption(f"{sum(bool(entry_tags(row)) for row in live)} / {len(live)} entrées avec des tags. "
               "Le catalogue 🏷️ Tags permet de rechercher, filtrer et corriger par lot. "
               "Pour une seule entrée : Explorer → Tags de cette entrée.")
    if catalogue:
        st.dataframe(pd.DataFrame(catalogue[:40]),
                     hide_index=True, width="stretch", height=min(350, 36 * len(catalogue) + 40))
