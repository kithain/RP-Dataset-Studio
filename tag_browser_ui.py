"""Searchable tag catalogue, filter controls and filtered JSONL exports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core import LANG_CODES, as_messages, effective_language, export_jsonl
from tagging import entry_tags
from tag_management import (apply_tag_change, default_filters, plan_tag_change,
                            tag_catalogue)


def set_filters(filters):
    st.session_state.filters = filters
    st.session_state.filter_revision = st.session_state.get("filter_revision", 0) + 1
    st.session_state.explorer_page = 1
    st.session_state.pop("export_filtered", None)


def render_filters(live):
    current = {**default_filters(), **st.session_state.filters}
    catalogue = tag_catalogue(live)
    counts = {item["Tag"].casefold(): item["Entrées"] for item in catalogue}
    names = {item["Tag"].casefold(): item["Tag"] for item in catalogue}
    # Keep absent selected tags visible (0 entries), rather than silently broadening a filter.
    for tag in current["tags"] + current["excluded_tags"]:
        names.setdefault(tag.casefold(), tag)
    options = sorted(names.values(), key=str.casefold)
    format_tag = lambda tag: f"{tag} · {counts.get(tag.casefold(), 0)}"
    st.caption("Tags exacts, casse ignorée. Les nombres indiquent les entrées du dataset actif. "
               "Les raccourcis remplacent les critères de tags et conservent les autres critères.")
    buttons = st.columns(5)
    for col, title, tag, status in zip(buttons[:4], ["SFW", "NSFW", "Needs Review", "Sans tags"],
                                      ["SFW", "NSFW", "Needs Review", None],
                                      ["all", "all", "all", "untagged"]):
        if col.button(title, key=f"quick_tag_{title}", use_container_width=True):
            set_filters({**current, "tags": [tag] if tag else [], "excluded_tags": [],
                         "tag_mode": "all", "tag_status": status})
            st.rerun()
    if buttons[4].button("Réinitialiser les filtres", use_container_width=True):
        set_filters(default_filters())
        st.rerun()
    # A fresh form after programmatic changes prevents stale widget values overriding them.
    with st.form(f"search_filters_{st.session_state.get('filter_revision', 0)}"):
        col1, col2 = st.columns(2)
        with col1:
            tags = st.multiselect("Tags à conserver", options=options,
                                  default=[names[tag.casefold()] for tag in current["tags"]],
                                  format_func=format_tag)
            mode = st.radio("Combiner les tags à conserver", ["all", "any"],
                            index=["all", "any"].index(current["tag_mode"]), horizontal=True,
                            format_func=lambda value: {"all": "Tous (ET)", "any": "Au moins un (OU)"}[value])
            contains = st.text_input("Mots à conserver (séparés par ; = OU)", value=current["contains"],
                                     placeholder="nécromancien ; romance ; bataille")
            available_langs = sorted({effective_language(row) for row in live} | set(current["languages"]),
                                     key=lambda value: (value == "?", value))
            languages = st.multiselect("Langues", options=available_langs,
                                       default=current["languages"],
                                       format_func=lambda value: f"{value} — {LANG_CODES.get(value, value)}")
        with col2:
            excluded = st.multiselect("Tags à exclure (au moins un)", options=options,
                                      default=[names[tag.casefold()] for tag in current["excluded_tags"]],
                                      format_func=format_tag)
            status = st.selectbox("Présence de tags", ["all", "tagged", "untagged"],
                                  index=["all", "tagged", "untagged"].index(current["tag_status"]),
                                  format_func=lambda value: {"all": "Toutes les entrées",
                                      "tagged": "Avec des tags", "untagged": "Sans tags"}[value])
            excludes = st.text_input("Mots à exclure (séparés par ;)", value=current["excludes"],
                                     placeholder="advertising ; spam")
            regex = st.checkbox("Utiliser des expressions régulières", value=current["regex"])
            unconvertible = st.checkbox("Seulement les formats non convertibles", value=current["unconvertible"])
        if st.form_submit_button("Appliquer les filtres", type="primary"):
            set_filters({"tags": tags, "excluded_tags": excluded, "tag_mode": mode, "tag_status": status,
                         "contains": contains, "excludes": excludes, "languages": languages,
                         "regex": regex, "unconvertible": unconvertible})
            st.rerun()
    summary = []
    if current["tags"]:
        summary.append(("Tous : " if current["tag_mode"] == "all" else "Au moins un : ") + " · ".join(current["tags"]))
    if current["excluded_tags"]:
        summary.append("Exclus : " + " · ".join(current["excluded_tags"]))
    if current["tag_status"] != "all":
        summary.append("Avec des tags" if current["tag_status"] == "tagged" else "Sans tags")
    if current["contains"]:
        summary.append("Texte inclus : " + current["contains"])
    if current["excludes"]:
        summary.append("Texte exclu : " + current["excludes"])
    if current["languages"]:
        summary.append("Langues : " + ", ".join(current["languages"]))
    if current["regex"]:
        summary.append("Expressions régulières")
    if current["unconvertible"]:
        summary.append("Non convertibles")
    st.caption("Filtres appliqués — " + (" | ".join(summary) if summary else "aucun"))
    if set(map(str.casefold, current["tags"])) & set(map(str.casefold, current["excluded_tags"])):
        st.warning("Un tag est à la fois demandé et exclu. Vérifie les critères choisis.")


def render_filtered_export(filtered):
    with st.expander("📦 Exporter uniquement les résultats filtrés"):
        st.caption("Tous les résultats, sur toutes les pages. Le dataset ouvert est conservé.")
        chatml = st.radio("Format de l'export filtré", ["Structure d'origine", "Unsloth / ChatML"],
                          horizontal=True, key="filtered_export_format") == "Unsloth / ChatML"
        count = sum(as_messages(row.data) is not None for row in filtered) if chatml else len(filtered)
        st.write(f"{count} entrée(s) à exporter sur {len(filtered)} résultat(s).")
        if chatml and count < len(filtered):
            st.caption(f"{len(filtered) - count} entrée(s) incompatible(s) avec ChatML exclue(s).")
        signature = (st.session_state.revision, json.dumps(st.session_state.filters, sort_keys=True), chatml)
        if st.button("Préparer l'export filtré", disabled=count == 0):
            blob, amount = export_jsonl(filtered, chatml=chatml)
            st.session_state.export_filtered = (signature, blob, amount)
        prepared = st.session_state.get("export_filtered")
        if prepared and prepared[0] == signature:
            _, blob, amount = prepared
            base = Path(st.session_state.dataset_name).stem.replace(".rpwork", "") or "dataset"
            suffix = "unsloth" if chatml else "original"
            st.download_button(f"⬇️ Télécharger {amount} résultats filtrés", blob,
                               file_name=f"{base}_filtre_{suffix}.jsonl", mime="application/x-ndjson")


def render_tag_browser(rows, mutation):
    st.subheader("🏷️ Catalogue des tags")
    st.write("Retrouve un tag, consulte sa fréquence et ouvre les conversations correspondantes dans Explorer.")
    live = [row for row in rows if not row.deleted]
    catalogue = tag_catalogue(live)
    tagged = sum(bool(entry_tags(row)) for row in live)
    st.caption(f"{len(catalogue)} tags distincts · {tagged} entrées avec tags · {len(live) - tagged} sans tags. "
               "La corbeille est exclue. Les variantes de casse sont regroupées.")
    if "tag_manager_notice" in st.session_state:
        st.success(st.session_state.pop("tag_manager_notice"))
    col1, col2, col3 = st.columns([2, 2, 1])
    query = col1.text_input("Rechercher un tag", placeholder="Romance, Vore, nom de personnage…")
    family = col2.selectbox("Famille de tags", ["Toutes"] + sorted({item["Famille"] for item in catalogue}))
    sort = col3.selectbox("Trier les tags", ["Fréquence", "Alphabétique"])
    visible = [item for item in catalogue if query.casefold().strip() in item["Tag"].casefold()
               and (family == "Toutes" or item["Famille"] == family)]
    if sort == "Alphabétique":
        visible.sort(key=lambda item: item["Tag"].casefold())
    st.caption(f"{len(visible)} tag(s) correspondent à la recherche.")
    if not visible:
        st.info("Aucun tag correspondant. Tu peux annoter les entrées dans Qualité ou corriger une entrée dans Explorer.")
        return
    st.dataframe(pd.DataFrame(visible), hide_index=True, width="stretch", height=350)
    counts = {item["Tag"]: item["Entrées"] for item in visible}
    selected = st.selectbox("Tag à utiliser", list(counts),
                            format_func=lambda tag: f"{tag} · {counts[tag]} entrée(s)")
    a, b, c = st.columns(3)
    if a.button("Voir ce tag dans Explorer", use_container_width=True):
        set_filters({**default_filters(), "tags": [selected]})
        st.session_state.navigate_after_action = "🔎 Explorer"
        st.rerun()
    for col, title, target, opposite in ((b, "Ajouter au filtre à conserver", "tags", "excluded_tags"),
                                         (c, "Ajouter au filtre à exclure", "excluded_tags", "tags")):
        if col.button(title, use_container_width=True):
            filters = {**default_filters(), **st.session_state.filters}
            filters[target] = [tag for tag in filters[target] if tag.casefold() != selected.casefold()] + [selected]
            filters[opposite] = [tag for tag in filters[opposite] if tag.casefold() != selected.casefold()]
            filters["tag_status"] = "all"
            set_filters(filters)
            st.session_state.navigate_after_action = "🔎 Explorer"
            st.rerun()
    st.caption("« Voir ce tag » réinitialise les autres filtres. « Ajouter au filtre » conserve les critères existants.")
    with st.expander("Renommer, fusionner ou retirer ce tag par lot"):
        st.write(f"Le tag « {selected} » apparaît dans {counts[selected]} entrée(s) actives.")
        action = st.radio("Action sur le tag", ["Renommer / fusionner", "Retirer"], horizontal=True)
        replacement = st.text_input("Nouveau nom du tag", value=selected,
                                    key=f"rename_tag_{selected}_{st.session_state.revision}") if action != "Retirer" else None
        try:
            changes = plan_tag_change(live, selected, replacement)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.write(f"**{len(changes)} entrée(s) seront modifiées.** "
                 "Un nom déjà présent fusionne les tags sans doublon.")
        if changes:
            st.caption("Aperçu des cinq premières corrections :")
            st.dataframe(pd.DataFrame([{"ID": row.uid, "Avant": " · ".join(entry_tags(row)),
                                        "Après": " · ".join(updated) or "Sans tags"}
                                       for row, updated in changes[:5]]),
                         hide_index=True, width="stretch")
        st.caption("Correction manuelle conservée lors des prochaines analyses automatiques. "
                   "Les autres tags restent présents, avec la règle habituelle NSFW prioritaire sur SFW. "
                   "L'action peut être annulée dans la barre latérale.")
        token = hashlib.sha256(json.dumps([selected, replacement, st.session_state.revision]).encode()).hexdigest()[:16]
        confirmed = st.checkbox(f"Confirmer la modification des {len(changes)} entrées", key=f"tag_change_{token}")
        if st.button("Appliquer la modification du tag", disabled=not changes or not confirmed):
            before = apply_tag_change(changes)
            mutation({"kind": "tags", "before": before})
            # Keep existing Explorer criteria explicit; old names remain visible with a zero count.
            st.session_state.filter_revision = st.session_state.get("filter_revision", 0) + 1
            st.session_state.tag_manager_notice = f"{len(changes)} entrée(s) corrigée(s). Action annulable."
            st.rerun()
