"""Multi-file merge workflow for RP Dataset Studio."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from core import export_jsonl
from merge_datasets import analyze_merge, prepare_merge
from tagging import tag_entries


def render_merge_page(on_load):
    st.title("🧩 Fusionner des datasets pour Unsloth")
    st.write("Rassemble tes fichiers, vérifie le bilan, puis prépare un JSONL au format `messages`. "
             "Les fichiers d'origine restent intacts.")
    uploads = st.file_uploader("Fichiers à fusionner", accept_multiple_files=True,
                               type=["jsonl", "json", "csv", "tsv", "parquet"], key="merge_uploads")
    raw_paths = st.text_area("Ou chemins locaux, un par ligne", key="merge_paths", height=120,
                             placeholder="D:\\Datasets\\lot_1.jsonl\nD:\\Datasets\\lot_2.jsonl")
    paths = [Path(line.strip().strip('"')) for line in raw_paths.splitlines() if line.strip()]
    repair = st.checkbox("Réparer les erreurs JSONL simples pendant la fusion", value=True,
                          key="merge_repair")
    signatures = [(upload.name, upload.size, getattr(upload, "file_id", None)) for upload in uploads]
    for path in paths:
        try:
            info = path.stat()
            signatures.append((str(path.resolve()), info.st_size, info.st_mtime_ns))
        except OSError:
            signatures.append((str(path), None, None))
    signature = (tuple(signatures), repair, 2)
    if st.session_state.get("merge_signature") != signature:
        st.session_state.pop("merge_analysis", None)
    file_count = len(uploads) + len(paths)
    st.caption("Choisis au moins deux fichiers. Ils sont lus dans l'ordre de sélection, "
               "puis dans l'ordre des chemins locaux.")
    if st.button("Analyser les fichiers à fusionner", disabled=file_count < 2, type="primary"):
        sources = [(upload.name, io.BytesIO(upload.getvalue())) for upload in uploads]
        sources.extend((str(path), path) for path in paths)
        with st.spinner("Lecture des fichiers et recherche des doublons..."):
            st.session_state.merge_analysis = analyze_merge(sources, repair_jsonl=repair)
        st.session_state.merge_signature = signature
    analysis = st.session_state.get("merge_analysis")
    if analysis is None:
        return
    st.subheader("Bilan par fichier")
    st.dataframe(pd.DataFrame(analysis.sources), hide_index=True, width="stretch")
    if analysis.repairs:
        st.success(f"{len(analysis.repairs)} ligne(s) JSONL réparée(s).")
        with st.expander("Détail des réparations"):
            for repair_note in analysis.repairs[:30]:
                st.write(repair_note)
    if analysis.errors:
        st.warning("Certaines lignes ou certains fichiers n'ont pas pu être lus. "
                   "Ils ne seront pas inclus dans la fusion ; consulte le détail ci-dessous.")
        with st.expander("Erreurs de lecture", expanded=True):
            for error in analysis.errors[:30]:
                st.write(error)
    st.subheader("Préparer le dataset final")
    auto_convert = st.checkbox("Convertir automatiquement les entrées récupérables", value=True,
                               key="merge_auto_convert",
                               help="Retire les messages vides et normalise les formats reconnus, "
                                    "sans inventer de réponses ni retirer de contenu non textuel.")
    deduplicate = st.checkbox("Retirer les doublons entre tous les fichiers", value=True,
                              key="merge_deduplicate",
                              help="Compare les conversations après normalisation des rôles. "
                                   "Les différences de casse et d'espaces sont ignorées. "
                                   "La première occurrence retenue est conservée.")
    exclude = st.checkbox("Écarter les entrées incompatibles ou incomplètes", value=True,
                           key="merge_exclude",
                           help="Écarte les entrées sans échange utilisateur/assistant et les conversions "
                                "qui omettraient des images, des appels d'outils ou des messages non reconnus.")
    shuffle = st.checkbox("Mélanger les conversations", value=True, key="merge_shuffle")
    auto_tags = st.checkbox("Ajouter les tags automatiques après la fusion", value=True, key="merge_auto_tags",
                            help="Classe les entrées actives localement. Les corrections manuelles "
                                 "des projets sources sont conservées ; les tags restent hors du JSONL ChatML.")
    seed = int(st.number_input("Graine du mélange", value=42, min_value=0, step=1,
                                disabled=not shuffle, key="merge_seed",
                                help="La même valeur permet de retrouver le même ordre avec les mêmes fichiers."))
    selection = prepare_merge(analysis, remove_duplicates=deduplicate, exclude_incompatible=exclude,
                              shuffle=shuffle, seed=seed, auto_convert=auto_convert)
    a, b, c, d = st.columns(4)
    a.metric("Conversations dans l'export Unsloth", f"{selection.convertible:,}".replace(",", " "))
    b.metric("Doublons écartés", selection.duplicates)
    c.metric("Entrées incompatibles écartées", selection.excluded)
    d.metric("Entrées converties", selection.converted)
    if selection.already_deleted:
        st.caption(f"{selection.already_deleted} entrée(s) déjà dans la corbeille des projets sources restent exclues.")
    unresolved = (analysis.incompatible | analysis.partial) - (set(analysis.recoveries) if auto_convert else set())
    if not exclude and unresolved:
        st.warning("Les entrées non convertibles resteront dans le projet mais seront absentes du JSONL Unsloth. "
                   "Les conversions incomplètes peuvent omettre des messages ou des blocs non textuels.")
    st.caption("Les entrées écartées restent récupérables dans la corbeille du projet fusionné. "
               "Le JSONL Unsloth contient uniquement les messages ; les métadonnées restent dans le projet.")
    if st.session_state.get("dataset_name"):
        st.info("La fusion ouvrira un nouveau dataset dans cet espace de travail. "
                "Exporte d'abord le projet actuel si tu souhaites conserver des changements non sauvegardés.")
    if st.button("Fusionner et préparer l'export Unsloth", disabled=selection.convertible == 0, type="primary"):
        with st.spinner("Fusion et préparation du JSONL Unsloth..."):
            if auto_tags:
                tag_entries(selection.rows)
            blob, amount = export_jsonl(selection.rows, chatml=True)
        report = {"fichiers": analysis.sources, "conversations_exportees": amount,
                  "doublons_ecartes": selection.duplicates, "incompatibles_ecartees": selection.excluded,
                  "deja_supprimees": selection.already_deleted, "melange": shuffle, "graine": seed,
                  "conversions_automatiques": selection.converted,
                  "tags_automatiques": auto_tags,
                  "erreurs": analysis.errors, "reparations": analysis.repairs}
        on_load(selection.rows, "fusion_dataset.jsonl", analysis.errors, analysis.repairs, analysis.origins)
        st.session_state.merge_report = report
        st.session_state.export_chatml = (blob, amount)
        st.session_state.pop("merge_analysis", None)
        st.session_state.navigate_after_action = "📦 Exporter"
        st.rerun()
