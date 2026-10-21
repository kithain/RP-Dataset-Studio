"""Preview and apply recoverable Unsloth conversions in the quality page."""
import pandas as pd
import streamlit as st

from conversation_conversion import analyze_conversions


def render_conversion_panel(rows, mutation):
    st.markdown("#### Convertir les entrées incompatibles pour Unsloth")
    st.write("Analyse les entrées écartées et convertis celles dont le dialogue peut être conservé. "
             "Les messages vides peuvent être retirés et les rôles connus normalisés. "
             "Aucune réponse n'est inventée.")
    notice = st.session_state.pop("conversion_notice", None)
    if notice:
        st.success(notice)
    include_deleted = st.checkbox("Inclure les entrées de la corbeille", key="conversion_include_deleted",
                                  help="Les entrées converties depuis la corbeille seront restaurées. "
                                       "Les doublons déjà compatibles ne sont pas concernés.")
    signature = (st.session_state.revision, include_deleted)
    if st.session_state.get("conversion_signature") != signature:
        st.session_state.pop("conversion_preview", None)
    if st.button("Analyser les entrées incompatibles"):
        with st.spinner("Analyse des conversions possibles..."):
            st.session_state.conversion_preview = analyze_conversions(rows, include_deleted=include_deleted)
        st.session_state.conversion_signature = signature
    if "conversion_preview" in st.session_state:
        candidates = st.session_state.conversion_preview
        recoverable = [candidate for candidate in candidates if candidate.result.data is not None]
        a, b = st.columns(2)
        a.metric("Entrées récupérables", len(recoverable))
        b.metric("À corriger manuellement", len(candidates) - len(recoverable))
        if candidates:
            st.dataframe(pd.DataFrame([
                {"ID": candidate.uid, "Ligne": candidate.source_line,
                 "Fichier": st.session_state.get("entry_sources", {}).get(candidate.uid, ""),
                 "Résultat": "Récupérable" if candidate.result.data is not None else "Manuel",
                 "Détail": "; ".join(candidate.result.notes) or candidate.result.reason}
                for candidate in candidates]), hide_index=True, width="stretch")
        else:
            st.info("Aucune entrée incompatible dans la sélection analysée.")
        st.caption("Les conversions sont annulables. Exporte ensuite le JSONL Unsloth pour conserver le résultat.")
        if st.button(f"Convertir les {len(recoverable)} entrées récupérables", disabled=not recoverable,
                      type="primary"):
            by_uid = {row.uid: row for row in rows}
            before = [(candidate.uid, by_uid[candidate.uid].data, by_uid[candidate.uid].deleted)
                      for candidate in recoverable]
            for candidate in recoverable:
                row = by_uid[candidate.uid]
                row.data = candidate.result.data
                row.deleted = False
            mutation({"kind": "convert", "before": before})
            st.session_state.conversion_notice = (
                f"{len(recoverable)} entrée(s) convertie(s) et active(s). "
                "Tu peux maintenant préparer le JSONL Unsloth dans Exporter.")
            st.rerun()
