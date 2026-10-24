"""Explicit controls for local dataset optimisation and model resource limits."""
from __future__ import annotations

from dataclasses import asdict
from collections import Counter
from functools import partial
import json
from pathlib import Path

import streamlit as st

from finetuning import (VERSION, Runner, create_job, digest, exclusion_report, export_bundle,
                        restore_checkpoint, save_checkpoint)
from install_assets import resolve_setting
from llm_tagging import BASE, DEFAULT_SERVER, QwenConfig, QwenError
from memory_guard import memory_snapshot
from qwen_ui import clients, generate_for_config


SETTINGS = BASE / "finetuning_settings.json"
CHECKPOINTS = BASE / "data" / "finetuning"
PAGE = "🛠️ Fine-tuning FR"


def initialize_settings():
    defaults = {"model": "", "server": DEFAULT_SERVER, "gpu": 32, "context": 4096,
                "batch": 128, "ubatch": 64, "cache": 1024, "ram": 5.0, "vram": 1536}
    try:
        saved = json.loads(SETTINGS.read_text(encoding="utf-8-sig"))
        if isinstance(saved, dict):
            for key, default in defaults.items():
                value = saved.get(key)
                if type(value) is type(default):
                    defaults[key] = value
    except (OSError, ValueError):
        pass
    limits = {"gpu": (0, 999), "context": (2048, 32768), "batch": (32, 512),
              "ubatch": (16, 512), "cache": (0, 2048), "ram": (1., 32.), "vram": (256, 16384)}
    for key, (minimum, maximum) in limits.items():
        defaults[key] = max(minimum, min(maximum, defaults[key]))
    defaults["ubatch"] = min(defaults["ubatch"], defaults["batch"])
    for key, value in defaults.items():
        st.session_state.setdefault("ft_" + key, value)
    return tuple(defaults)


def model_config():
    return QwenConfig(model=resolve_setting(st.session_state.ft_model, BASE),
        server=resolve_setting(st.session_state.ft_server, BASE),
        gpu_layers=int(st.session_state.ft_gpu), context=int(st.session_state.ft_context),
        batch=int(st.session_state.ft_batch), ubatch=int(st.session_state.ft_ubatch),
        cache_mib=int(st.session_state.ft_cache), low_memory=True,
        reserve_ram_gib=float(st.session_state.ft_ram), reserve_vram_mib=int(st.session_state.ft_vram))


def model_signature(config):
    value = asdict(config)
    for label in ("model", "server"):
        try:
            stat = Path(value[label]).stat()
            value[label + "_file"] = [stat.st_size, stat.st_mtime_ns]
        except OSError:
            value[label + "_file"] = None
    return value


def render_exclusion_details(job, *, width="stretch"):
    report = exclusion_report(job)
    if not report["total"]:
        return
    with st.expander(f"Motifs des exclusions ({report['total']})", expanded=True):
        st.caption("Ces motifs proviennent de la préparation en cours : contrôles avant génération, "
                   "classement du modèle ou décision manuelle. Chaque entrée compte une fois dans le résumé ; "
                   "les numéros de doublons y sont regroupés. Les décisions automatiques restent à vérifier.")
        st.dataframe(report["summary"], hide_index=True, width=width,
                     key=f"ft_exclusions_summary_{job['id']}")
        st.markdown("**Détail par entrée : numéro, ligne source et motif exact**")
        st.dataframe(report["rows"], hide_index=True, width=width, height=260,
                     key=f"ft_exclusions_rows_{job['id']}")
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in report["rows"])
        st.download_button("Télécharger les motifs d'exclusion", payload.encode("utf-8"),
                           f"{Path(job['parameters']['name']).stem}_motifs_exclusions.jsonl",
                           "application/x-ndjson", key="ft_exclusion_download")


def render_preparation_report(job):
    pending = len(job["items"]) - len(job["results"])
    structural = [item for item in job["items"] if item["flags"]]
    warnings = [item for item in job["items"] if item.get("warnings")]
    st.info(f"{pending} passage(s) restant(s) pour le modèle · "
            f"{len(structural)} doublon(s) ou erreur(s) de structure.")
    if structural:
        with st.expander("Motifs des contrôles avant génération"):
            for reason, count in Counter(reason for item in structural for reason in item["flags"]).most_common():
                st.text(f"{count} entrée(s) : {reason}")
    if warnings:
        with st.expander(f"Avertissements sans exclusion ({len(warnings)} entrées)"):
            st.caption("Ces passages restent disponibles pour le modèle. Tu décides de leur validation.")
            for reason, count in Counter(reason for item in warnings for reason in item["warnings"]).most_common():
                st.text(f"{count} entrée(s) : {reason}")
    if not pending and len(structural) == len(job["items"]):
        st.warning("Aucun passage n'est prêt pour la génération. Consulte les motifs ci-dessus ; "
                   "charger le modèle ne corrigera pas les doublons ou les formats non pris en charge.")


@st.fragment(run_every=1)
def render_progress(runner, stale):
    job = runner.snapshot()
    total, done = len(job["items"]), len(job["results"])
    st.progress(done / total, text=f"{done} / {total} passages traités")
    counts = {status: sum(result["status"] == status for result in job["results"].values())
              for status in ("propose", "valide", "a_revoir", "exclu")}
    columns = st.columns(4)
    for column, (status, label) in zip(columns, (("propose", "Proposés"), ("valide", "Validés par toi"),
                                              ("a_revoir", "À revoir"), ("exclu", "Écartés"))):
        column.metric(label, counts[status])
    render_exclusion_details(job)
    if runner.running:
        st.info("Passe en cours sur ce PC. Tu peux demander une pause ; elle prend effet après l'exemple en cours.")
        if st.button("Pause après l'exemple en cours", key="ft_pause"):
            runner.stop.set()
            st.info("Pause demandée.")
    if runner.notice:
        st.info(runner.notice)
    if job["errors"]:
        with st.expander("Erreurs à réessayer"):
            for key, error in list(job["errors"].items())[:30]:
                st.text(f"Entrée {key} : {error}")
    st.caption(f"Sauvegarde : {runner.checkpoint}. Pour reprendre après fermeture, recharge le même "
               "dataset, retrouve ces réglages et clique sur Préparer / reprendre.")
    if stale:
        st.info("Les données ou les réglages ont changé. Prépare une nouvelle passe avant de générer. "
                "La progression précédente reste disponible sur disque.")
    completed = [item for item in job["items"] if item["id"] in job["results"]]
    if completed:
        st.subheader("Comparer et relire")
        selected = st.selectbox("Passage préparé", [item["id"] for item in completed],
            format_func=lambda key: f"Entrée {key} · {job['results'][key]['status']}", key=f"ft_select_{job['id']}")
        item = next(item for item in completed if item["id"] == selected)
        result = job["results"][selected]
        if item.get("system") is not None:
            with st.expander("Message système conservé dans les exports"):
                st.text(item["system"])
        left, right = st.columns(2)
        with left:
            st.markdown("**Avant : consigne et réponse sources**")
            st.text(item["old_prompt"] or "Texte brut, sans consigne.")
            st.text(item["original"])
        with right:
            st.markdown("**Après : consigne proposée et réponse conservée**")
            prompt = st.text_area("Consigne à relire / corriger", value=result["prompt"],
                key=f"ft_prompt_{job['id']}_{selected}", disabled=runner.running or not result["eligible"])
            st.text(item["text"])
            st.markdown("**Motif de l'exclusion**" if result["status"] == "exclu" else "**Motif du classement**")
            st.text(result.get("reason") or "Motif non enregistré.")
        a, b = st.columns(2)
        accept = a.button("Valider cet exemple", disabled=runner.running or not result["eligible"], key="ft_accept")
        reject = b.button("Écarter cet exemple", disabled=runner.running, key="ft_reject")
        if accept or reject:
            try:
                runner.review(selected, prompt, accept)
                st.success("Décision sauvegardée.")
            except (OSError, ValueError, QwenError) as exc:
                st.error(str(exc))
    st.subheader("Exporter une nouvelle base")
    st.caption("Par défaut, seuls les exemples validés par toi sont placés dans le dataset d'entraînement. "
               "Les propositions automatiques et les cas douteux sont conservés séparément pour relecture.")
    include = st.checkbox("Inclure les propositions et les avertissements non relus", key="ft_include")
    if include:
        st.caption("Les consignes générées signalées à revoir seront aussi exportées. Le rapport a_revoir "
                   "et la provenance conservent leurs avertissements. Les doublons et les erreurs de structure "
                   "ou de génération sans consigne utilisable restent hors du dataset d'entraînement.")
    percent = st.number_input("Validation (% des groupes de source)", 0, 40, 10, key="ft_validation")
    signature = digest([job, include, percent])
    if st.button("Préparer l'archive JSONL", disabled=runner.running, key="ft_export"):
        blob, summary = export_bundle(job, include_proposals=include, validation_percent=int(percent))
        st.session_state.ft_export_data = (signature, blob, summary)
    prepared = st.session_state.get("ft_export_data")
    if prepared and prepared[0] == signature:
        st.json(prepared[2]["counts"])
        st.download_button("Télécharger la nouvelle base", prepared[1],
            f"{Path(job['parameters']['name']).stem}_narration_fr.zip", "application/zip", key="ft_download")


def render_finetuning_page(rows, name, sources=None, *, generator_version=VERSION):
    st.title("🛠️ Préparer un dataset de narration française")
    st.write("Transforme tes textes bruts ou tes paires consigne/réponse, avec message système facultatif, en nouveaux exemples pour le "
             "fine-tuning. Le GGUF sert à construire et contrôler les consignes ; le futur Qwen 9B "
             "sera entraîné plus tard avec les exports.")
    keys = initialize_settings()
    runner = st.session_state.get("ft_runner")
    running = runner is not None and runner.running
    with st.expander("Modèle GGUF et mémoire", expanded=runner is None):
        st.text_input("Modèle GGUF de préparation", key="ft_model", disabled=running,
                      placeholder=r"D:\I.A\GGUF\mon_modele.gguf")
        st.text_input("Serveur llama.cpp", key="ft_server", disabled=running)
        a, b = st.columns(2)
        with a:
            st.number_input("Couches sur le GPU", 0, 999, key="ft_gpu", disabled=running)
            st.number_input("Contexte total (tokens)", 2048, 32768, step=512, key="ft_context", disabled=running)
            st.number_input("Batch de lecture", 32, 512, step=32, key="ft_batch", disabled=running)
            st.number_input("Microbatch", 16, 512, step=16, key="ft_ubatch", disabled=running)
        with b:
            st.number_input("Cache des préfixes (Mio)", 0, 2048, step=128, key="ft_cache", disabled=running)
            st.number_input("Réserve de RAM libre (Gio)", 1., 32., step=0.5, key="ft_ram", disabled=running)
            st.number_input("Réserve de VRAM libre (Mio)", 256, 16384, step=256, key="ft_vram", disabled=running)
        st.caption("Point de départ prudent pour un gros GGUF : 32 couches GPU, contexte 4 096, "
                   "batch 128 / microbatch 64, une requête à la fois. Le modèle reste partagé entre RAM et GPU. "
                   "L'arrêt surveille la mémoire libre pendant le chargement et les requêtes ; ce contrôle "
                   "périodique ne garantit pas un plafond strict. Sans nvidia-smi, seule la RAM est surveillée. "
                   "Avec plusieurs GPU NVIDIA, la mesure VRAM utilise le minimum libre observé.")
        if st.button("Mesurer la mémoire libre", disabled=running):
            snapshot = memory_snapshot()
            ram = f"{snapshot.ram_gib:.1f} Gio" if snapshot.ram_gib is not None else "indisponible"
            vram = f"{snapshot.vram_mib} Mio" if snapshot.vram_mib is not None else "indisponible"
            st.info(f"RAM libre : {ram} · VRAM libre : {vram}")
        if st.button("Enregistrer ces réglages", disabled=running):
            try:
                SETTINGS.write_text(json.dumps({key: st.session_state["ft_" + key] for key in keys},
                    ensure_ascii=False, indent=2), encoding="utf-8")
                st.success("Réglages enregistrés sur ce PC.")
            except OSError as exc:
                st.error(str(exc))
        if st.button("Charger le GGUF de préparation", disabled=running or not st.session_state.ft_model.strip()):
            try:
                with st.spinner("Chargement local avec surveillance mémoire..."):
                    clients().load(model_config())
                st.success("Modèle chargé. Aucun exemple n'a été traité.")
            except QwenError as exc:
                st.error(str(exc))
        if st.button("Décharger le modèle de préparation", disabled=running):
            clients().unload()
            st.success("Modèle déchargé.")
        st.caption("Le moteur est partagé avec les autres pages : un seul modèle est conservé à la fois.")
    if not name:
        st.info("Ouvre d'abord ton dataset depuis la barre latérale. Pour préparer sans charger "
                "de modèle à l'import, décoche « Ajouter les tags automatiquement à l'import ».")
        return
    st.caption(f"Dataset actif : {name}. La préparation crée une copie de travail séparée.")
    clean = st.checkbox("Nettoyer les séparateurs isolés et les en-têtes de chapitre", value=True,
                         key="ft_clean", disabled=running,
                         help="Retire les lignes d'astérisques, un titre de chapitre initial et les espaces aux bords. Aucun récit n'est réécrit.")
    max_chars = int(st.number_input("Seuil d'avertissement sur la longueur (caractères)", 500, 16000, 6000,
                                    500, key="ft_max_chars", disabled=running))
    config = model_config()
    model = model_signature(config)
    signature = digest([[(row.uid, row.data, row.deleted) for row in rows], sources or {},
                        name, model, clean, max_chars])
    st.write("Les seuls rejets automatiques concernent les doublons et les erreurs de structure. "
             "Aucun filtre de thème ou de genre n'est appliqué. Les contrôles de longueur, de langue, "
             "de forme et de cohérence sont des avertissements à relire ; tu peux valider manuellement. "
             "Le modèle prépare la consigne et conserve la réponse source, avec le nettoyage choisi.")
    if st.button("Préparer / reprendre le dataset", type="primary", disabled=running):
        try:
            job = create_job(rows, name, model, clean=clean, max_chars=max_chars, sources=sources)
            checkpoint = CHECKPOINTS / f"{job['id']}.json"
            if checkpoint.is_file():
                restore_checkpoint(job, checkpoint.read_bytes())
            save_checkpoint(job, checkpoint)
            runner = Runner(job, checkpoint)
            st.session_state.ft_runner = runner
            st.session_state.ft_signature = signature
            st.session_state.pop("ft_export_data", None)
        except (OSError, ValueError) as exc:
            st.error(f"Préparation impossible : {exc}")
    if runner is None:
        st.caption("Préparer effectue seulement les contrôles de structure. La génération nécessite ensuite un clic explicite.")
        return
    state = runner.snapshot()
    legacy = state["parameters"].get("version") != generator_version
    stale = legacy or st.session_state.get("ft_signature") != signature
    if legacy:
        st.warning("Cette préparation utilise les anciens filtres. Clique sur Préparer / reprendre "
                   "pour appliquer uniquement les rejets de doublons et d'erreurs de structure. "
                   "L'ancienne sauvegarde reste conservée séparément.")
    pending = len(state["items"]) - len(state["results"])
    if not legacy:
        render_preparation_report(state)
    st.caption("Sans provenance de livre, le découpage train/validation se base sur l'ordre des lignes "
               "du fichier. Les groupes de 16 passages et leur marge de 2 limitent les voisinages ; "
               "ils ne remplacent pas une séparation par chapitre ou par livre contrôlée manuellement.")
    a, b = st.columns(2)
    disabled = running or stale or not pending or not st.session_state.ft_model.strip()
    preview = a.button("Générer un aperçu (3 exemples)", disabled=disabled)
    full = b.button("Traiter tout / reprendre", disabled=disabled)
    if preview or full:
        try:
            config.validate()
            generator = partial(generate_for_config, config, _pool=clients())
            runner.start(generator, limit=3 if preview else None)
        except QwenError as exc:
            st.error(str(exc))
    render_progress(runner, stale)
