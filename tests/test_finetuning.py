from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import zipfile

from streamlit.testing.v1 import AppTest

from core import Entry
from finetuning import (VERSION, Runner, create_job, digest, exclusion_report, export_bundle, process_item,
                        restore_checkpoint, save_checkpoint, source_parts, source_text)
from finetuning_ui import PAGE
from llm_tagging import QwenContextError, QwenError, QwenUnavailable
from qwen_ui import RULES


TEXT = ("Lina ouvre la fenêtre de la bibliothèque. Elle observe la pluie dans la cour et cherche "
        "le carnet rouge posé sur la table. Hugo arrive avec une lettre, puis attend en silence près "
        "de la porte. Elle range doucement le carnet dans son sac et lui demande de patienter.")
PROMPT = ("Raconte en français comment Lina cherche le carnet rouge dans la bibliothèque pendant "
          "que Hugo arrive avec une lettre, en conservant une narration à la troisième personne au présent.")


def rows(count=1):
    return [Entry(index + 1, {"messages": [{"role": "user", "content": "Ancienne demande."},
                 {"role": "assistant", "content": TEXT + f" Elle compte {index + 1} lettres."}]}, index + 1)
            for index in range(count)]


def generated(system, prompt, *, schema, max_tokens):
    if "consigne" in schema["properties"]:
        return json.dumps({"consigne": PROMPT})
    return json.dumps({"verdict": "OK", "motif": "Fidèle."})


def read_bundle(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as bundle:
        return {name: [json.loads(line) for line in bundle.read(name).decode().splitlines() if line]
                for name in bundle.namelist() if name.endswith(".jsonl")}


class TestFinetuning(unittest.TestCase):
    def test_preparation_has_no_theme_filter(self):
        # Clinical wording and genre labels, with no explicit scene or real corpus.
        data = [Entry(1, {"text": TEXT + " Le glossaire définit le terme clitoris."}, 1),
                Entry(2, {"text": TEXT + " Genres indiqués : romance adulte, horreur, érotisme."}, 2)]
        job = create_job(data, "themes.jsonl", "m")
        self.assertFalse(job["results"])
        self.assertTrue(all(not item["flags"] for item in job["items"]))
        self.assertEqual(job["parameters"]["automatic_exclusions"], "duplicates_and_structure")

    def test_short_long_and_editorial_passages_are_warnings_only(self):
        data = [Entry(1, {"text": "Bonjour, Lina."}, 1),
                Entry(2, {"text": TEXT * 30}, 2),
                Entry(3, {"text": "Copyright 2026. ISBN 1234567890. Tous droits réservés. " + TEXT}, 3)]
        job = create_job(data, "warnings.jsonl", "m")
        self.assertFalse(job["results"])
        self.assertTrue(all(item["warnings"] for item in job["items"]))
        self.assertTrue(all(not item["flags"] for item in job["items"]))

    def test_only_duplicate_and_structural_errors_block_preparation(self):
        data = [Entry(1, {"text": TEXT}, 1), Entry(2, {"text": TEXT}, 2),
                Entry(3, {"text": ""}, 3), Entry(4, {"messages": "invalid"}, 4),
                Entry(5, {"text": "Un court texte conservé."}, 5)]
        job = create_job(data, "technical.jsonl", "m")
        self.assertEqual({key: result["status"] for key, result in job["results"].items()},
                         {"2": "exclu", "3": "a_revoir", "4": "a_revoir"})
        self.assertNotIn("5", job["results"])

    def test_manual_validation_can_override_quality_warnings_but_not_empty_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            job = create_job(rows(), "x", "m")
            job["items"][0]["warnings"] = ["Longueur à vérifier."]
            job["results"]["1"] = process_item(job["items"][0], generated)
            self.assertEqual(job["results"]["1"]["status"], "a_revoir")
            self.assertTrue(job["results"]["1"]["eligible"])
            runner = Runner(job, Path(directory) / "state.json")
            custom = "Ma consigne libre :\n[un format choisi par l'utilisateur]"
            runner.review("1", custom, True)
            self.assertEqual(runner.snapshot()["results"]["1"]["status"], "valide")
            exported = read_bundle(export_bundle(runner.snapshot())[0])
            self.assertEqual(exported["dataset.jsonl"][0]["messages"][0]["content"], custom)
            with self.assertRaises(QwenError):
                runner.review("1", " ", True)

    def test_exclusion_report_groups_duplicate_ids_and_keeps_exact_reasons(self):
        job = create_job(rows(4), "x", "m")
        for index, item in enumerate(job["items"][:3]):
            reason = f"Doublon du passage {index + 10}."
            if index == 2:
                reason = "Passage sexualisé détecté. " + reason
            item["flags"] = [reason]
            job["results"][item["id"]] = {"status": "exclu", "reason": reason}
        before = deepcopy(job)
        report = exclusion_report(job)
        self.assertEqual(report["total"], 3)
        self.assertEqual([row["Entrées"] for row in report["summary"]], [2, 1])
        self.assertEqual(report["summary"][0]["Motif regroupé"], "Doublon de texte.")
        self.assertEqual(report["rows"][0]["Motif"], "Doublon du passage 10.")
        self.assertEqual(report["rows"][0]["Ligne source"], 1)
        self.assertTrue(all(row["Étape"] == "Contrôles avant génération" for row in report["rows"]))
        self.assertEqual(job, before)

    def test_exclusion_report_distinguishes_model_manual_and_pending_decisions(self):
        job = create_job(rows(4), "x", "m")
        job["results"]["1"] = {"status": "exclu", "reason": "Contenu non narratif.", "human_review": False}
        job["results"]["2"] = {"status": "exclu", "reason": "Décision manuelle.", "human_review": True}
        job["results"]["3"] = {"status": "a_revoir", "reason": "Doute."}
        report = exclusion_report(job)
        self.assertEqual(report["total"], 2)
        self.assertEqual([row["Étape"] for row in report["rows"]], ["Classement du modèle", "Décision manuelle"])
        self.assertEqual([row["Entrée"] for row in report["rows"]], ["1", "2"])

    def test_exclusion_report_handles_older_missing_reasons_without_inventing_one(self):
        job = create_job(rows(2), "x", "m")
        job["results"] = {"1": {"status": "exclu"}, "2": {"status": "exclu", "reason": ""}}
        job["items"][0]["flags"] = ["Doublon du passage 7."]
        report = exclusion_report(job)
        self.assertEqual(report["rows"][0]["Motif"], "Doublon du passage 7.")
        self.assertIn("non enregistré", report["rows"][1]["Motif"])

    def test_exclusion_report_does_not_include_source_text_or_other_statuses(self):
        job = create_job(rows(), "x", "m")
        self.assertEqual(exclusion_report(job), {"total": 0, "summary": [], "rows": []})
        job["results"]["1"] = {"status": "exclu", "reason": "Doublon du passage 2."}
        self.assertNotIn(TEXT, json.dumps(exclusion_report(job), ensure_ascii=False))

    def test_prepare_is_copy_only_and_tracks_cleaning_and_dedup(self):
        data = rows()
        data[0].data["messages"][1]["content"] = "Chapitre 2\n\n" + TEXT + "\n* * *\n"
        data.append(deepcopy(data[0]))
        data[1].uid = 2
        data.append(Entry(3, {"text": "Court."}, 3, deleted=True))
        before = deepcopy(data)
        job = create_job(data, "test.jsonl", "teacher")
        self.assertEqual(data, before)
        self.assertEqual(len(job["items"]), 2)
        self.assertEqual(job["items"][0]["text"], TEXT)
        self.assertEqual(job["results"]["2"]["status"], "exclu")
        self.assertNotIn("1", job["results"])

    def test_raw_text_accepted_and_multiturn_not_silently_flattened(self):
        self.assertEqual(source_text({"text": TEXT}), (TEXT, ""))
        multi = rows()[0].data
        multi["messages"].append({"role": "user", "content": "Suite ?"})
        with self.assertRaises(ValueError):
            source_text(multi)
        job = create_job([Entry(1, multi, 1)], "x", "m")
        self.assertFalse(job["results"]["1"]["eligible"])
        self.assertNotIn("trop court", job["results"]["1"]["reason"])

    def test_optional_system_is_kept_in_checkpoint_exports_and_review(self):
        system = "Tu écris des scènes françaises.\nConserve les apostrophes d’origine et le présent."
        data = rows()
        data[0].data["messages"].insert(0, {"role": "system", "content": system})
        before = deepcopy(data)
        source = source_parts(data[0].data)
        self.assertEqual(source, (data[0].data["messages"][-1]["content"], "Ancienne demande.", system))
        self.assertEqual(source_text(data[0].data), source[:2])
        job = create_job(data, "system_demo.jsonl", "teacher")
        self.assertFalse(job["results"])
        generate = Mock(side_effect=generated)
        job["results"]["1"] = process_item(job["items"][0], generate)
        # A source system prompt is preserved as data, never promoted to teacher instructions.
        self.assertTrue(all(system not in " ".join(call.args) for call in generate.call_args_list))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_checkpoint(job, path)
            restored = create_job(data, "system_demo.jsonl", "teacher")
            restore_checkpoint(restored, path.read_bytes())
            self.assertEqual(restored["items"][0]["system"], system)
        blob, summary = export_bundle(job, include_proposals=True)
        exported = read_bundle(blob)
        for filename in ("dataset.jsonl", "train.jsonl"):
            messages = exported[filename][0]["messages"]
            self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
            self.assertEqual(messages[0]["content"], system)
            self.assertEqual(messages[1]["content"], PROMPT)
            self.assertEqual(messages[2]["content"], source[0])
        self.assertEqual(summary["with_system"], 1)
        self.assertEqual(exported["provenance.jsonl"][0]["system_sha256"], digest(system))
        review, _ = export_bundle(job)
        self.assertEqual(read_bundle(review)["a_revoir.jsonl"][0]["system"], system)
        self.assertEqual(data, before)

    def test_optional_system_supports_sharegpt_alpaca_and_empty_system(self):
        records = [
            {"conversations": [{"from": "système", "value": "Écris en français."},
                               {"from": "human", "value": "Raconte la scène."},
                               {"from": "gpt", "value": TEXT}]},
            {"system": "Écris en français.", "instruction": "Raconte la scène.", "output": TEXT},
            {"messages": [{"role": "system", "content": ""},
                          {"role": "user", "content": "Raconte la scène."},
                          {"role": "assistant", "content": TEXT}]},
            {"system": "", "instruction": "Raconte la scène.", "output": TEXT},
        ]
        for index, record in enumerate(records):
            with self.subTest(index=index):
                answer, prompt, system = source_parts(record)
                self.assertEqual(answer, TEXT)
                self.assertEqual(prompt, "Raconte la scène.")
                self.assertEqual(system, "" if index >= 2 else "Écris en français.")
                job = create_job([Entry(1, record, 1)], "system.jsonl", "m")
                self.assertNotIn("1", job["results"])
        self.assertEqual(source_parts({"text": TEXT, "system": "Français."}), (TEXT, "", "Français."))

    def test_bad_system_order_tools_multiturn_and_multimodal_remain_rejected(self):
        user = {"role": "user", "content": "Raconte."}
        assistant = {"role": "assistant", "content": TEXT}
        system = {"role": "system", "content": "Style français."}
        sequences = [
            [user, system, assistant], [system, system, user, assistant],
            [system, user, assistant, user, assistant],
            [user, dict(assistant, tool_calls=[{"id": "call"}])],
            [user, {"role": "assistant", "content": [{"type": "text", "text": TEXT}]}],
            [user, {"role": "unknown", "content": TEXT}],
            [system, {"role": "user", "content": ""}, assistant],
        ]
        for index, sequence in enumerate(sequences):
            with self.subTest(index=index), self.assertRaises(ValueError):
                source_parts({"messages": sequence})

    def test_previous_reader_checkpoint_is_not_reused(self):
        job = create_job(rows(), "x", "m")
        for version in ("narration-dataset-v1", "narration-dataset-v2"):
            old = deepcopy(job)
            old["parameters"]["version"] = version
            self.assertNotEqual(VERSION, version)
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "incompatible"):
                restore_checkpoint(job, json.dumps(old))

    def test_generation_keeps_target_and_uses_new_consigne_not_old_prompt(self):
        job = create_job(rows(), "x", "m")
        generate = Mock(side_effect=generated)
        result = process_item(job["items"][0], generate)
        self.assertEqual(result["status"], "propose")
        self.assertEqual(result["prompt"], PROMPT)
        self.assertFalse(result["human_review"])
        self.assertEqual(generate.call_count, 2)
        self.assertNotIn("Ancienne demande", generate.call_args_list[0].args[1])
        self.assertIn("non fiables", generate.call_args.args[0])

    def test_model_has_no_content_exclusion_operation(self):
        item = create_job(rows(), "x", "m")["items"][0]
        generate = Mock(side_effect=generated)
        result = process_item(item, generate)
        self.assertEqual(result["status"], "propose")
        self.assertEqual([set(call.kwargs["schema"]["properties"]) for call in generate.call_args_list],
                         [{"consigne"}, {"verdict", "motif"}])
        self.assertTrue(result["eligible"])

    def test_critic_and_lexical_errors_are_kept_for_review(self):
        def wrong(system, prompt, *, schema, max_tokens):
            if "verdict" in schema["properties"]:
                return json.dumps({"verdict": "FAIT", "motif": "Un lien familial est ajouté."})
            if "consigne" in schema["properties"]:
                return json.dumps({"consigne": "Raconte la fuite du capitaine vers une planète perdue dans le cosmos."})
            return generated(system, prompt, schema=schema, max_tokens=max_tokens)
        result = process_item(create_job(rows(), "x", "m")["items"][0], wrong)
        self.assertEqual(result["status"], "a_revoir")
        self.assertIn("ancrée", result["reason"])
        self.assertIn("FAIT", result["reason"])

    def test_user_can_export_unreviewed_warnings_with_their_report(self):
        job = create_job(rows(2), "x", "m")
        job["results"]["1"] = dict(process_item(job["items"][0], generated),
                                    status="a_revoir", reason="DOUTE : consigne à vérifier.")
        job["results"]["2"] = {"status": "a_revoir", "prompt": "", "eligible": False,
                                 "human_review": False, "reason": "Erreur de contexte."}
        default, _ = export_bundle(job)
        self.assertFalse(read_bundle(default)["dataset.jsonl"])
        included, _ = export_bundle(job, include_proposals=True)
        report = read_bundle(included)
        self.assertEqual(len(report["dataset.jsonl"]), 1)
        self.assertEqual(len(report["a_revoir.jsonl"]), 2)
        self.assertEqual(report["provenance.jsonl"][0]["review_status"], "a_revoir")
        self.assertIn("DOUTE", report["provenance.jsonl"][0]["review_reason"])

    def test_checkpoint_rejects_changed_data_settings_and_invalid_results(self):
        job = create_job(rows(), "x", "m")
        for changed in (create_job(rows(), "x", "another"), create_job(rows(), "x", "m", clean=False)):
            with self.assertRaises(ValueError):
                restore_checkpoint(changed, json.dumps(job))
        bad = deepcopy(job)
        bad["results"]["1"] = {"status": "valide", "prompt": PROMPT, "eligible": False, "human_review": True}
        with self.assertRaises(ValueError):
            restore_checkpoint(job, json.dumps(bad))

    def test_preview_resume_error_and_manual_review_are_checkpointed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            runner = Runner(create_job(rows(3), "x", "m"), path)
            runner._run(generated, 1)
            self.assertEqual(len(runner.snapshot()["results"]), 1)
            restored = create_job(rows(3), "x", "m")
            restore_checkpoint(restored, path.read_bytes())
            runner = Runner(restored, path)
            failure = Mock(side_effect=QwenUnavailable("Réserve RAM atteinte"))
            runner._run(failure, None)
            failure.assert_called_once()
            self.assertIn("2", runner.snapshot()["errors"])
            runner._run(generated, None)
            self.assertEqual(len(runner.snapshot()["results"]), 3)
            self.assertFalse(runner.snapshot()["errors"])
            runner.review("1", PROMPT, True)
            self.assertTrue(runner.snapshot()["results"]["1"]["human_review"])
            self.assertEqual(runner.snapshot()["results"]["1"]["status"], "valide")

    def test_background_pause_stops_after_current_example(self):
        started, release = threading.Event(), threading.Event()
        def slow(*args, **kwargs):
            started.set()
            self.assertTrue(release.wait(5))
            return generated(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(create_job(rows(3), "x", "m"), Path(directory) / "state.json")
            runner.start(slow)
            self.assertTrue(started.wait(5))
            self.assertFalse(runner.start(generated))
            runner.stop.set()
            release.set()
            runner.thread.join(5)
            self.assertFalse(runner.running)
            self.assertEqual(len(runner.snapshot()["results"]), 1)

    def test_oversized_context_goes_to_review_without_stopping_next_item(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(create_job(rows(2), "x", "m"), Path(directory) / "state.json")
            def generate(system, prompt, **kwargs):
                if "compte 1 lettres" in prompt:
                    raise QwenContextError("Passage trop long, aucune troncature appliquée.")
                return generated(system, prompt, **kwargs)
            runner._run(generate, None)
            job = runner.snapshot()
            self.assertEqual(job["results"]["1"]["status"], "a_revoir")
            self.assertEqual(job["results"]["2"]["status"], "propose")
            self.assertFalse(job["errors"])

    def test_exports_require_review_and_group_split_avoids_duplicate_leaks(self):
        data = rows(64)
        data.append(Entry(65, deepcopy(data[0].data), 65))
        job = create_job(data, "x", "m")
        for item in job["items"][:64]:
            job["results"][item["id"]] = process_item(item, generated)
        empty, _ = export_bundle(job)
        self.assertFalse(read_bundle(empty)["dataset.jsonl"])
        blob, summary = export_bundle(job, include_proposals=True)
        exported = read_bundle(blob)
        self.assertEqual(len(exported["dataset.jsonl"]), 64)
        self.assertEqual(len(exported["exclusions.jsonl"]), 1)
        self.assertTrue(summary["complete"])
        train = {row["messages"][1]["content"] for row in exported["train.jsonl"]}
        validation = {row["messages"][1]["content"] for row in exported["validation.jsonl"]}
        self.assertFalse(train & validation)
        self.assertEqual(exported["dataset.jsonl"][0]["messages"][1]["content"], data[0].data["messages"][1]["content"])
        provenance = exported["provenance.jsonl"]
        train_groups = {row["group"] for row in provenance if row["partition"] == "train"}
        val_groups = {row["group"] for row in provenance if row["partition"] == "validation"}
        self.assertFalse(train_groups & val_groups)
        for value in provenance:
            row = exported[value["partition"] + ".jsonl"][value["partition_line"] - 1]
            self.assertEqual(digest(row["messages"][1]["content"]), value["target_sha256"])
        self.assertEqual(blob, export_bundle(job, include_proposals=True)[0])

    def test_single_group_gives_empty_validation_and_no_empty_conversation(self):
        job = create_job(rows(), "x", "m")
        job["results"]["1"] = dict(process_item(job["items"][0], generated), status="valide", human_review=True)
        blob, _ = export_bundle(job)
        self.assertEqual(len(read_bundle(blob)["train.jsonl"]), 1)
        self.assertFalse(read_bundle(blob)["validation.jsonl"])


class TestFinetuningUI(unittest.TestCase):
    def open_page(self, data):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
        app.session_state.rows = data
        app.session_state.dataset_name = "system_demo.jsonl"
        app.session_state.tagging_engine = RULES
        app.session_state.ft_model = "not-loaded.gguf"
        app.run()
        next(r for r in app.radio if r.label == "Navigation").set_value(PAGE).run()
        self.assertFalse(app.exception)
        return app

    def test_warning_only_record_can_be_processed_without_loading_on_prepare(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("finetuning_ui.CHECKPOINTS", Path(directory)), \
                patch("finetuning_ui.SETTINGS", Path(directory) / "settings.json"), \
                patch("qwen_ui.LocalQwen.load") as load:
            app = self.open_page([Entry(1, {"text": "Un court texte conservé."}, 1)])
            next(b for b in app.button if b.label == "Préparer / reprendre le dataset").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(exp.label.startswith("Avertissements sans exclusion") for exp in app.expander))
            self.assertFalse(next(b for b in app.button if b.label == "Générer un aperçu (3 exemples)").disabled)
            self.assertFalse(app.session_state.ft_runner.snapshot()["results"])
            load.assert_not_called()

    def test_existing_exclusions_show_summary_and_exact_line_without_reprocessing(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("finetuning_ui.CHECKPOINTS", Path(directory)), \
                patch("finetuning_ui.SETTINGS", Path(directory) / "settings.json"), \
                patch("qwen_ui.LocalQwen.load") as load, \
                patch("finetuning_ui.generate_for_config") as generate:
            data = rows(3)
            data[1].data = deepcopy(data[0].data)
            data[2].data = deepcopy(data[0].data)
            app = self.open_page(data)
            next(b for b in app.button if b.label == "Préparer / reprendre le dataset").click().run()
            self.assertFalse(app.exception)
            job = app.session_state.ft_runner.snapshot()
            path = Path(app.session_state.ft_runner.checkpoint)
            saved = path.read_bytes()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(path.read_bytes(), saved)
            self.assertEqual(app.session_state.ft_runner.snapshot(), job)
            self.assertTrue(any(exp.label == "Motifs des exclusions (2)" for exp in app.expander))
            summary = next(frame.value for frame in app.dataframe if "Motif regroupé" in frame.value.columns)
            detail = next(frame.value for frame in app.dataframe if "Ligne source" in frame.value.columns)
            self.assertEqual(summary["Entrées"].tolist(), [2])
            self.assertEqual(detail["Entrée"].tolist(), ["2", "3"])
            self.assertEqual(detail["Motif"].tolist(), ["Doublon du passage 1."] * 2)
            self.assertEqual(detail["Ligne source"].tolist(), [2, 3])
            load.assert_not_called()
            generate.assert_not_called()

    def test_no_exclusion_does_not_display_empty_exclusion_tables(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("finetuning_ui.CHECKPOINTS", Path(directory)), \
                patch("finetuning_ui.SETTINGS", Path(directory) / "settings.json"):
            app = self.open_page(rows())
            next(b for b in app.button if b.label == "Préparer / reprendre le dataset").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(any(exp.label.startswith("Motifs des exclusions") for exp in app.expander))

    def test_three_message_dataset_can_prepare_preview_and_display_system(self):
        data = rows(4)
        for row in data:
            row.data["messages"].insert(0, {"role": "system", "content": "Écris des scènes en français."})
        with tempfile.TemporaryDirectory() as directory, \
                patch("finetuning_ui.CHECKPOINTS", Path(directory)), \
                patch("finetuning_ui.SETTINGS", Path(directory) / "settings.json"), \
                patch("qwen_ui.LocalQwen.load") as load:
            app = self.open_page(data)
            next(b for b in app.button if b.label == "Préparer / reprendre le dataset").click().run()
            self.assertFalse(app.exception)
            state = app.session_state.ft_runner.snapshot()
            self.assertFalse(state["results"])
            self.assertTrue(all(item["system"] == "Écris des scènes en français." for item in state["items"]))
            self.assertFalse(next(b for b in app.button if b.label == "Générer un aperçu (3 exemples)").disabled)
            def fake(config, *args, _pool=None, **kwargs):
                return generated(*args, **kwargs)
            with patch("finetuning_ui.QwenConfig.validate"), patch("finetuning_ui.generate_for_config", side_effect=fake):
                next(b for b in app.button if b.label == "Générer un aperçu (3 exemples)").click().run()
                app.session_state.ft_runner.thread.join(5)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state.ft_runner.snapshot()["results"]), 3)
            self.assertTrue(any("Message système conservé" in expander.label for expander in app.expander))
            load.assert_not_called()

    def test_legacy_session_requires_prepare_and_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("finetuning_ui.CHECKPOINTS", Path(directory)), \
                patch("finetuning_ui.SETTINGS", Path(directory) / "settings.json"), \
                patch("qwen_ui.LocalQwen.load") as load:
            app = self.open_page(rows())
            next(b for b in app.button if b.label == "Préparer / reprendre le dataset").click().run()
            old = app.session_state.ft_runner.snapshot()
            old["parameters"]["version"] = "narration-dataset-v1"
            old["id"] = "legacy_state"
            path = Path(directory) / "legacy.json"
            save_checkpoint(old, path)
            original = path.read_bytes()
            app.session_state.ft_runner = Runner(old, path)
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any("anciens filtres" in message.value for message in app.warning))
            self.assertTrue(next(b for b in app.button if b.label == "Générer un aperçu (3 exemples)").disabled)
            next(b for b in app.button if b.label == "Préparer / reprendre le dataset").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.ft_runner.snapshot()["parameters"]["version"], VERSION)
            self.assertEqual(path.read_bytes(), original)
            load.assert_not_called()

    def test_all_incompatible_records_show_reason_without_a_false_shortness_warning(self):
        data = rows()
        data[0].data["messages"].append({"role": "user", "content": "Suite ?"})
        with tempfile.TemporaryDirectory() as directory, \
                patch("finetuning_ui.CHECKPOINTS", Path(directory)), \
                patch("finetuning_ui.SETTINGS", Path(directory) / "settings.json"):
            app = self.open_page(data)
            next(b for b in app.button if b.label == "Préparer / reprendre le dataset").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Aucun passage" in message.value for message in app.warning))
            self.assertTrue(any("plusieurs tours" in text.value for text in app.text))
            self.assertFalse(any("trop court" in text.value for text in app.text))

    def test_open_and_prepare_never_load_model_and_stale_settings_disable_generation(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("finetuning_ui.CHECKPOINTS", Path(directory)), \
                patch("finetuning_ui.SETTINGS", Path(directory) / "settings.json"), \
                patch("finetuning_ui.generate_for_config") as generate, \
                patch("qwen_ui.LocalQwen.load") as load:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
            app.session_state.rows = rows(4)
            app.session_state.dataset_name = "demo.jsonl"
            app.session_state.tagging_engine = RULES
            app.session_state.ft_model = "not-loaded.gguf"
            app.run()
            next(r for r in app.radio if r.label == "Navigation").set_value(PAGE).run()
            self.assertFalse(app.exception)
            next(b for b in app.button if b.label == "Préparer / reprendre le dataset").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state.ft_runner.snapshot()["items"]), 4)
            load.assert_not_called()
            generate.assert_not_called()
            def fake_generator(config, *args, _pool=None, **kwargs):
                return generated(*args, **kwargs)
            with patch("finetuning_ui.QwenConfig.validate"), patch("finetuning_ui.generate_for_config", side_effect=fake_generator) as fake:
                next(b for b in app.button if b.label == "Générer un aperçu (3 exemples)").click().run()
                app.session_state.ft_runner.thread.join(5)
                self.assertEqual(fake.call_count, 6)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state.ft_runner.snapshot()["results"]), 3)
            next(n for n in app.number_input if n.label == "Couches sur le GPU").set_value(20).run()
            self.assertTrue(next(b for b in app.button if b.label == "Traiter tout / reprendre").disabled)
            self.assertEqual(len(app.session_state.rows), 4)


if __name__ == "__main__":
    unittest.main()
