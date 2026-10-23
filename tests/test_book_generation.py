from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

import book_ui
from book_generation import (VERSION, checkpoint_bytes, create_job, decode_book, job_rows, make_record,
                             non_narrative_reason, restore_checkpoint, run_job, save_checkpoint,
                             split_book, tasks)
from core import Entry, export_jsonl, export_project, load_dataset
from llm_tagging import LocalQwen, QwenConfig, QwenError, QwenUnavailable, generate_json
from qwen_ui import QWEN_2B, QWEN_4B, RULES


APP = Path(__file__).resolve().parents[1] / "app.py"
SOURCE = ("Lina posa la lettre sur la table. La pluie frappait les vitres de la bibliothèque. "
          "Hugo resta près de la porte. — Tu l'as ouverte ? demanda-t-il. "
          "— Non, répondit Lina. Elle est pour toi. Elle recula pour le laisser passer. ")


def job(modes=("rp", "narration"), text=SOURCE):
    return create_job(text, "livre.txt", "La lettre", modes, 800, "qwen-test")


def rp():
    return json.dumps({"character": "Lina",
                       "messages": [{"role": "user", "content": "— Tu l'as ouverte ? demanda-t-il."},
                                    {"role": "assistant", "content": "— Non, répondit Lina. Elle est pour toi."}]})


def generated(system, prompt, **kwargs):
    return json.dumps({"prompt": "Raconte la remise d'une lettre dans une bibliothèque sous la pluie."}) \
        if "Ne génère pas la réponse" in prompt else rp()


class TestBookGeneration(unittest.TestCase):
    def test_ebook_typography_is_matched_but_only_original_characters_are_exported(self):
        text = SOURCE.replace("Tu l'as ouverte", "Tu\u00a0l’as   ouverte").replace(
            "— Non, répondit Lina.", "– Non,\u202frépondit Lina.")
        current = job(("rp",), text)
        run_job(current, Mock(return_value=rp()))
        rows, _ = job_rows(current)
        self.assertFalse(current["errors"])
        self.assertIn("Tu\u00a0l’as   ouverte", rows[0].data["messages"][1]["content"])
        self.assertEqual(rows[0].data["messages"][2]["content"], "– Non,\u202frépondit Lina. Elle est pour toi.")
        restored = job(("rp",), text)
        restore_checkpoint(restored, checkpoint_bytes(current))
        self.assertEqual(restored["results"], current["results"])

    def test_typography_tolerance_does_not_accept_rewriting_or_reordered_quotes(self):
        current = job(("rp",))
        for wrong in (rp().replace("Non,", "Jamais,"), rp().replace("ouverte", "fermée")):
            with self.assertRaises(QwenError):
                make_record(current, 0, "rp", wrong)
        value = json.loads(rp())
        value["messages"].reverse()
        value["messages"][0]["role"], value["messages"][1]["role"] = "user", "assistant"
        with self.assertRaises(QwenError):
            make_record(current, 0, "rp", json.dumps(value))

    def test_flattened_contents_rights_and_playlists_are_skipped_before_inference(self):
        pages = [
            "Sommaire Couverture Titre CHAPITRE UN : LE PORT CHAPITRE DEUX : LA LETTRE "
            "CHAPITRE TROIS : LE RETOUR CHAPITRE QUATRE : LE VOYAGE",
            "© Éditions de démonstration, 2020, pour la présente édition. "
            "ISBN : 9780000000000. Ce document numérique est un exemple de test.",
            "ATMOSPHÈRE First Song – First Artist Second Song – Second Artist "
            "Third Song – Third Artist Fourth Song – Fourth Artist",
        ]
        for page in pages:
            with self.subTest(page=page):
                self.assertIsNotNone(non_narrative_reason(page))
                current = job(text=page)
                generate = Mock(side_effect=AssertionError("editorial page sent to model"))
                run_job(current, generate)
                generate.assert_not_called()
                self.assertEqual(len(current["results"]), 2)
                self.assertFalse(job_rows(current)[0])
                self.assertFalse(current["errors"])
                with self.assertRaises(QwenError):
                    make_record(current, 0, "narration", generated("", "Ne génère pas la réponse"))

    def test_a_chapter_heading_followed_by_prose_is_kept(self):
        text = "CHAPITRE UN : LA LETTRE\n\n" + SOURCE
        self.assertIsNone(non_narrative_reason(text))
        current = job(("narration",), text)
        run_job(current, generated)
        self.assertEqual(job_rows(current)[0][0].data["messages"][-1]["content"], text)

    def test_preview_moves_past_editorial_pages_to_an_actual_scene(self):
        text = "Sommaire " + "CHAPITRE UN : LA LETTRE " * 25 + "\n\n" + SOURCE
        current = job(text=text)
        self.assertEqual(len(current["passages"]), 2)
        generate = Mock(side_effect=generated)
        run_job(current, generate, limit=1)
        generate.assert_called_once()
        self.assertIn("skipped", current["results"]["0:narration"])
        self.assertIn("messages", current["results"]["1:rp"])
        self.assertNotIn("1:narration", current["results"])

    def test_unrelated_and_wrong_language_instructions_are_not_accepted(self):
        for prompt in (
            "Écris une scène de cuisine où une femme prépare un dîner sous une lumière rouge.",
            "Écris en anglais la remise de la lettre entre Lina et Hugo dans la bibliothèque.",
            "Describe the letter with the rain and the library where she was waiting with him.",
        ):
            with self.subTest(prompt=prompt), self.assertRaises(QwenError):
                make_record(job(), 0, "narration", json.dumps({"prompt": prompt}))
        current = job(("narration",))
        run_job(current, Mock(return_value=json.dumps({"prompt":
            "Écris une scène de cuisine où une femme prépare un dîner sous une lumière rouge."})))
        self.assertFalse(current["results"])
        self.assertIn("0:narration", current["errors"])

    def test_old_generator_progress_is_not_reused_as_new_validated_data(self):
        current = job()
        old = deepcopy(current)
        old["parameters"]["version"] = "book-generation-v1"
        old_payload = {"format": "book-generation-v1", "id": current["id"], "results": {}, "errors": {}}
        self.assertNotEqual(current["parameters"]["version"], old["parameters"]["version"])
        with self.assertRaises(ValueError):
            restore_checkpoint(current, json.dumps(old_payload))

    def test_decoding_is_explicit_and_never_silently_drops_accents(self):
        for encoding in ("utf-8-sig", "utf-16", "cp1252"):
            self.assertEqual(decode_book("Éloïse\r\nLina".encode(encoding), encoding), "Éloïse\nLina")
        for blob in (b"\xff", b"123", b"abc\x00def"):
            with self.assertRaises(ValueError):
                decode_book(blob)

    def test_partition_preserves_every_character_even_in_long_paragraphs(self):
        text = SOURCE * 20 + "\n\n" + "x" * 1700 + "FIN"
        passages = split_book(text, 800)
        self.assertEqual("".join(p["text"] for p in passages), text)
        self.assertEqual(passages[-1]["end"], len(text))
        for passage in passages:
            self.assertEqual(text[passage["start"]:passage["end"]], passage["text"])
            self.assertLessEqual(len(passage["text"]), 800)
        with self.assertRaises(ValueError):
            split_book(text, 0)

    def test_narration_answer_is_source_not_teacher_rewriting(self):
        current = job(("narration",))
        run_job(current, generated)
        rows, sources = job_rows(current)
        self.assertEqual(rows[0].data["messages"][-1]["content"], SOURCE)
        self.assertEqual(rows[0].data["book_source"]["assistant_origin"], "original")
        self.assertIn("Needs Review", rows[0].auto_tags)
        self.assertIn("passage 1", sources[1])
        chatml, amount = export_jsonl(rows, chatml=True)
        self.assertEqual(amount, 1)
        self.assertEqual(set(json.loads(chatml)), {"messages"})
        original, _ = export_jsonl(rows)
        self.assertEqual(json.loads(original)["book_source"]["excerpt"], SOURCE)
        restored, errors = load_dataset(io.BytesIO(export_project(rows, "livre", sources)), "livre.rpwork.json")
        self.assertFalse(errors)
        self.assertEqual(restored, rows)

    def test_rp_has_original_context_and_alternating_turns(self):
        record = make_record(job(), 0, "rp", rp())
        self.assertEqual([m["role"] for m in record["messages"]], ["system", "user", "assistant"])
        self.assertEqual(record["book_source"]["assistant_origin"], "original")
        self.assertEqual(record["book_source"]["preparation"], "qwen_extraction")
        self.assertEqual(record["book_source"]["excerpt"], SOURCE)
        self.assertIn("Lina posa la lettre sur la table.", record["messages"][0]["content"])
        self.assertNotIn("Non, répondit Lina", record["messages"][0]["content"])
        with self.assertRaisesRegex(QwenError, "n'apparaît pas"):
            make_record(job(), 0, "rp", rp().replace("Lina", "Fantôme"))

    def test_invalid_refused_and_invented_responses_stay_pending(self):
        values = ["I cannot do that", "[]", "{}", None, '{"prompt":""}',
                  json.dumps({"prompt": "OK", "answer": "invented"})]
        for value in values:
            with self.subTest(value=value):
                current = job(("narration",))
                run_job(current, Mock(return_value=value))
                self.assertFalse(current["results"])
                self.assertIn("0:narration", current["errors"])
        invalid_rp = json.loads(rp())
        invalid_rp["messages"][0]["role"] = "system"
        current = job(("rp",))
        run_job(current, Mock(return_value=json.dumps(invalid_rp)))
        self.assertFalse(current["results"])

    def test_wrapped_json_and_empty_rp_are_handled_explicitly(self):
        current = job(("narration",))
        run_job(current, Mock(return_value='<think>hidden</think>```json\n{"prompt":"Raconte la remise de la lettre dans la bibliothèque."}\n```'))
        self.assertEqual(len(job_rows(current)[0]), 1)
        current = job(("rp",))
        run_job(current, Mock(return_value='{"character":"","messages":[]}'))
        self.assertIn("skipped", current["results"]["0:rp"])
        self.assertFalse(current["errors"])
        self.assertFalse(job_rows(current)[0])
        # Resuming skipped tasks must not try to read nonexistent message arrays.
        run_job(current, Mock(side_effect=AssertionError("already done")))

    def test_rewritten_reordered_and_answer_leaking_rp_are_rejected(self):
        original = json.loads(rp())
        rewritten = deepcopy(original)
        rewritten["messages"][1]["content"] = "Lina invente une nouvelle aventure."
        reordered = deepcopy(original)
        reordered["messages"][0]["content"], reordered["messages"][1]["content"] = (
            reordered["messages"][1]["content"], reordered["messages"][0]["content"])
        leaking = deepcopy(original)
        leaking["context"] = original["messages"][1]["content"]
        invented = deepcopy(original)
        invented["context"] = "Lina est la sœur de Hugo."
        for value in (rewritten, reordered, leaking, invented):
            with self.subTest(value=value), self.assertRaises(QwenError):
                make_record(job(), 0, "rp", json.dumps(value))

    def test_short_passages_are_reported_without_calling_the_model(self):
        current = job(text="Lina.")
        generate = Mock(side_effect=AssertionError("short source"))
        run_job(current, generate)
        generate.assert_not_called()
        self.assertEqual(len(current["results"]), 2)
        self.assertTrue(all("skipped" in row for row in current["results"].values()))

    def test_explicit_wrong_speaker_is_rejected_and_one_repair_can_correct_it(self):
        wrong = json.loads(rp())
        wrong["character"] = "Hugo"
        with self.assertRaisesRegex(QwenError, "autre personnage"):
            make_record(job(), 0, "rp", json.dumps(wrong))
        generate = Mock(side_effect=[json.dumps(wrong), rp()])
        current = job(("rp",))
        run_job(current, generate)
        self.assertEqual(generate.call_count, 2)
        self.assertIn("validation_error", generate.call_args.args[1])
        self.assertFalse(current["errors"])
        self.assertIn("Tu incarnes Lina.", job_rows(current)[0][0].data["messages"][0]["content"])

    def test_successes_are_saved_before_interruption_then_resumed_once(self):
        current = job()
        save = Mock()
        with self.assertRaises(KeyboardInterrupt):
            run_job(current, generated, save=save, progress=Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(set(current["results"]), {"0:rp"})
        save.assert_called_once()
        generate = Mock(side_effect=generated)
        restored = job()
        restore_checkpoint(restored, checkpoint_bytes(current))
        run_job(restored, generate)
        generate.assert_called_once()
        self.assertEqual(len(job_rows(restored)[0]), 2)
        self.assertFalse(restored["errors"])

    def test_runtime_failure_stops_and_retry_clears_error_without_regenerating_successes(self):
        current = job()
        generate = Mock(side_effect=[rp(), QwenUnavailable("disconnected")])
        run_job(current, generate)
        self.assertEqual(set(current["results"]), {"0:rp"})
        self.assertIn("0:narration", current["errors"])
        retry = Mock(side_effect=generated)
        run_job(current, retry)
        retry.assert_called_once()
        self.assertFalse(current["errors"])
        self.assertEqual(len(job_rows(current)[0]), 2)

    def test_duplicate_results_are_counted_and_resume_safely(self):
        current = job(("rp",), (SOURCE * 2 + "\n\n") * 3)
        run_job(current, generated)
        self.assertEqual(len(current["results"]), len(tasks(current)))
        self.assertEqual(len(job_rows(current)[0]), 1)
        self.assertTrue(any("skipped" in item for item in current["results"].values()))
        run_job(current, generated)

    def test_checkpoint_mismatch_corruption_and_tampering_do_not_replace_progress(self):
        current = job(("narration",))
        run_job(current, generated)
        payload = json.loads(checkpoint_bytes(current))
        for altered in ({**payload, "id": "wrong"}, {**payload, "results": []},
                        {**payload, "results": {"99:narration": {}}}):
            target = job(("narration",))
            with self.assertRaises(ValueError):
                restore_checkpoint(target, json.dumps(altered))
            self.assertFalse(target["results"])
        payload["results"]["0:narration"]["messages"][1]["content"] = "invented answer"
        with self.assertRaises(ValueError):
            restore_checkpoint(job(("narration",)), json.dumps(payload))
        rp_job = job(("rp",))
        run_job(rp_job, generated)
        corrupted = json.loads(checkpoint_bytes(rp_job))
        corrupted["results"]["0:rp"]["messages"][0]["content"] = 12
        with self.assertRaises(ValueError):
            restore_checkpoint(job(("rp",)), json.dumps(corrupted))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.bookwork.json"
            save_checkpoint(current, path)
            target = job(("narration",))
            restore_checkpoint(target, path.read_bytes())
            self.assertEqual(target["results"], current["results"])
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_prompt_marks_source_as_data_and_long_generation_has_its_own_schema(self):
        current = job()
        generate = Mock(side_effect=generated)
        run_job(current, generate, limit=1)
        system, prompt = generate.call_args.args
        self.assertIn("untrusted DATA", system)
        self.assertIn(SOURCE, prompt)
        self.assertEqual(generate.call_args.kwargs["max_tokens"], 2048)
        self.assertIn("character", generate.call_args.kwargs["schema"]["properties"])


class TestBookRuntime(unittest.TestCase):
    def test_book_schema_and_output_budget_are_sent_to_both_backends(self):
        for backend in ("gguf", "comfy"):
            client = LocalQwen(QwenConfig(backend=backend))
            result = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
            with patch.object(client, "load"), patch.object(client, "_json_http", return_value=result) as http, \
                    patch.object(client, "_request", return_value={"text": "{}", "truncated": False}) as request:
                self.assertEqual(generate_json(client, "s", "p", schema={"type": "object"}, max_tokens=2048), "{}")
                if backend == "gguf":
                    self.assertEqual(http.call_args.args[1]["max_tokens"], 2048)
                    self.assertEqual(http.call_args.args[1]["response_format"]["json_schema"]["schema"], {"type": "object"})
                else:
                    self.assertEqual(request.call_args.args[0]["op"], "generate_json")
                    self.assertEqual(request.call_args.args[0]["max_tokens"], 2048)

    def test_truncated_book_output_is_rejected_without_partial_entries(self):
        client = LocalQwen(QwenConfig(backend="comfy"))
        with patch.object(client, "load"), patch.object(client, "close") as close, \
                patch.object(client, "_request", return_value={"text": "{}", "truncated": True}):
            with self.assertRaises(QwenUnavailable):
                generate_json(client, "s", "p", schema={})
            close.assert_called_once()


class TestBookUI(unittest.TestCase):
    def open_book(self, directory):
        app = AppTest.from_file(str(APP), default_timeout=30)
        app.session_state.tagging_engine = QWEN_2B
        app.session_state.rows = [Entry(1, {"text": "Existing dataset"}, 1)]
        app.session_state.dataset_name = "existing.jsonl"
        app.run()
        next(r for r in app.radio if r.label == "Navigation").set_value("📖 Depuis un livre").run()
        path = Path(directory) / "livre.txt"
        path.write_text(SOURCE, encoding="utf-8")
        next(t for t in app.text_input if t.label == "Ou chemin local du livre TXT").set_value(str(path)).run()
        return app, path

    def click(self, app, label):
        next(b for b in app.button if b.label == label).click().run()
        self.assertFalse(app.exception)

    def test_preview_exports_and_opening_preserve_source_and_do_not_run_tagging(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(book_ui, "BASE", Path(directory)):
            app, path = self.open_book(directory)
            self.click(app, "Préparer le livre / reprendre la sauvegarde")
            with patch("book_ui.generate_book", side_effect=generated) as generate:
                self.click(app, "Générer un aperçu")
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(app.session_state.dataset_name, "existing.jsonl")
            self.assertEqual(path.read_text(encoding="utf-8"), SOURCE)
            self.assertTrue(Path(app.session_state.book_checkpoint).is_file())
            self.assertTrue(any("JSONL Unsloth" in item.label for item in app.get("download_button")))
            with patch("qwen_ui.annotate_rows", side_effect=AssertionError("unexpected tagging")):
                self.click(app, "Ouvrir les exemples dans l'éditeur")
            self.assertEqual(len(app.session_state.rows), 2)
            self.assertEqual(app.session_state.navigation_page, "🔎 Explorer")
            self.assertTrue(app.session_state.entry_sources)
            self.assertIn("book_source", app.session_state.rows[0].data)

    def test_settings_changes_disable_generation_and_saved_progress_is_restored(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(book_ui, "BASE", Path(directory)):
            app, _ = self.open_book(directory)
            self.click(app, "Préparer le livre / reprendre la sauvegarde")
            with patch("book_ui.generate_book", side_effect=generated):
                self.click(app, "Générer un aperçu")
            result = deepcopy(app.session_state.book_job["results"])
            app.session_state.book_job = None
            self.click(app, "Préparer le livre / reprendre la sauvegarde")
            self.assertEqual(app.session_state.book_job["results"], result)
            next(t for t in app.text_input if t.label == "Titre du livre (facultatif)").set_value("Autre titre").run()
            self.assertTrue(next(b for b in app.button if b.label == "Générer tout le livre / reprendre").disabled)
            self.assertTrue(any("réglages ont changé" in item.value for item in app.info))

    def test_rules_mode_cannot_generate_a_book_and_page_works_without_dataset(self):
        app = AppTest.from_file(str(APP), default_timeout=30)
        app.session_state.tagging_engine = RULES
        app.run()
        next(r for r in app.radio if r.label == "Navigation").set_value("📖 Depuis un livre").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("nécessite Qwen" in item.value for item in app.info))
        self.assertTrue(next(b for b in app.button if b.label.startswith("Préparer le livre")).disabled)

    def test_switching_to_4b_creates_separate_progress_without_overwriting_2b(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(book_ui, "BASE", Path(directory)):
            app, _ = self.open_book(directory)
            self.click(app, "Préparer le livre / reprendre la sauvegarde")
            with patch("book_ui.generate_book", side_effect=generated):
                self.click(app, "Générer un aperçu")
            old_path = Path(app.session_state.book_checkpoint)
            original = old_path.read_bytes()
            next(s for s in app.selectbox if s.label == "Moteur de tags").set_value(QWEN_4B).run()
            self.assertFalse(app.exception)
            self.assertTrue(next(b for b in app.button if b.label == "Générer tout le livre / reprendre").disabled)
            self.click(app, "Préparer le livre / reprendre la sauvegarde")
            self.assertNotEqual(Path(app.session_state.book_checkpoint), old_path)
            self.assertEqual(old_path.read_bytes(), original)
            self.assertIn(QWEN_4B, app.session_state.book_job["parameters"]["model"])
            self.assertFalse(app.session_state.book_job["results"])

    def test_a_live_old_preparation_is_blocked_until_the_book_is_prepared_again(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(book_ui, "BASE", Path(directory)):
            app, _ = self.open_book(directory)
            self.click(app, "Préparer le livre / reprendre la sauvegarde")
            app.session_state.book_job["parameters"]["version"] = "book-generation-v1"
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(next(b for b in app.button if b.label == "Générer un aperçu").disabled)
            self.assertTrue(any("ancien générateur" in item.value for item in app.warning))
            self.click(app, "Préparer le livre / reprendre la sauvegarde")
            self.assertEqual(app.session_state.book_job["parameters"]["version"], VERSION)


if __name__ == "__main__":
    unittest.main()
