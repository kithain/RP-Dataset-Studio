from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from core import Entry
from llm_tagging import (LocalQwen, QwenConfig, QwenError, QwenUnavailable, analyze_record,
                         parse_result, record_text, tag_with_qwen, text_chunks)
from qwen_ui import QWEN_2B, QWEN_4B, RULES, clients


APP = Path(__file__).resolve().parents[1] / "app.py"


def response(rating="SFW", tags=None):
    return json.dumps({"rating": rating, "tags": ["Romance"] if tags is None else tags})


class TestSemanticTags(unittest.TestCase):
    def test_uses_model_result_for_implicit_genres_without_regex_fallback(self):
        generate = Mock(return_value=response(tags=["romance", "slice of life"]))
        data = {"text": "They blush, exchange flowers and hold hands."}
        self.assertEqual(analyze_record(data, generate), ["SFW", "Romance", "Slice of Life"])
        system, prompt = generate.call_args.args
        self.assertIn("untrusted DATA", system)
        self.assertIn(data["text"], prompt)

    def test_fenced_and_thinking_wrapped_json_are_supported(self):
        self.assertEqual(parse_result('```json\n' + response() + '\n```'), ("SFW", ["Romance"]))
        self.assertEqual(parse_result('<think>private analysis</think>' + response()), ("SFW", ["Romance"]))

    def test_refusals_truncation_and_invalid_structures_are_rejected(self):
        for value in ("I cannot classify this.", '{"rating":"SFW"', "[]", "null", "{}",
                      '{"rating":"bad","tags":[]}', '{"rating":"SFW","tags":"Romance"}',
                      response(tags=[1]), response(tags=[""]), response(tags=["long\ntext"]),
                      response(tags=["x" * 101]), response(tags=[str(i) for i in range(25)])):
            with self.subTest(value=value), self.assertRaises(QwenError):
                parse_result(value)

    def test_all_text_is_processed_and_nsfw_anywhere_takes_precedence(self):
        text = "A" * 13000 + "END_SIGNAL"
        calls = []
        def generate(system, prompt):
            calls.append(prompt)
            return response("NSFW", ["Vore"]) if "END_SIGNAL" in prompt else response("SFW", ["Comedy"])
        self.assertEqual(analyze_record({"text": text}, generate), ["NSFW", "Comedy", "Vore"])
        self.assertEqual(len(calls), 3)
        self.assertIn("END_SIGNAL", calls[-1])

    def test_chunk_overlap_has_no_gaps_or_missing_end(self):
        original = "0123456789" * 1300 + "END"
        chunks = list(text_chunks(original, size=1000, overlap=100))
        restored = chunks[0] + "".join(chunk[100:] for chunk in chunks[1:])
        self.assertEqual(restored, original)
        for size, overlap in ((0, 0), (100, 100), (100, -1)):
            with self.assertRaises(ValueError):
                list(text_chunks(original, size=size, overlap=overlap))

    def test_uncertainty_is_not_reported_as_sfw(self):
        self.assertEqual(analyze_record({"text": "Ambiguous"}, Mock(return_value=response("Uncertain", []))),
                         ["Needs Review"])
        generate = Mock(side_effect=[response("SFW", ["SFW"]), response("Uncertain", ["NSFW"])])
        self.assertEqual(analyze_record({"text": "x" * 7000}, generate), ["Needs Review"])

    def test_unrecognized_empty_and_numeric_records_do_not_call_model(self):
        generate = Mock()
        for data in ({}, {"id": "Romance"}, {"text": "123"}, {"text": ""}, {"unknown": "NSFW"}):
            self.assertEqual(analyze_record(data, generate), [])
        generate.assert_not_called()

    def test_card_fields_are_preserved_but_images_tools_and_old_tags_are_not_sent(self):
        data = {"data": {"name": "Lina", "personality": "shy", "tags": ["OLD_TAG"]},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Hello"}, {"type": "image_url", "image_url": "IMAGE_PAYLOAD"}]},
                    {"role": "tool", "content": "TOOL_PAYLOAD"}]}
        text = record_text(data)
        for expected in ("Lina", "shy", "Hello"):
            self.assertIn(expected, text)
        for ignored in ("OLD_TAG", "IMAGE_PAYLOAD", "TOOL_PAYLOAD"):
            self.assertNotIn(ignored, text)

    def test_partial_failure_keeps_previous_tags_manual_overrides_and_deleted_rows(self):
        rows = [Entry(1, {"text": "First"}, 1, tags_override=[]),
                Entry(2, {"text": "Second"}, 2, auto_tags=["Old"], tags_override=["Manual"]),
                Entry(3, {"text": "Deleted"}, 3, deleted=True)]
        original = deepcopy(rows)
        generate = Mock(side_effect=[response(), "Not JSON"])
        progress = Mock()
        before, errors = tag_with_qwen(rows, generate, progress)
        self.assertEqual(before, [(1, [])])
        self.assertEqual(rows[0].auto_tags, ["SFW", "Romance"])
        self.assertEqual(rows[0].tags_override, [])
        self.assertEqual(rows[1:], original[1:])
        self.assertEqual([uid for uid, _ in errors], [2])
        self.assertEqual(progress.call_count, 2)
        self.assertEqual([row.data for row in rows], [row.data for row in original])

    def test_infrastructure_failure_stops_the_batch_without_touching_remaining_rows(self):
        rows = [Entry(i, {"text": f"Entry {i}"}, i, auto_tags=["Previous"]) for i in range(3)]
        original = deepcopy(rows)
        generate = Mock(side_effect=QwenUnavailable("Model unavailable"))
        before, errors = tag_with_qwen(rows, generate)
        self.assertEqual(before, [])
        self.assertEqual(len(errors), 1)
        self.assertEqual(rows, original)
        self.assertEqual(generate.call_count, 1)

    def test_interrupted_progress_does_not_leave_untracked_changes(self):
        rows = [Entry(1, {"text": "First"}, 1, auto_tags=["Original"])]
        original = deepcopy(rows)
        with self.assertRaises(KeyboardInterrupt):
            tag_with_qwen(rows, Mock(return_value=response()), Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(rows, original)


class TestLocalRuntime(unittest.TestCase):
    def test_missing_paths_and_incompatible_formats_are_reported_before_launch(self):
        with patch("llm_tagging.subprocess.Popen") as popen:
            client = LocalQwen(QwenConfig(model="missing-model.gguf"))
            with self.assertRaises(QwenUnavailable):
                client.load()
            popen.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "weights.safetensors"
            server = Path(directory) / "llama-server.exe"
            model.touch()
            server.touch()
            with self.assertRaises(QwenError):
                QwenConfig(model=str(model), server=str(server)).validate()

    def test_http_request_uses_json_schema_and_disables_thinking(self):
        client = LocalQwen(QwenConfig())
        result = {"choices": [{"finish_reason": "stop", "message": {"content": response()}}]}
        with patch.object(client, "load"), patch.object(client, "_json_http", return_value=result) as http:
            self.assertEqual(client.generate("system", "text"), response())
            route, body = http.call_args.args
            self.assertEqual(route, "/v1/chat/completions")
            self.assertEqual(body["response_format"]["type"], "json_schema")
            self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(body["messages"][0]["content"], "system")

    def test_truncated_response_and_timeout_close_the_child(self):
        client = LocalQwen(QwenConfig(), timeout=.01)
        result = {"choices": [{"finish_reason": "length", "message": {"content": response()}}]}
        with patch.object(client, "load"), patch.object(client, "_json_http", return_value=result), \
                patch.object(client, "close") as close:
            with self.assertRaises(QwenUnavailable):
                client.generate("system", "text")
            close.assert_called_once()
        client.process = Mock()
        with self.assertRaisesRegex(QwenError, "Délai dépassé"):
            client._request({"op": "generate"})
        client.process = None

    def test_unload_terminates_only_the_owned_child(self):
        client = LocalQwen(QwenConfig())
        child = Mock()
        child.poll.return_value = None
        client.process = child
        client.close()
        child.terminate.assert_called_once()
        child.wait.assert_called_once()
        self.assertIsNone(client.process)

    def test_server_is_bound_to_loopback_and_cannot_silently_shift_context(self):
        with tempfile.TemporaryDirectory() as directory:
            model, server = Path(directory) / "model.gguf", Path(directory) / "llama-server.exe"
            model.touch()
            server.touch()
            client = LocalQwen(QwenConfig(model=str(model), server=str(server)))
            child = Mock()
            child.poll.return_value = None
            with patch("llm_tagging.subprocess.Popen", return_value=child) as popen, \
                    patch.object(client, "_json_http", return_value={"status": "ok"}):
                client.load()
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
                self.assertIn("--no-context-shift", command)
                self.assertFalse(popen.call_args.kwargs.get("shell", False))
            client.close()


class TestQwenUI(unittest.TestCase):
    def setUp(self):
        clients.clear()

    def tearDown(self):
        clients().unload()
        clients.clear()

    def open_app(self, rows=None):
        app = AppTest.from_file(str(APP), default_timeout=30)
        app.session_state.tagging_engine = QWEN_2B
        if rows is not None:
            app.session_state.rows = rows
            app.session_state.dataset_name = "test.jsonl"
        app.run()
        self.assertFalse(app.exception)
        return app

    def click(self, app, label):
        next(button for button in app.button if button.label == label).click().run()
        self.assertFalse(app.exception)

    def test_import_uses_qwen_and_keeps_dataset_on_classification_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text('{"text":"They hold hands"}\n', encoding="utf-8")
            app = self.open_app()
            next(t for t in app.text_input if t.label == "Chemin du fichier sur ce PC").set_value(str(path)).run()
            with patch("qwen_ui.LocalQwen.generate", return_value=response()) as generate:
                self.click(app, "Ouvrir le chemin local")
            generate.assert_called_once()
            self.assertEqual(app.session_state.rows[0].auto_tags, ["SFW", "Romance"])
            with patch("qwen_ui.LocalQwen.generate", side_effect=QwenUnavailable("Load failed")):
                self.click(app, "Ouvrir le chemin local")
            self.assertEqual(len(app.session_state.rows), 1)
            self.assertEqual(app.session_state.rows[0].auto_tags, [])
            self.assertTrue(any("Aucun remplacement automatique" in item.value for item in app.warning))

    def test_batch_uses_qwen_is_undoable_and_preserves_manual_corrections(self):
        rows = [Entry(1, {"text": "They hold hands"}, 1, auto_tags=["Old"], tags_override=["Custom"])]
        app = self.open_app(rows)
        next(r for r in app.radio if r.label == "Navigation").set_value("🧹 Qualité").run()
        with patch("qwen_ui.LocalQwen.generate", return_value=response()):
            self.click(app, "Détecter / actualiser les tags")
        self.assertEqual(app.session_state.rows[0].auto_tags, ["SFW", "Romance"])
        self.assertEqual(app.session_state.rows[0].tags_override, ["Custom"])
        self.click(app, "↶ Annuler la dernière action")
        self.assertEqual(app.session_state.rows[0].auto_tags, ["Old"])

    def test_failed_single_entry_analysis_keeps_override_and_history(self):
        row = Entry(1, {"text": "A text"}, 1, auto_tags=["Old"], tags_override=["Manual"])
        app = self.open_app([row])
        with patch("qwen_ui.LocalQwen.generate", side_effect=QwenError("Invalid JSON")):
            self.click(app, "Utiliser les tags automatiques")
        self.assertEqual(app.session_state.rows[0].tags_override, ["Manual"])
        self.assertEqual(app.session_state.rows[0].auto_tags, ["Old"])
        self.assertFalse(app.session_state.history)
        self.assertTrue(any("Tags conservés" in item.value for item in app.error))

    def test_both_models_and_explicit_rules_mode_are_selectable(self):
        app = self.open_app()
        next(s for s in app.selectbox if s.label == "Moteur de tags").set_value(QWEN_4B).run()
        self.assertTrue(any(t.label == "Modèle Qwen FP8 / safetensors" for t in app.text_input))
        with patch("qwen_ui.LocalQwen.load") as load:
            self.click(app, "Charger / vérifier Qwen")
        load.assert_called_once()
        self.assertEqual(clients().client.config.backend, "comfy")
        self.click(app, "Décharger Qwen de la mémoire")
        self.assertIsNone(clients().client)
        next(s for s in app.selectbox if s.label == "Moteur de tags").set_value(RULES).run()
        self.assertFalse(any(b.label == "Charger / vérifier Qwen" for b in app.button))


if __name__ == "__main__":
    unittest.main()
