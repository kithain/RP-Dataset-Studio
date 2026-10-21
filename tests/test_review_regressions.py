"""Regression coverage for the six findings in COMPTE_RENDU_REVIEW.md."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from core import Entry, as_messages, export_jsonl, export_project, load_dataset, preview
from conversation_conversion import analyze_conversions, convert_conversation, partial_conversion
from merge_datasets import analyze_merge, prepare_merge


APP = Path(__file__).resolve().parents[1] / "app.py"
MESSAGES = [{"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"}]


def json_stream(payload):
    return io.BytesIO(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


class TestConversionRegressions(unittest.TestCase):
    def test_message_attachments_are_flagged_and_excluded_by_default_merge(self):
        for sequence_key in ("messages", "conversations"):
            for key in ("image", "images", "audio", "video", "attachments", "tool_calls", "function_call"):
                with self.subTest(sequence=sequence_key, field=key):
                    messages = deepcopy(MESSAGES)
                    messages[0][key] = ["asset.png"]
                    data = {sequence_key: messages}
                    before = deepcopy(data)
                    self.assertTrue(partial_conversion(data, as_messages(data)))
                    self.assertIsNone(convert_conversation(data).data)
                    candidates = analyze_conversions([Entry(1, data, 1)])
                    self.assertEqual(len(candidates), 1)
                    self.assertIsNone(candidates[0].result.data)
                    analysis = analyze_merge([("assets.jsonl", json_stream(data))])
                    self.assertEqual(analysis.partial, {1})
                    self.assertFalse(analysis.recoveries)
                    selected = prepare_merge(analysis)
                    self.assertEqual((selected.excluded, selected.active), (1, 0))
                    self.assertTrue(selected.rows[0].deleted)
                    self.assertEqual(export_jsonl(selected.rows, chatml=True), (b"", 0))
                    self.assertEqual(data, before)

    def test_empty_attachment_fields_remain_text_compatible(self):
        data = {"messages": [{**MESSAGES[0], "images": [], "attachments": None}, MESSAGES[1]]}
        self.assertFalse(partial_conversion(data, as_messages(data)))
        self.assertIsNotNone(convert_conversation(data).data)

    def test_instruction_response_context_survives_conversion_merge_and_chatml_export(self):
        for context in ("Bonjour", "  العربية\n日本語  ", "", "   "):
            with self.subTest(context=context):
                data = {"instruction": "Translate the input", "input": context,
                        "response": "Hello", "system": "Translator", "metadata": {"keep": True}}
                before = deepcopy(data)
                result = convert_conversation(data)
                self.assertIsNotNone(result.data)
                blob, amount = export_jsonl([Entry(1, result.data, 1)], chatml=True)
                self.assertEqual(amount, 1)
                expected = "Translate the input" + ("\n\n" + context if context.strip() else "")
                self.assertEqual(json.loads(blob)["messages"][1]["content"], expected)
                self.assertEqual({key: result.data[key] for key in data}, before)
                analysis = analyze_merge([("input.jsonl", json_stream(data))])
                selected = prepare_merge(analysis)
                self.assertEqual((selected.active, selected.converted), (1, 1))
                self.assertEqual(export_jsonl(selected.rows, chatml=True), (blob, 1))
                self.assertEqual(data, before)

    def test_instruction_response_ambiguous_context_requires_manual_correction(self):
        for context in (None, 0, False, [], ["Bonjour"], {"text": "Bonjour"}):
            with self.subTest(context=context):
                data = {"instruction": "Translate", "input": context, "response": "Hello"}
                result = convert_conversation(data)
                self.assertIsNone(result.data)
                self.assertIn("input", result.reason)
                analysis = analyze_merge([("input.jsonl", json_stream(data))])
                self.assertEqual(prepare_merge(analysis).excluded, 1)


class TestProjectValidation(unittest.TestCase):
    def setUp(self):
        self.entry = {"uid": 1, "data": {"messages": MESSAGES}, "source_line": 7,
                      "deleted": False, "language_override": "fr"}
        self.payload = {"format": "rp-dataset-studio-v1", "entries": [self.entry],
                        "entry_sources": {"1": "original.jsonl"}}

    def test_invalid_project_structures_are_rejected_without_mutating_origins(self):
        invalid = [None, [], "project", 42, {},
                   {**self.payload, "entries": None}, {**self.payload, "entries": {}},
                   {**self.payload, "entries": "entries"}]
        for item in (None, [], "entry", {}, {"uid": 2, "data": None},
                     {"uid": 2, "data": []}, {"uid": 2, "data": "text"}):
            invalid.append({**self.payload, "entries": [self.entry, item]})
        for key, values in {"uid": (None, True, 1.5, "1", 0, -1),
                            "source_line": (None, True, 1.5, "7", -1),
                            "deleted": (None, "false", 0, 1, []),
                            "language_override": ("invalid", "", 42, [], {})}.items():
            for value in values:
                invalid.append({**self.payload, "entries": [{**self.entry, key: value}]})
        invalid.append({**self.payload, "entries": [self.entry, deepcopy(self.entry)]})
        for origins in (None, [], "sources", {"oops": "a.jsonl"}, {"2": "a.jsonl"},
                        {"01": "a.jsonl"}, {"1": None}, {"1": []}):
            invalid.append({**self.payload, "entry_sources": origins})
        for payload in invalid:
            with self.subTest(payload=payload):
                origins = {99: "previous.jsonl"}
                with self.assertRaises(ValueError):
                    load_dataset(json_stream(payload), "bad.rpwork.json", entry_sources=origins)
                self.assertEqual(origins, {99: "previous.jsonl"})

    def test_old_projects_without_origins_and_optional_fields_still_load(self):
        payload = {"format": "rp-dataset-studio-v1", "entries": [{"uid": 5, "data": {"text": "old"}}]}
        origins = {99: "previous.jsonl"}
        rows, errors = load_dataset(json_stream(payload), "old.rpwork.json", entry_sources=origins)
        self.assertFalse(errors)
        self.assertEqual(rows, [Entry(5, {"text": "old"}, 0)])
        self.assertEqual(origins, {})

    def test_merge_save_load_and_merge_again_preserve_original_sources_and_lines(self):
        sources = [("first.jsonl", io.BytesIO(b'\n{"prompt":"Q1","completion":"A1"}\n')),
                   ("second.jsonl", io.BytesIO(b'{"prompt":"Q2","completion":"A2"}\n'
                                              b'\n{"text":"incompatible"}\n'))]
        analysis = analyze_merge(sources)
        selected = prepare_merge(analysis, seed=3)
        blob = export_project(selected.rows, "merged.jsonl", analysis.origins)
        origins = {}
        rows, errors = load_dataset(io.BytesIO(blob), "merged.rpwork.json", entry_sources=origins)
        self.assertFalse(errors)
        self.assertEqual(rows, selected.rows)
        self.assertEqual(origins, {1: "first.jsonl", 2: "second.jsonl", 3: "second.jsonl"})
        self.assertEqual({row.uid: row.source_line for row in rows}, {1: 2, 2: 1, 3: 3})
        merged_again = analyze_merge([("saved.rpwork.json", io.BytesIO(blob))])
        for row in merged_again.rows:
            previous = next(r for r in rows if r.data == row.data)
            self.assertEqual(merged_again.origins[row.uid], origins[previous.uid])
            self.assertEqual(row.source_line, previous.source_line)


class TestReviewUI(unittest.TestCase):
    def open_rows(self, rows, page="🔎 Explorer"):
        app = AppTest.from_file(str(APP), default_timeout=60)
        app.session_state.rows = rows
        app.session_state.dataset_name = "current.jsonl"
        app.session_state.navigation_page = page
        app.run()
        self.assertFalse(app.exception)
        return app

    def grid_key(self, app):
        return next(key for key in app.session_state.keys() if key.startswith("explorer_grid_"))

    def select_second_row(self, app):
        app.session_state[self.grid_key(app)] = {
            "edited_rows": {1: {"Cocher": True}}, "added_rows": [], "deleted_rows": []}
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(next(b for b in app.button if "Supprimer 1 cochée" in b.label).disabled)

    def test_filter_change_clears_selection_before_delete_or_language_assignment(self):
        rows = [Entry(uid, {"text": text}, uid) for uid, text in (
            (1, "alpha beta"), (2, "alpha"), (3, "beta"), (4, "alpha beta"))]
        app = self.open_rows(rows)
        include = "Mots à conserver (séparés par ; = OU)"
        next(t for t in app.text_input if t.label == include).set_value("alpha")
        next(b for b in app.button if b.label == "Appliquer les filtres").click().run()
        self.select_second_row(app)
        old_key = self.grid_key(app)
        next(t for t in app.text_input if t.label == include).set_value("beta")
        next(b for b in app.button if b.label == "Appliquer les filtres").click().run()
        self.assertFalse(app.exception)
        self.assertNotEqual(self.grid_key(app), old_key)
        self.assertTrue(next(b for b in app.button if "Supprimer 0 cochée" in b.label).disabled)
        self.assertTrue(next(b for b in app.button if b.label == "Attribuer cette langue").disabled)
        self.assertFalse(any(row.deleted or row.language_override for row in app.session_state.rows))
        self.select_second_row(app)
        next(b for b in app.button if "Supprimer 1 cochée" in b.label).click().run()
        self.assertFalse(app.exception)
        self.assertEqual([row.uid for row in app.session_state.rows if row.deleted], [3])

    def test_reordering_intermediate_rows_clears_selection(self):
        rows = [Entry(uid, {"text": str(uid)}, uid) for uid in range(1, 5)]
        app = self.open_rows(rows)
        self.select_second_row(app)
        old_key = self.grid_key(app)
        app.session_state.rows = [rows[0], rows[2], rows[1], rows[3]]
        app.run()
        self.assertFalse(app.exception)
        self.assertNotEqual(self.grid_key(app), old_key)
        self.assertTrue(next(b for b in app.button if "Supprimer 0 cochée" in b.label).disabled)

    def test_invalid_import_keeps_current_dataset_history_and_prepared_export(self):
        rows = [Entry(9, {"text": "keep"}, 8)]
        app = self.open_rows(rows)
        app.session_state.entry_sources = {9: "keep.jsonl"}
        app.session_state.history = [{"kind": "lang", "before": [(9, None)]}]
        app.session_state.export_project_data = b"prepared"
        revision = app.session_state.revision
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.rpwork.json"
            for entry in ({"uid": 1, "data": None},
                          {"uid": 1, "data": {}, "language_override": "invalid"},
                          {"uid": 1, "data": {}, "deleted": "false"}):
                with self.subTest(entry=entry):
                    path.write_text(json.dumps({"format": "rp-dataset-studio-v1", "entries": [entry]}),
                                    encoding="utf-8")
                    next(t for t in app.text_input if t.label == "Chemin du fichier sur ce PC").set_value(str(path)).run()
                    next(b for b in app.button if b.label == "Ouvrir le chemin local").click().run()
                    self.assertFalse(app.exception)
                    self.assertIn("Impossible d'ouvrir le fichier", app.error[0].value)
                    self.assertEqual(app.session_state.rows, rows)
                    self.assertEqual(app.session_state.dataset_name, "current.jsonl")
                    self.assertEqual(app.session_state.entry_sources, {9: "keep.jsonl"})
                    self.assertEqual(len(app.session_state.history), 1)
                    self.assertEqual(app.session_state.export_project_data, b"prepared")
                    self.assertEqual(app.session_state.revision, revision)

    def test_project_ui_save_and_import_restore_origins_in_inspector(self):
        rows = [Entry(2, {"prompt": "Q", "completion": "A"}, 17),
                Entry(5, {"text": "deleted"}, 23, deleted=True)]
        app = self.open_rows(rows, "📦 Exporter")
        app.session_state.entry_sources = {2: "first.jsonl", 5: "second.jsonl"}
        next(b for b in app.button if b.label == "Préparer la sauvegarde du projet").click().run()
        self.assertFalse(app.exception)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.rpwork.json"
            path.write_bytes(app.session_state.export_project_data)
            app.session_state.entry_sources = {99: "stale.jsonl"}
            next(t for t in app.text_input if t.label == "Chemin du fichier sur ce PC").set_value(str(path)).run()
            next(b for b in app.button if b.label == "Ouvrir le chemin local").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.rows, rows)
            self.assertEqual(app.session_state.entry_sources, {2: "first.jsonl", 5: "second.jsonl"})
            next(r for r in app.radio if r.label == "Navigation").set_value("🔎 Explorer").run()
            self.assertTrue(any(c.value == "Fichier source : first.jsonl" for c in app.caption))
            self.assertTrue(any("ligne 17" in c.value for c in app.caption))
            next(t for t in app.text_input if t.label == "Chemin du fichier sur ce PC").set_value(str(APP.parent / "examples" / "exemple_rp.jsonl")).run()
            next(b for b in app.button if b.label == "Ouvrir le chemin local").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.entry_sources, {})

    def test_duplicate_display_prepares_only_first_100_previews_and_deletes_all_duplicates(self):
        rows = [Entry(uid, {"messages": deepcopy(MESSAGES)}, uid) for uid in range(1, 152)]
        with patch("core.preview", wraps=preview) as previews:
            app = self.open_rows(rows, "🧹 Qualité")
        self.assertEqual(previews.call_count, 100)
        displayed = next(frame.value for frame in app.dataframe if "Conversation" in frame.value.columns)
        self.assertEqual(displayed["ID"].tolist(), list(range(2, 102)))
        next(b for b in app.button if "150 doublons dans la corbeille" in b.label).click().run()
        self.assertFalse(app.exception)
        self.assertEqual([row.uid for row in app.session_state.rows if not row.deleted], [1])


if __name__ == "__main__":
    unittest.main()
