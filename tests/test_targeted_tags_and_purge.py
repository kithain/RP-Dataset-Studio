from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from core import Entry, export_jsonl, export_project, load_dataset, purge_deleted
from llm_tagging import QwenUnavailable
from qwen_ui import QWEN_2B, RULES
from tag_management import untagged_entries


APP = Path(__file__).resolve().parents[1] / "app.py"
RESPONSE = json.dumps({"rating": "SFW", "tags": ["Romance"]})


class TestTargetingAndPurge(unittest.TestCase):
    def test_eligible_rows_have_no_effective_tags_or_manual_correction(self):
        rows = [Entry(1, {"text": "New"}, 1),
                Entry(2, {"tags": ["Source"]}, 2),
                Entry(3, {}, 3, auto_tags=["Auto"]),
                Entry(4, {}, 4, tags_override=[]),
                Entry(5, {}, 5, tags_override=["Manual"]),
                Entry(6, {}, 6, deleted=True)]
        before = deepcopy(rows)
        self.assertEqual([row.uid for row in untagged_entries(rows)], [1])
        self.assertEqual(rows, before)

    def test_purge_keeps_survivors_ids_order_and_project_roundtrip(self):
        rows = [Entry(7, {"text": "Keep"}, 2, auto_tags=["SFW"]),
                Entry(42, {"text": "Discard"}, 6, deleted=True),
                Entry(108, {"text": "Keep too"}, 12, tags_override=[])]
        alias = rows
        expected = deepcopy([rows[0], rows[2]])
        cleaned, amount = export_jsonl(rows)
        self.assertEqual(purge_deleted(rows), {42})
        self.assertIs(rows, alias)
        self.assertEqual(rows, expected)
        self.assertEqual(export_jsonl(rows), (cleaned, amount))
        restored, errors = load_dataset(io.BytesIO(export_project(rows, "test")), "test.rpwork.json")
        self.assertEqual(restored, expected)
        self.assertFalse(errors)
        self.assertEqual(purge_deleted(rows), set())

    def test_purge_all_and_empty_are_supported(self):
        rows = [Entry(1, {}, 1, deleted=True)]
        self.assertEqual(purge_deleted(rows), {1})
        self.assertEqual(rows, [])
        self.assertEqual(purge_deleted(rows), set())


class TestTargetedTagsAndPurgeUI(unittest.TestCase):
    def open(self, rows, page, *, engine=RULES):
        app = AppTest.from_file(str(APP), default_timeout=60)
        app.session_state.tagging_engine = engine
        app.session_state.rows = deepcopy(rows)
        app.session_state.dataset_name = "test.jsonl"
        app.run()
        self.navigate(app, page)
        return app

    def navigate(self, app, page):
        next(item for item in app.radio if item.label == "Navigation").set_value(page).run()
        self.assertFalse(app.exception)

    def click(self, app, label):
        next(item for item in app.button if item.label == label).click().run()
        self.assertFalse(app.exception)

    def test_targeted_qwen_only_analyzes_missing_rows_and_is_undoable(self):
        rows = [Entry(1, {"text": "Blushing and holding hands"}, 1),
                Entry(2, {"text": "Already classified"}, 2, auto_tags=["Comedy"]),
                Entry(3, {"text": "Source tags", "tags": ["SFW"]}, 3),
                Entry(4, {"text": "Manual clear"}, 4, tags_override=[]),
                Entry(5, {"text": "Deleted"}, 5, deleted=True)]
        app = self.open(rows, "🧹 Qualité", engine=QWEN_2B)
        app.session_state.tagging_errors = [(1, "Old failure"), (2, "Other issue")]
        app.session_state.export_original = b"stale"
        with patch("qwen_ui.LocalQwen.generate", return_value=RESPONSE) as generate:
            self.click(app, "Analyser uniquement les entrées sans tags")
        generate.assert_called_once()
        self.assertIn(rows[0].data["text"], generate.call_args.args[-1])
        self.assertEqual(app.session_state.rows[0].auto_tags, ["SFW", "Romance"])
        self.assertEqual(app.session_state.rows[1:], rows[1:])
        self.assertEqual(app.session_state.tagging_errors, [(2, "Other issue")])
        self.assertNotIn("export_original", app.session_state)
        self.assertTrue(next(item for item in app.button
                             if item.label == "Analyser uniquement les entrées sans tags").disabled)
        self.click(app, "↶ Annuler la dernière action")
        self.assertEqual(app.session_state.rows, rows)

    def test_interrupted_pass_can_resume_without_reanalyzing_successes(self):
        rows = [Entry(uid, {"text": f"Conversation {uid}"}, uid) for uid in range(1, 4)]
        app = self.open(rows, "🧹 Qualité", engine=QWEN_2B)
        with patch("qwen_ui.LocalQwen.generate", side_effect=[RESPONSE, QwenUnavailable("offline")]) as generate:
            self.click(app, "Analyser uniquement les entrées sans tags")
        self.assertEqual(generate.call_count, 2)
        self.assertEqual([bool(row.auto_tags) for row in app.session_state.rows], [True, False, False])
        with patch("qwen_ui.LocalQwen.generate", return_value=RESPONSE) as generate:
            self.click(app, "Analyser uniquement les entrées sans tags")
        self.assertEqual(generate.call_count, 2)
        self.assertTrue(all("Conversation 1" not in call.args[-1] for call in generate.call_args_list))
        self.assertTrue(all(row.auto_tags for row in app.session_state.rows))
        self.assertFalse(app.session_state.tagging_errors)

    def test_purge_requires_confirmation_clears_copies_and_exports_only_survivors(self):
        rows = [Entry(1, {"text": "Retain"}, 1, auto_tags=["SFW"]),
                Entry(2, {"text": "Discard secret"}, 2, deleted=True)]
        app = self.open(rows, "🗑️ Corbeille")
        self.assertTrue(next(item for item in app.button
                             if item.label == "Vider définitivement la corbeille").disabled)
        app.session_state.history = [{"kind": "edit", "id": 2, "before": rows[1].data}]
        app.session_state.entry_sources = {1: "keep.jsonl", 2: "discard.jsonl"}
        app.session_state.tagging_errors = [(1, "Keep error"), (2, "Discard error")]
        for key in ("export_original", "export_chatml", "export_project_data", "export_parts", "export_filtered",
                    "conversion_preview", "conversion_signature", "conversion_notice", "merge_analysis", "merge_signature"):
            app.session_state[key] = "cached data"
        next(item for item in app.checkbox if item.label.startswith("Confirmer la purge")).check().run()
        self.click(app, "Vider définitivement la corbeille")
        self.assertEqual(app.session_state.rows, [rows[0]])
        self.assertEqual(app.session_state.entry_sources, {1: "keep.jsonl"})
        self.assertEqual(app.session_state.tagging_errors, [(1, "Keep error")])
        self.assertFalse(app.session_state.history)
        self.assertTrue(next(item for item in app.button if item.label == "↶ Annuler la dernière action").disabled)
        for key in ("export_original", "export_chatml", "export_project_data", "export_parts", "export_filtered",
                    "conversion_preview", "conversion_signature", "conversion_notice", "merge_analysis", "merge_signature"):
            self.assertNotIn(key, app.session_state)
        self.assertTrue(any("1 entrée(s) retirée(s) définitivement" in item.value for item in app.success))
        self.navigate(app, "📦 Exporter")
        self.click(app, "Préparer la sauvegarde du projet")
        payload = json.loads(app.session_state.export_project_data)
        self.assertEqual([item["uid"] for item in payload["entries"]], [1])
        self.assertNotIn("Discard secret", app.session_state.export_project_data.decode())

    def test_purge_all_rows_keeps_empty_project_export_and_disables_purge(self):
        app = self.open([Entry(1, {"text": "Discard"}, 1, deleted=True)], "🗑️ Corbeille")
        next(item for item in app.checkbox if item.label.startswith("Confirmer la purge")).check().run()
        self.click(app, "Vider définitivement la corbeille")
        self.assertEqual(app.session_state.rows, [])
        self.assertTrue(next(item for item in app.button
                             if item.label == "Vider définitivement la corbeille").disabled)
        self.assertTrue(app.success)
        self.navigate(app, "📦 Exporter")
        self.click(app, "Préparer la sauvegarde du projet")
        self.assertEqual(json.loads(app.session_state.export_project_data)["entries"], [])


if __name__ == "__main__":
    unittest.main()
