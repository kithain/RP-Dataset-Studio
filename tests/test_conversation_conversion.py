from copy import deepcopy
import io
import json
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from core import Entry, as_messages, export_jsonl
from conversation_conversion import analyze_conversions, convert_conversation, partial_conversion
from merge_datasets import analyze_merge, prepare_merge


APP = Path(__file__).resolve().parents[1] / "app.py"


class TestConversationConversion(unittest.TestCase):
    def test_empty_messages_removed_and_content_metadata_and_source_unchanged(self):
        data = {"language": "fr", "metadata": {"keep": True}, "messages": [
            {"role": "user", "content": "Question\nAvec des accents et العربية", "name": "player"},
            {"role": "assistant", "content": "   "},
            {"role": "model", "content": "Réponse\nAvec des guillemets \"et\" 日本語", "weight": 1},
            {"role": "user", "content": None}]}
        before = deepcopy(data)
        result = convert_conversation(data)
        self.assertIsNotNone(result.data)
        self.assertEqual(len(result.data["messages"]), 2)
        self.assertEqual(as_messages(result.data), as_messages(data))
        self.assertFalse(partial_conversion(result.data, as_messages(result.data)))
        self.assertEqual(result.data["metadata"], data["metadata"])
        self.assertEqual(result.data["messages"][0]["name"], "player")
        self.assertEqual(result.data["messages"][1]["weight"], 1)
        self.assertEqual(result.data["messages"][1]["role"], "assistant")
        self.assertIn("2 message(s) vide(s)", "; ".join(result.notes))
        self.assertEqual(data, before)

    def test_sharegpt_and_serialized_messages(self):
        sequence = [{"from": "human", "value": "Question", "id": 1},
                    {"from": "gpt", "value": "Answer", "id": 2}]
        for data in ({"conversations": sequence, "meta": "keep"},
                     {"messages": json.dumps(sequence), "meta": "keep"},
                     {"dialogue": sequence, "meta": "keep"}):
            with self.subTest(data=data):
                result = convert_conversation(data)
                self.assertEqual(result.data["meta"], "keep")
                self.assertEqual([message["role"] for message in result.data["messages"]], ["user", "assistant"])
                self.assertEqual([message["content"] for message in result.data["messages"]], ["Question", "Answer"])
                self.assertEqual([message["id"] for message in result.data["messages"]], [1, 2])

    def test_explicit_pairs_preserve_system_and_existing_fields(self):
        for user_key, assistant_key in (("user", "assistant"), ("human", "gpt"),
                                         ("instruction", "response"), ("question", "response")):
            data = {user_key: "Question", assistant_key: "Answer", "system": "Instructions", "other": 42}
            result = convert_conversation(data)
            self.assertEqual({key: result.data[key] for key in data}, data)
            self.assertEqual(as_messages(result.data), [{"role": "system", "content": "Instructions"},
                                                       {"role": "user", "content": "Question"},
                                                       {"role": "assistant", "content": "Answer"}])

    def test_text_blocks_are_preserved_and_empty_blocks_can_be_removed(self):
        parts = [{"type": "input_text", "text": "Question", "metadata": 42}, "More context"]
        data = {"messages": [{"role": "user", "content": parts},
                              {"role": "assistant", "content": []},
                              {"role": "assistant", "content": [{"type": "output_text", "text": "Answer"}]}]}
        result = convert_conversation(data)
        self.assertEqual(result.data["messages"][0]["content"], parts)
        self.assertEqual(len(result.data["messages"]), 2)
        self.assertFalse(partial_conversion(result.data, as_messages(result.data)))

    def test_unknown_roles_images_tools_and_missing_answers_are_not_fabricated(self):
        user = {"role": "user", "content": "Question"}
        answer = {"role": "assistant", "content": "Answer"}
        for data in ({"text": "Some story without an explicit conversation"},
                     {"user": "Question", "assistant": ""},
                     {"messages": [user, {"role": "assistant", "content": ""}]},
                     {"messages": [user, answer, {"role": "unknown", "content": "Do not lose this"}]},
                     {"messages": [user, {**answer, "tool_calls": [{"id": "x"}]}]},
                     {"messages": [user, {"role": "assistant", "content": [{"type": "image_url", "image_url": "x"}]}]},
                     {"messages": [user, {**answer, "audio": {"data": "x"}}]},
                     {"messages": [user, "Unknown message"]},
                     {"messages": '[{"role":"user"'}):
            with self.subTest(data=data):
                before = deepcopy(data)
                result = convert_conversation(data)
                self.assertIsNone(result.data)
                self.assertTrue(result.reason)
                self.assertEqual(data, before)

    def test_analysis_skips_compatible_and_requires_opt_in_for_deleted(self):
        good = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        empty = {"messages": good["messages"] + [{"role": "assistant", "content": ""}]}
        rows = [Entry(1, good, 1), Entry(2, empty, 2), Entry(3, empty, 3, deleted=True),
                Entry(4, {"text": "No answer"}, 4)]
        self.assertEqual([candidate.uid for candidate in analyze_conversions(rows)], [2, 4])
        self.assertEqual([candidate.uid for candidate in analyze_conversions(rows, include_deleted=True)], [2, 3, 4])

    def test_merge_auto_converts_before_deduplication_and_can_be_disabled(self):
        good = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        empty = {"messages": good["messages"] + [{"role": "assistant", "content": ""}]}
        pair = {"user": "Other question", "assistant": "Other answer"}
        analysis = analyze_merge([("a.jsonl", io.BytesIO((json.dumps(empty) + "\n").encode())),
                                  ("b.jsonl", io.BytesIO((json.dumps(good) + "\n" + json.dumps(pair)).encode()))])
        self.assertEqual(set(analysis.recoveries), {1, 3})
        selected = prepare_merge(analysis, shuffle=False)
        self.assertEqual((selected.active, selected.converted, selected.duplicates, selected.excluded), (2, 2, 1, 0))
        self.assertEqual(export_jsonl(selected.rows, chatml=True)[1], 2)
        strict = prepare_merge(analysis, auto_convert=False, shuffle=False)
        self.assertEqual((strict.active, strict.converted, strict.excluded), (1, 0, 2))
        self.assertEqual(len(analysis.rows[0].data["messages"]), 3)


class TestConversionUI(unittest.TestCase):
    def quality_page(self, rows):
        app = AppTest.from_file(str(APP), default_timeout=60)
        app.session_state.rows = rows
        app.session_state.dataset_name = "test.jsonl"
        app.run()
        next(r for r in app.radio if r.label == "Navigation").set_value("🧹 Qualité").run()
        self.assertFalse(app.exception)
        return app

    def test_preview_conversion_restore_and_undo_preserve_original(self):
        data = {"metadata": "keep", "messages": [{"role": "user", "content": "Question"},
                {"role": "assistant", "content": ""}, {"role": "model", "content": "Answer"}]}
        app = self.quality_page([Entry(1, deepcopy(data), 1), Entry(2, deepcopy(data), 2, deleted=True),
                                 Entry(3, {"text": "No known answer"}, 3)])
        app.session_state.export_chatml = (b"stale", 0)
        next(c for c in app.checkbox if c.label == "Inclure les entrées de la corbeille").check().run()
        next(b for b in app.button if b.label == "Analyser les entrées incompatibles").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(next(m.value for m in app.metric if m.label == "Entrées récupérables"), "2")
        self.assertEqual(next(m.value for m in app.metric if m.label == "À corriger manuellement"), "1")
        next(b for b in app.button if b.label == "Convertir les 2 entrées récupérables").click().run()
        self.assertFalse(app.exception)
        self.assertNotIn("export_chatml", app.session_state)
        self.assertEqual(len(app.session_state.rows[0].data["messages"]), 2)
        self.assertFalse(app.session_state.rows[1].deleted)
        self.assertTrue(any("2 entrée(s) convertie(s)" in item.value for item in app.success))
        next(b for b in app.button if "Annuler la dernière action" in b.label).click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.rows[0].data, data)
        self.assertEqual(app.session_state.rows[1].data, data)
        self.assertTrue(app.session_state.rows[1].deleted)
        self.assertEqual(app.session_state.rows[2].data, {"text": "No known answer"})

    def test_conversion_preview_invalidated_when_scope_changes(self):
        app = self.quality_page([Entry(1, {"user": "Question", "assistant": "Answer"}, 1)])
        next(b for b in app.button if b.label == "Analyser les entrées incompatibles").click().run()
        self.assertIn("conversion_preview", app.session_state)
        next(c for c in app.checkbox if c.label == "Inclure les entrées de la corbeille").check().run()
        self.assertNotIn("conversion_preview", app.session_state)


if __name__ == "__main__":
    unittest.main()
