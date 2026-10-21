from copy import deepcopy
import json
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from core import (Entry, conversation_fields, export_jsonl, update_conversation_text)


APP = Path(__file__).resolve().parents[1] / "app.py"


class TestConversationFields(unittest.TestCase):
    def test_chat_edit_preserves_roles_metadata_and_original(self):
        data = {"id": "custom", "language": "fr", "tags": ["test"], "messages": [
            {"role": "system", "content": "Consigne", "extra": {"keep": True}},
            {"role": "user", "content": "Question", "name": "Player"},
            {"role": "assistant", "content": "Réponse", "tool_calls": [{"id": "tool1"}]},
            {"role": "tool", "content": "Résultat", "tool_call_id": "tool1"}]}
        before = deepcopy(data)
        fields = conversation_fields(data)
        self.assertEqual([f.role for f in fields], ["system", "user", "assistant", "tool"])
        changed = update_conversation_text(data, {fields[2].path: 'Nouvelle réponse\n"é" العربية 日本語'})
        expected = deepcopy(before)
        expected["messages"][2]["content"] = 'Nouvelle réponse\n"é" العربية 日本語'
        self.assertEqual(changed, expected)
        self.assertEqual(data, before)
        output, _ = export_jsonl([Entry(1, changed, 1)])
        self.assertEqual(json.loads(output), expected)

    def test_sharegpt_stays_in_original_format(self):
        data = {"conversations": [{"from": "human", "value": "Question"},
                                   {"from": "gpt", "value": "Answer", "weight": 0.5}]}
        fields = conversation_fields(data)
        self.assertEqual([f.role for f in fields], ["user", "assistant"])
        updated = update_conversation_text(data, {fields[1].path: "Edited"})
        self.assertNotIn("messages", updated)
        self.assertEqual(updated["conversations"][1], {"from": "gpt", "value": "Edited", "weight": 0.5})

    def test_alpaca_fields_are_edited_separately(self):
        data = {"system": "System", "instruction": "Instruction", "input": "Context",
                "output": "Answer", "metadata": {"source": "test"}}
        fields = conversation_fields(data)
        self.assertEqual([f.path for f in fields], [("system",), ("instruction",), ("input",), ("output",)])
        updated = update_conversation_text(data, {("input",): "New context"})
        self.assertEqual(updated, {**data, "input": "New context"})

    def test_other_text_schemas_and_incomplete_conversations(self):
        for data in ({"prompt": "Q", "completion": "A"}, {"question": "Q", "answer": "A"},
                     {"prompt": "Q", "response": "A"}, {"input": "Q", "output": "A"},
                     {"text": "Text"}, {"content": "Content"},
                     {"messages": [{"speaker": "narrator", "text": "Only assistant"}]},
                     {"messages": [{"role": "user", "content": ""}]}):
            with self.subTest(data=data):
                fields = conversation_fields(data)
                self.assertTrue(fields)
                self.assertEqual(update_conversation_text(data, {f.path: f.text for f in fields}), data)

    def test_edit_text_blocks_preserves_images_tools_and_unknown_messages(self):
        data = {"messages": [
            {"role": "user", "content": [{"type": "text", "text": "Look", "extra": True},
                                          {"type": "image_url", "image_url": {"url": "local.png"}},
                                          "Plain text", {"type": "input_text", "text": "Input"}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
            {"role": "tool", "content": {"unknown": 1}}, "unknown message"]}
        fields = conversation_fields(data)
        self.assertEqual([f.text for f in fields], ["Look", "Plain text", "Input"])
        updated = update_conversation_text(data, {fields[0].path: "Changed"})
        expected = deepcopy(data)
        expected["messages"][0]["content"][0]["text"] = "Changed"
        self.assertEqual(updated, expected)

    def test_arbitrary_paths_and_non_text_edits_are_rejected(self):
        data = {"text": "Hello", "metadata": "Keep"}
        for edits in ({("metadata",): "Changed"}, {("text",): None}, {("missing",): "New"}):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                update_conversation_text(data, edits)
        self.assertEqual(conversation_fields({"unknown": [1, 2, 3]}), [])


class TestConversationEditorUI(unittest.TestCase):
    def open_entry(self, data):
        app = AppTest.from_file(str(APP), default_timeout=60)
        app.session_state.rows = [Entry(1, deepcopy(data), 1)]
        app.session_state.dataset_name = "test.jsonl"
        app.run()
        self.assertFalse(app.exception)
        return app

    def mode(self, app, mode):
        next(r for r in app.radio if r.label == "Affichage de l'entrée").set_value(mode).run()

    def test_read_full_text_safely_and_save_edit_then_undo(self):
        long_text = ("Une longue histoire.\n" * 500) + '<script>alert("x")</script> ![image](https://example.com/test.png)'
        data = {"language": "fr", "metadata": {"keep": True}, "messages": [
            {"role": "user", "content": "Continue", "extra": 7},
            {"role": "assistant", "content": long_text}]}
        app = self.open_entry(data)
        self.assertEqual(len(app.chat_message), 2)
        rendered = "\n".join(m.value for m in app.markdown)
        self.assertIn("Une longue histoire.\n" * 500, rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn('<script>alert("x")</script>', rendered)
        self.assertFalse(app.text_area)
        self.mode(app, "✏️ Modifier les messages")
        self.assertEqual(len(app.text_area), 2)
        self.assertEqual(app.text_area[1].value, long_text)
        app.text_area[1].set_value('Texte réécrit\nAvec "guillemets" et العربية.')
        next(b for b in app.button if b.label == "Enregistrer la conversation").click().run()
        self.assertFalse(app.exception)
        expected = deepcopy(data)
        expected["messages"][1]["content"] = 'Texte réécrit\nAvec "guillemets" et العربية.'
        self.assertEqual(app.session_state.rows[0].data, expected)
        self.assertIn("Conversation enregistrée", app.success[0].value)
        self.mode(app, "{ } JSON avancé")
        self.assertEqual(json.loads(app.text_area[0].value), expected)
        next(b for b in app.button if "Annuler la dernière action" in b.label).click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.rows[0].data, data)

    def test_raw_json_edits_and_validation(self):
        data = {"text": "Before", "metadata": {"keep": True}}
        app = self.open_entry(data)
        self.mode(app, "{ } JSON avancé")
        app.text_area[0].set_value('{"text":')
        next(b for b in app.button if b.label == "Enregistrer les modifications").click().run()
        self.assertFalse(app.exception)
        self.assertIn("JSON non valide", app.error[0].value)
        self.assertEqual(app.session_state.rows[0].data, data)
        changed = {**data, "text": "Changed", "new_field": 42}
        app.text_area[0].set_value(json.dumps(changed))
        next(b for b in app.button if b.label == "Enregistrer les modifications").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.rows[0].data, changed)
        self.mode(app, "✏️ Modifier les messages")
        self.assertEqual(app.text_area[0].value, "Changed")

    def test_unknown_schema_can_still_be_edited_as_json(self):
        app = self.open_entry({"custom": {"value": "Not a recognized conversation"}})
        self.assertTrue(any("Aucun texte reconnu" in info.value for info in app.info))
        self.mode(app, "{ } JSON avancé")
        self.assertIn("custom", app.text_area[0].value)

    def test_long_conversation_can_be_saved_from_top(self):
        data = {"metadata": "keep", "messages": [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"Message {index}"}
            for index in range(12)]}
        app = self.open_entry(data)
        self.mode(app, "✏️ Modifier les messages")
        buttons = [b for b in app.button if b.label == "Enregistrer la conversation"]
        self.assertEqual(len(buttons), 2)
        app.text_area[11].set_value("Last message edited")
        buttons[0].click().run()
        self.assertFalse(app.exception)
        expected = deepcopy(data)
        expected["messages"][11]["content"] = "Last message edited"
        self.assertEqual(app.session_state.rows[0].data, expected)


if __name__ == "__main__":
    unittest.main()
