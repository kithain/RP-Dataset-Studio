import io
import json
import unittest

from core import (as_messages, apply_filters, delete_ids, duplicate_ids, effective_language,
                  export_jsonl, export_project, load_dataset, restore_ids)


def make(records):
    payload = ("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n").encode("utf-8")
    return load_dataset(io.BytesIO(payload), "test.jsonl")[0]


class TestRPDataset(unittest.TestCase):
    def setUp(self):
        self.rows = make([
            {"language": "fr", "messages": [
                {"role": "user", "content": "*Je dégaine mon épée.*"},
                {"role": "assistant", "content": "*La garde s'avance.*"}]},
            {"language": "en", "conversations": [
                {"from": "human", "value": "*I open the door.*"},
                {"from": "gpt", "value": "*The guard looks at me.*"}]},
            {"instruction": "Décris une romance", "output": "Le soleil se couche."},
            {"language": "fr", "messages": [
                {"role": "user", "content": "*Je dégaine mon épée.*"},
                {"role": "assistant", "content": "*La garde s'avance.*"}]},
            {"text": "Autre format non convertible"},
        ])

    def test_import_jsonl(self):
        self.assertEqual(len(self.rows), 5)
        self.assertEqual(self.rows[0].source_line, 1)
        self.assertEqual(self.rows[4].uid, 5)

    def test_filter_language(self):
        result = apply_filters(self.rows, languages=["fr"])
        self.assertEqual([r.uid for r in result], [1, 4])

    def test_include_exclude_terms_and_regex(self):
        match = apply_filters(self.rows, contains="garde ; romance", excludes="door")
        self.assertEqual([r.uid for r in match], [1, 3, 4])
        regex = apply_filters(self.rows, contains=r"épée|romance", use_regex=True)
        self.assertEqual([r.uid for r in regex], [1, 3, 4])

    def test_chatml_and_preservation(self):
        self.assertIsNone(as_messages(self.rows[4].data))
        self.assertEqual(as_messages(self.rows[1].data)[1]["role"], "assistant")
        self.assertEqual(as_messages(self.rows[2].data)[0]["content"], "Décris une romance")
        raw, n_raw = export_jsonl(self.rows)
        chat, n_chat = export_jsonl(self.rows, chatml=True)
        self.assertEqual(n_raw, 5)
        self.assertEqual(n_chat, 4)
        self.assertEqual(json.loads(raw.splitlines()[1])["conversations"][0]["from"], "human")
        self.assertIn("messages", json.loads(chat.splitlines()[1]))
        self.assertNotIn("conversations", json.loads(chat.splitlines()[1]))

    def test_deduplication_and_delete_restore(self):
        self.assertEqual(duplicate_ids(self.rows), [4])
        changed = delete_ids(self.rows, {1, 4})
        self.assertEqual(changed, [1, 4])
        self.assertEqual(duplicate_ids(self.rows), [])
        self.assertEqual(export_jsonl(self.rows)[1], 3)
        restore_ids(self.rows, {4})
        self.assertFalse(self.rows[3].deleted)
        self.assertEqual(export_jsonl(self.rows)[1], 4)

    def test_project_saves_deleted_and_language(self):
        delete_ids(self.rows, {2})
        self.rows[0].language_override = "de"
        project = export_project(self.rows, "original.jsonl")
        read, errors = load_dataset(io.BytesIO(project), "save.rpwork.json")
        self.assertFalse(errors)
        self.assertTrue(read[1].deleted)
        self.assertEqual(effective_language(read[0]), "de")
        self.assertEqual(len(read), 5)
        original, _ = export_jsonl(read)
        self.assertEqual(json.loads(original.splitlines()[0])["language"], "de")

    def test_csv_import_messages_json(self):
        text = 'language,messages\nfr,"[{""role"": ""user"", ""content"": ""Salut""}, {""role"": ""assistant"", ""content"": ""Bonsoir""}]"\n'
        rows, errors = load_dataset(io.BytesIO(text.encode()), "test.csv")
        self.assertFalse(errors)
        self.assertEqual(as_messages(rows[0].data)[1]["content"], "Bonsoir")

    def test_semicolon_csv(self):
        rows, errors = load_dataset(io.BytesIO('language;prompt;completion\nfr;salut;bonjour\n'.encode()), 'data.csv')
        self.assertFalse(errors)
        self.assertEqual(as_messages(rows[0].data)[1]['content'], 'bonjour')

    def test_json_import(self):
        rows, _ = load_dataset(io.BytesIO(b'{"data":[{"prompt":"ok","completion":"yes"}]}'), "test.json")
        self.assertEqual(as_messages(rows[0].data)[0]["role"], "user")

    def test_invalid_jsonl_only_skips_invalid(self):
        data = b'{"messages":[]}\nnot_json\n{"instruction":"ok","output":"yes"}\n'
        rows, errors = load_dataset(io.BytesIO(data), "test.jsonl")
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(errors), 1)
        self.assertEqual(rows[1].source_line, 3)

    def test_repair_concatenated_objects_preserves_all_records_and_line_numbers(self):
        first = {"messages": [{"role": "assistant", "content": 'Accents é, العربية, 日本語; ",}" et ",]"; \\ "'}]}
        second = {"text": "Two objects on one line"}
        payload = ("\n" + json.dumps(first, ensure_ascii=False) +
                   json.dumps(second) + '\n{"text":"next line"}\n').encode()
        repairs = []
        rows, errors = load_dataset(io.BytesIO(payload), "test.jsonl", repairs=repairs)
        self.assertFalse(errors)
        self.assertEqual([r.data for r in rows], [first, second, {"text": "next line"}])
        self.assertEqual([r.uid for r in rows], [1, 2, 3])
        self.assertEqual([r.source_line for r in rows], [2, 2, 3])
        self.assertIn("Ligne 2 : 2 objets JSON séparés", repairs[0])
        exported, count = export_jsonl(rows)
        reloaded, errors = load_dataset(io.BytesIO(exported), "fixed.jsonl", repair_jsonl=False)
        self.assertFalse(errors)
        self.assertEqual(count, 3)
        self.assertEqual([r.data for r in reloaded], [r.data for r in rows])

    def test_repair_trailing_commas_outside_strings_only(self):
        original = {"text": 'Keep ,} and ,] and escaped "quotes" and \\ paths', "items": [1, {"x": 2}]}
        # Explicit nested trailing commas plus a separator at the end of the line.
        malformed = '{"text":' + json.dumps(original["text"]) + ',"items":[1,{"x":2,},],},'
        repairs = []
        rows, errors = load_dataset(io.BytesIO(malformed.encode()), "test.jsonl", repairs=repairs)
        self.assertFalse(errors)
        self.assertEqual(rows[0].data, original)
        self.assertIn("virgules superflues", repairs[0])

    def test_repair_comma_separated_objects(self):
        rows, errors = load_dataset(io.BytesIO(b'{"a":1}, {"b":2},\n'), "test.jsonl")
        self.assertFalse(errors)
        self.assertEqual([r.data for r in rows], [{"a": 1}, {"b": 2}])

    def test_repair_rejects_whole_line_with_invalid_suffix(self):
        for invalid in (b'{"a":1}garbage', b'{"a":1}{"truncated":',
                        b'{"a":1}42', b'{"a":1}[]', b'{"a":1},,{"b":2}',
                        b'{"a":1,,}', b"{'a':1}", b'{"a":"unterminated}', b'\xff'):
            with self.subTest(invalid=invalid):
                repairs = []
                rows, errors = load_dataset(io.BytesIO(invalid + b'\n{"valid":true}\n'),
                                            "test.jsonl", repairs=repairs)
                self.assertEqual([r.data for r in rows], [{"valid": True}])
                self.assertEqual(rows[0].source_line, 2)
                self.assertEqual(len(errors), 1)
                self.assertFalse(repairs)

    def test_repair_can_be_disabled(self):
        payload = b'{"a":1}{"b":2}\n{"c":3,}\n{"valid":true}\n'
        repairs = []
        rows, errors = load_dataset(io.BytesIO(payload), "test.jsonl", repair_jsonl=False,
                                    repairs=repairs)
        self.assertEqual([r.data for r in rows], [{"valid": True}])
        self.assertEqual(len(errors), 2)
        self.assertFalse(repairs)

    def test_valid_jsonl_is_not_reported_as_repaired(self):
        payload = b'\xef\xbb\xbf{"text":"literal ,} and ,]","items":[]}\r\n'
        repairs = []
        rows, errors = load_dataset(io.BytesIO(payload), "test.jsonl", repairs=repairs)
        self.assertFalse(errors)
        self.assertFalse(repairs)
        self.assertEqual(rows[0].data["text"], "literal ,} and ,]")

    def test_convert_only_valid_pair(self):
        self.assertIsNone(as_messages({"messages": [{"role": "assistant", "content": "abc"}]}))
        self.assertEqual(as_messages({"messages": [{"role": "user", "content": "a"},
                                                   {"role": "assistant", "content": [{"type":"text", "text":"b"}]}]})[1]["content"], "b")


if __name__ == "__main__":
    unittest.main()
