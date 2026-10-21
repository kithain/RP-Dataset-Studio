from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from zipfile import ZipFile

from streamlit.testing.v1 import AppTest

from core import Entry, export_jsonl
from split_export import export_split_zip


APP = Path(__file__).resolve().parents[1] / "app.py"


class TestSplitExport(unittest.TestCase):
    def test_default_boundaries_no_missing_or_duplicate_entries(self):
        for count, sizes in ((0, []), (1, [1]), (1999, [1999]), (2000, [2000]),
                             (2001, [2000, 1]), (4000, [2000, 2000]), (4001, [2000, 2000, 1])):
            with self.subTest(count=count):
                rows = [Entry(i, {"id": i, "text": "é العربية 日本語\nSecond line"}, i) for i in range(count)]
                blob, parts = export_split_zip(rows, "sample")
                self.assertEqual([size for _, size in parts], sizes)
                with ZipFile(io.BytesIO(blob)) as archive:
                    self.assertEqual(archive.namelist(), [filename for filename, _ in parts])
                    merged = b""
                    for filename, size in parts:
                        payload = archive.read(filename)
                        self.assertEqual(len(payload.splitlines()), size)
                        merged += payload
                    self.assertEqual(merged, export_jsonl(rows)[0])
                    if parts:
                        self.assertEqual(parts[0][0], "sample_nettoye_partie_0001.jsonl")

    def test_deletions_annotations_and_original_records_preserved(self):
        rows = [Entry(1, {"text": "Same", "metadata": {"keep": True}}, 1, language_override="fr"),
                Entry(2, {"text": "Deleted"}, 2, deleted=True),
                Entry(3, {"text": "Same", "metadata": {"keep": True}}, 3),
                Entry(4, {"text": "Last"}, 4)]
        before = deepcopy(rows)
        blob, parts = export_split_zip(rows, "sample", entries_per_file=2)
        self.assertEqual([count for _, count in parts], [2, 1])
        with ZipFile(io.BytesIO(blob)) as archive:
            merged = b"".join(archive.read(name) for name, _ in parts)
        self.assertEqual(merged, export_jsonl(rows)[0])
        self.assertEqual(json.loads(merged.splitlines()[0])["language"], "fr")
        self.assertEqual(rows, before)

    def test_chatml_filters_before_counting_and_keeps_whole_conversations(self):
        rows = [Entry(1, {"text": "Not convertible"}, 1)]
        rows += [Entry(i, {"prompt": f"Question {i}", "completion": f"Answer {i}\nContinued",
                           "metadata": "keep in source"}, i, language_override="fr") for i in range(2, 7)]
        rows[3].deleted = True
        blob, parts = export_split_zip(rows, "chat", entries_per_file=3, chatml=True)
        self.assertEqual([count for _, count in parts], [3, 1])
        with ZipFile(io.BytesIO(blob)) as archive:
            merged = b"".join(archive.read(name) for name, _ in parts)
        self.assertEqual(merged, export_jsonl(rows, chatml=True)[0])
        self.assertEqual(list(json.loads(merged.splitlines()[0])), ["messages"])
        self.assertEqual(parts[0][0], "chat_unsloth_partie_0001.jsonl")

    def test_invalid_part_sizes(self):
        for size in (0, -1, 1.5, "2000", True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                export_split_zip([], entries_per_file=size)

    def test_archive_names_cannot_contain_paths(self):
        for name in ("../source.jsonl", r"C:\datasets\source.jsonl", "bad:name?.jsonl", ".."):
            with self.subTest(name=name):
                blob, parts = export_split_zip([Entry(1, {"text": "A"}, 1)], name)
                filename = parts[0][0]
                self.assertNotIn("/", filename)
                self.assertNotIn("\\", filename)
                self.assertNotIn(":", filename)
                self.assertNotIn("?", filename)


class TestSplitExportUI(unittest.TestCase):
    def test_prepare_zip_default_settings_and_invalidate_when_settings_change(self):
        app = AppTest.from_file(str(APP), default_timeout=60)
        app.session_state.rows = [Entry(i, {"text": f"Record {i}"}, i) for i in range(4001)]
        app.session_state.dataset_name = "sample.jsonl"
        app.run()
        next(r for r in app.radio if r.label == "Navigation").set_value("📦 Exporter").run()
        self.assertFalse(app.exception)
        self.assertEqual(next(n.value for n in app.number_input if n.label == "Entrées par fichier"), 2000)
        self.assertTrue(any("Dernier fichier : 1 entrée." in info.value for info in app.info))
        next(b for b in app.button if b.label == "Préparer les fichiers découpés").click().run()
        self.assertFalse(app.exception)
        blob, parts, settings = app.session_state.export_parts
        self.assertEqual([count for _, count in parts], [2000, 2000, 1])
        self.assertEqual(settings, (2000, False))
        self.assertEqual(len(app.get("download_button")), 1)
        next(n for n in app.number_input if n.label == "Entrées par fichier").set_value(3000).run()
        self.assertNotIn("export_parts", app.session_state)
        next(b for b in app.button if b.label == "Préparer les fichiers découpés").click().run()
        self.assertEqual([count for _, count in app.session_state.export_parts[1]], [3000, 1001])
        next(r for r in app.radio if r.label == "Format des fichiers découpés").set_value("Unsloth / ChatML").run()
        self.assertFalse(app.exception)
        self.assertNotIn("export_parts", app.session_state)
        self.assertTrue(next(b.disabled for b in app.button if b.label == "Préparer les fichiers découpés"))


if __name__ == "__main__":
    unittest.main()
