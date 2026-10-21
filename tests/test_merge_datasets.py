from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from streamlit.testing.v1 import AppTest

from core import as_messages, export_jsonl, export_project, load_dataset, restore_ids
from merge_datasets import analyze_merge, prepare_merge


APP = Path(__file__).resolve().parents[1] / "app.py"


def source(name, records):
    return name, io.BytesIO(("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n").encode())


class TestMergeDatasets(unittest.TestCase):
    def test_merge_formats_reindexes_ids_and_keeps_origins_metadata_and_system(self):
        first = [{"system": "System instruction", "prompt": "Question", "completion": "Answer", "extra": 1}]
        second = [{"conversations": [{"from": "human", "value": "Bonjour"},
                                      {"from": "gpt", "value": "Salut"}], "language": "fr"},
                  {"instruction": "Task", "input": "Context", "output": "Done"}]
        analysis = analyze_merge([source("first.jsonl", first), source("second.jsonl", second)])
        self.assertEqual([row.uid for row in analysis.rows], [1, 2, 3])
        self.assertEqual([row.source_line for row in analysis.rows], [1, 1, 2])
        self.assertEqual(analysis.origins, {1: "first.jsonl", 2: "second.jsonl", 3: "second.jsonl"})
        self.assertFalse(analysis.incompatible)
        self.assertFalse(analysis.partial)
        selected = prepare_merge(analysis, shuffle=False)
        self.assertEqual(selected.active, 3)
        self.assertEqual([row.data for row in selected.rows], first + second)
        payload, count = export_jsonl(selected.rows, chatml=True)
        self.assertEqual(count, 3)
        self.assertEqual(json.loads(payload.splitlines()[0])["messages"][0],
                         {"role": "system", "content": "System instruction"})
        self.assertTrue(all(list(json.loads(line)) == ["messages"] for line in payload.splitlines()))

    def test_global_duplicates_keep_first_and_can_be_restored(self):
        analysis = analyze_merge([
            source("a.jsonl", [{"prompt": "Hello  there", "completion": "Good day", "meta": "first"}]),
            source("b.jsonl", [{"messages": [{"role": "user", "content": "HELLO there"},
                                              {"role": "assistant", "content": "good DAY"}]}])])
        before = deepcopy(analysis.rows)
        selected = prepare_merge(analysis, shuffle=False)
        self.assertEqual((selected.duplicates, selected.active), (1, 1))
        self.assertEqual([row.deleted for row in selected.rows], [False, True])
        self.assertEqual(selected.rows[0].data["meta"], "first")
        self.assertEqual(analysis.rows, before)
        restore_ids(selected.rows, {2})
        self.assertEqual(export_jsonl(selected.rows, chatml=True)[1], 2)
        self.assertEqual(prepare_merge(analysis, remove_duplicates=False).active, 2)

    def test_incomplete_conversions_excluded_before_duplicate_detection(self):
        messages = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]
        analysis = analyze_merge([
            source("partial.jsonl", [{"messages": messages + [{"role": "tool", "content": "Result"}]}]),
            source("valid.jsonl", [{"messages": messages}, {"text": "Not a conversation"}])])
        self.assertEqual(analysis.partial, {1})
        self.assertEqual(analysis.incompatible, {3})
        selected = prepare_merge(analysis, shuffle=False)
        self.assertEqual((selected.excluded, selected.duplicates, selected.active), (2, 0, 1))
        self.assertEqual([row.uid for row in selected.rows if not row.deleted], [2])

    def test_multimodal_and_tool_calls_are_flagged(self):
        user = {"role": "user", "content": "Question"}
        answer = {"role": "assistant", "content": "Answer"}
        records = [{"messages": [user, {**answer, "tool_calls": [{"id": "x"}]}]},
                   {"messages": [{"role": "user", "content": [{"type": "text", "text": "Question"},
                                  {"type": "image_url", "image_url": {"url": "image.png"}}]}, answer]},
                   {"messages": [{"role": "user", "content": [{"type": "text", "text": "Other question"}]}, answer]}]
        analysis = analyze_merge([source("a.jsonl", records)])
        self.assertEqual(analysis.partial, {1, 2})
        selected = prepare_merge(analysis, shuffle=False)
        self.assertEqual((selected.active, selected.excluded), (1, 2))
        selected_all = prepare_merge(analysis, exclude_incompatible=False, remove_duplicates=False)
        self.assertEqual(selected_all.active, 3)

    def test_source_project_deleted_rows_overrides_and_roundtrip(self):
        rows, _ = load_dataset(source("a.jsonl", [{"prompt": "Q", "completion": "A"},
                                                  {"prompt": "Old", "completion": "Deleted"}])[1], "a.jsonl")
        rows[0].uid = 99
        rows[0].language_override = "ja"
        rows[1].deleted = True
        project = export_project(rows, "original.jsonl")
        analysis = analyze_merge([("source.rpwork.json", io.BytesIO(project)),
                                  source("other.jsonl", [{"prompt": "New", "completion": "Answer"}])])
        selected = prepare_merge(analysis, shuffle=False)
        self.assertEqual(selected.already_deleted, 1)
        self.assertEqual(selected.active, 2)
        self.assertEqual(selected.rows[0].language_override, "ja")
        saved, errors = load_dataset(io.BytesIO(export_project(selected.rows, "merged")), "merged.rpwork.json")
        self.assertFalse(errors)
        self.assertEqual(len(saved), 3)
        self.assertEqual(sum(row.deleted for row in saved), 1)

    def test_repairs_and_errors_include_source_names_and_other_files_continue(self):
        broken = io.BytesIO(b'{"prompt":"Q","completion":"A"}{"prompt":"B","completion":"C"}\ninvalid\n')
        analysis = analyze_merge([("repaired.jsonl", broken), ("bad.json", io.BytesIO(b'not json')),
                                  source("good.jsonl", [{"prompt": "Other", "completion": "Valid"}])])
        self.assertEqual(len(analysis.rows), 3)
        self.assertEqual(len(analysis.repairs), 1)
        self.assertIn("repaired.jsonl", analysis.repairs[0])
        self.assertEqual(len(analysis.errors), 2)
        self.assertIn("bad.json", analysis.errors[1])
        self.assertEqual(analysis.sources[1]["Entrées lues"], 0)
        self.assertEqual(prepare_merge(analysis).convertible, 3)

    def test_repair_disabled_and_missing_local_file(self):
        analysis = analyze_merge([("a.jsonl", io.BytesIO(b'{"prompt":"Q","completion":"A"},\n')),
                                  ("missing.jsonl", Path(__file__).parent / "missing.jsonl")], repair_jsonl=False)
        self.assertFalse(analysis.rows)
        self.assertEqual(len(analysis.errors), 2)
        self.assertEqual(prepare_merge(analysis).convertible, 0)

    def test_shuffle_is_reproducible_and_does_not_change_global_random_state(self):
        analysis = analyze_merge([source("a.jsonl", [{"prompt": str(i), "completion": "Answer"} for i in range(30)])])
        before = random.getstate()
        first = prepare_merge(analysis, seed=42)
        second = prepare_merge(analysis, seed=42)
        other = prepare_merge(analysis, seed=43)
        self.assertEqual([r.uid for r in first.rows], [r.uid for r in second.rows])
        self.assertNotEqual([r.uid for r in first.rows], [r.uid for r in other.rows])
        self.assertEqual({r.uid for r in first.rows}, set(range(1, 31)))
        self.assertEqual(random.getstate(), before)


class TestMergeUI(unittest.TestCase):
    def open_page(self):
        app = AppTest.from_file(str(APP), default_timeout=60).run()
        next(r for r in app.radio if r.label == "Navigation").set_value("🧩 Fusionner").run()
        self.assertFalse(app.exception)
        return app

    def test_merge_local_files_preview_options_and_prepared_export(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.jsonl", Path(directory) / "b.jsonl"
            first = source("a.jsonl", [{"prompt": "Bonjour", "completion": "Salut", "metadata": "keep"},
                                        {"text": "Not a conversation"}])[1].getvalue()
            second = source("b.jsonl", [{"messages": [{"role": "user", "content": "Bonjour"},
                                                       {"role": "assistant", "content": "Salut"}]},
                                         {"prompt": "Different", "completion": "New"}])[1].getvalue()
            a.write_bytes(first)
            b.write_bytes(second)
            app = self.open_page()
            self.assertTrue(next(b.disabled for b in app.button if b.label == "Analyser les fichiers à fusionner"))
            next(t for t in app.text_area if t.label == "Ou chemins locaux, un par ligne").set_value(f'{a}\n"{b}"').run()
            next(b for b in app.button if b.label == "Analyser les fichiers à fusionner").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(next(m.value for m in app.metric if m.label == "Conversations dans l'export Unsloth"), "2")
            self.assertNotIn("export_chatml", app.session_state)
            self.assertIsNone(app.session_state.dataset_name)
            next(c for c in app.checkbox if c.label == "Retirer les doublons entre tous les fichiers").uncheck().run()
            self.assertEqual(next(m.value for m in app.metric if m.label == "Conversations dans l'export Unsloth"), "3")
            next(c for c in app.checkbox if c.label == "Retirer les doublons entre tous les fichiers").check().run()
            next(b for b in app.button if b.label == "Fusionner et préparer l'export Unsloth").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.navigation_page, "📦 Exporter")
            self.assertEqual(len(app.session_state.rows), 4)
            self.assertEqual(sum(row.deleted for row in app.session_state.rows), 2)
            blob, count = app.session_state.export_chatml
            self.assertEqual(count, 2)
            self.assertEqual(len(blob.splitlines()), 2)
            self.assertEqual(app.session_state.merge_report["doublons_ecartes"], 1)
            self.assertEqual(app.session_state.merge_report["incompatibles_ecartees"], 1)
            self.assertEqual(len(app.session_state.entry_sources), 4)
            self.assertEqual(a.read_bytes(), first)
            self.assertEqual(b.read_bytes(), second)
            self.assertTrue(any("Fusion terminée" in item.value for item in app.success))

    def test_changing_inputs_invalidates_analysis_and_no_valid_conversations_disables_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.jsonl", Path(directory) / "b.jsonl"
            a.write_text('{"text":"Not a conversation"}\n', encoding="utf-8")
            b.write_text('{"text":"Other text"}\n', encoding="utf-8")
            app = self.open_page()
            next(t for t in app.text_area if t.label == "Ou chemins locaux, un par ligne").set_value(f"{a}\n{b}").run()
            next(b for b in app.button if b.label == "Analyser les fichiers à fusionner").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(next(b.disabled for b in app.button if b.label == "Fusionner et préparer l'export Unsloth"))
            next(t for t in app.text_area if t.label == "Ou chemins locaux, un par ligne").set_value(str(a)).run()
            self.assertNotIn("merge_analysis", app.session_state)


if __name__ == "__main__":
    unittest.main()
