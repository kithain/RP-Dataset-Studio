from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import runtime_imports


BASE = Path(__file__).resolve().parents[1]


class TestRuntimeImports(unittest.TestCase):
    def test_compatible_imports_are_not_reloaded(self):
        # Other tests may already have loaded the application modules.
        with patch("runtime_imports.importlib.reload") as reload_module:
            self.assertEqual(runtime_imports.refresh_cached_modules(), [])
        reload_module.assert_not_called()

    def test_module_from_another_directory_is_not_reloaded(self):
        foreign = SimpleNamespace(__file__=str(BASE.parent / "core.py"))
        with patch.object(runtime_imports, "REQUIRED_API", (("core", {"purge_deleted": ()}),)), \
                patch.dict(sys.modules, {"core": foreign}), \
                patch("runtime_imports.importlib.reload") as reload_module:
            self.assertEqual(runtime_imports.refresh_cached_modules(), [])
        reload_module.assert_not_called()

    def test_missing_api_on_disk_stays_an_explicit_import_error(self):
        stale = SimpleNamespace(__file__=str(BASE / "core.py"))
        with patch.object(runtime_imports, "REQUIRED_API", (("core", {"purge_deleted": ()}),)), \
                patch.dict(sys.modules, {"core": stale}), \
                patch("runtime_imports.importlib.reload", return_value=stale):
            with self.assertRaisesRegex(ImportError, "tous les fichiers"):
                runtime_imports.refresh_cached_modules()

    def test_stale_modules_recover_without_losing_rows_history_or_qwen_pool(self):
        # Isolate class redefinitions and the synthetic stale cache from the other tests.
        script = r'''
from pathlib import Path
from streamlit.testing.v1 import AppTest
import core
import tagging
import tag_management
import tagging_ui
import qwen_ui
import llm_tagging
import book_generation
import book_ui
import finetuning
import finetuning_ui

row = core.Entry(11, {"text": "Romance"}, 4)
pool = qwen_ui.clients()
original_normalize = tagging.normalize_tags
def legacy_normalize(value):
    return original_normalize(value)
tagging.normalize_tags = legacy_normalize
del core.purge_deleted
del tag_management.untagged_entries
del tagging_ui.analyze_tag_batch
del llm_tagging.generate_json
del qwen_ui.generate_book
del book_generation.validate_narration
book_ui.render_book_page = lambda on_load: None
del finetuning.source_parts
del finetuning.exclusion_report
del finetuning_ui.render_exclusion_details
finetuning_ui.render_finetuning_page = lambda rows, name, sources=None: None

app = AppTest.from_file(str(Path("app.py").resolve()), default_timeout=30)
app.session_state.rows = [row]
app.session_state.dataset_name = "open.jsonl"
app.session_state.tagging_engine = qwen_ui.RULES
app.session_state.history = [{"kind": "lang", "before": [(11, None)]}]
app.session_state.entry_sources = {11: "source.jsonl"}
app.run()
assert not app.exception, app.exception
assert callable(core.purge_deleted)
assert callable(tag_management.untagged_entries)
assert callable(tagging_ui.analyze_tag_batch)
assert callable(llm_tagging.generate_json)
assert callable(qwen_ui.generate_book)
assert callable(book_generation.validate_narration)
assert "generator_version" in __import__("inspect").signature(book_ui.render_book_page).parameters
assert callable(finetuning.source_parts)
assert callable(finetuning.exclusion_report)
assert callable(finetuning_ui.render_exclusion_details)
assert "generator_version" in __import__("inspect").signature(finetuning_ui.render_finetuning_page).parameters
assert "resolve_classification" in __import__("inspect").signature(tagging.normalize_tags).parameters
assert app.session_state.rows[0] is row
assert app.session_state.history == [{"kind": "lang", "before": [(11, None)]}]
assert app.session_state.entry_sources == {11: "source.jsonl"}
assert qwen_ui.clients() is pool
next(widget for widget in app.radio if widget.label == "Navigation").set_value("🧹 Qualité").run()
assert not app.exception, app.exception
next(widget for widget in app.button if widget.label == "Analyser uniquement les entrées sans tags").click().run()
assert not app.exception, app.exception
assert row.auto_tags == ["SFW", "Romance"], row.auto_tags
assert qwen_ui.clients() is pool
print("reload_ok_state_and_qwen_preserved")
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=BASE,
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("reload_ok_state_and_qwen_preserved", result.stdout)


if __name__ == "__main__":
    unittest.main()
