from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from core import Entry
from finetuning_ui import PAGE
from qwen_ui import RULES


APP = Path(__file__).resolve().parents[1] / "app.py"


class TestStreamlitWidths(unittest.TestCase):
    def test_sidebar_tags_and_repeated_exclusion_tables_emit_no_width_deprecations(self):
        text = ("Le voyageur traverse la gare et attend son train près de la grande horloge. "
                "Il relit sa lettre pendant que les premiers passagers rejoignent le quai. ") * 2
        rows = [Entry(index, {"prompt": "Raconte cette attente.", "completion": text}, index,
                      auto_tags=["Adventure"]) for index in range(1, 4)]
        with tempfile.TemporaryDirectory() as directory, \
                patch("finetuning_ui.CHECKPOINTS", Path(directory)), \
                patch("finetuning_ui.SETTINGS", Path(directory) / "settings.json"), \
                patch("streamlit.deprecation_util._LOGGER.warning") as warnings, \
                patch("qwen_ui.LocalQwen.load") as load:
            app = AppTest.from_file(str(APP), default_timeout=30)
            app.session_state.rows = rows
            app.session_state.dataset_name = "widths.jsonl"
            app.session_state.tagging_engine = RULES
            app.run()
            self.assertFalse(app.exception)
            next(r for r in app.radio if r.label == "Navigation").set_value("🏷️ Tags").run()
            self.assertFalse(app.exception)
            next(r for r in app.radio if r.label == "Navigation").set_value(PAGE).run()
            next(b for b in app.button if b.label == "Préparer / reprendre le dataset").click().run()
            for _ in range(3):
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(any(exp.label == "Motifs des exclusions (2)" for exp in app.expander))
            messages = [str(call) for call in warnings.call_args_list]
            self.assertFalse([message for message in messages if "use_container_width" in message])
            self.assertEqual(len(app.session_state.rows), 3)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
