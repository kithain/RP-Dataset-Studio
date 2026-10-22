from pathlib import Path
import builtins
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from core import Entry, effective_language


APP = Path(__file__).resolve().parents[1] / "app.py"


class TestLanguageDetection(unittest.TestCase):
    def quality_page(self, rows):
        app = AppTest.from_file(str(APP), default_timeout=60)
        app.session_state.tagging_engine = "Règles rapides (sans LLM)"
        app.session_state.rows = rows
        app.session_state.dataset_name = "test.jsonl"
        app.run()
        next(r for r in app.radio if r.label == "Navigation").set_value("🧹 Qualité").run()
        self.assertFalse(app.exception)
        return app

    def detect(self, app):
        next(b for b in app.button if "Détecter la langue" in b.label).click().run()

    def test_detect_languages_refresh_and_undo(self):
        rows = [
            Entry(1, {"messages": [
                {"role": "user", "content": "Continue the story."},
                {"role": "assistant", "content":
                 "Le chevalier traverse la forêt et découvre une maison abandonnée. "
                 "Il ouvre doucement la porte et demande si quelqu'un habite ici."}]}, 1),
            Entry(2, {"conversations": [
                {"from": "human", "value": "Continue."},
                {"from": "gpt", "value":
                 "The knight walks through the forest and discovers an abandoned house. "
                 "He carefully opens the door and asks whether anyone lives here."}]}, 2),
            Entry(3, {"instruction": "واصل القصة", "output":
                      "دخل الفارس إلى القرية القديمة وسأل الحارس عن الطريق إلى القصر. "
                      "كان الليل هادئا والنجوم تضيء السماء فوق الجبال البعيدة."}, 3),
            Entry(4, {"text": "This deleted entry must stay untouched."}, 4, deleted=True),
            Entry(5, {"language": "de", "text": "Keep this language."}, 5),
            Entry(6, {"text": "Hi!"}, 6, language_override="?"),
        ]
        app = self.quality_page(rows)
        app.session_state.export_original = b"outdated export"
        self.detect(app)
        self.assertFalse(app.exception)
        actual = app.session_state.rows
        self.assertEqual([effective_language(r) for r in actual],
                         ["fr", "en", "ar", "?", "de", "?"])
        self.assertIsNone(actual[3].language_override)
        self.assertIsNone(actual[4].language_override)
        self.assertNotIn("export_original", app.session_state)
        self.assertIn("4 conversations analysées", app.success[0].value)
        self.assertIn("3 langues attribuées", app.success[0].value)
        self.assertEqual(next(m.value for m in app.metric if m.label == "Langue inconnue"), "1")
        next(b for b in app.button if "Annuler la dernière action" in b.label).click().run()
        self.assertFalse(app.exception)
        self.assertEqual([r.language_override for r in app.session_state.rows],
                         [None, None, None, None, None, "?"])

    def test_missing_detector_error_stays_visible(self):
        app = self.quality_page([Entry(1, {"text": "Une conversation à analyser."}, 1)])
        original_import = builtins.__import__

        def without_lingua(name, *args, **kwargs):
            if name == "lingua":
                raise ImportError("Detector unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_lingua):
            self.detect(app)
        self.assertFalse(app.exception)
        self.assertIn("Le détecteur est absent", app.error[0].value)
        self.assertIsNone(app.session_state.rows[0].language_override)
        self.assertFalse(app.session_state.history)


class TestImportRepairs(unittest.TestCase):
    def test_import_repair_summary_and_strict_reimport(self):
        payload = b'{"text":"first"}{"text":"second"}\n{"text":"third",}\nnot_json\n{"text":"fourth"}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_bytes(payload)
            app = AppTest.from_file(str(APP), default_timeout=60)
            app.session_state.tagging_engine = "Règles rapides (sans LLM)"
            app.run()
            next(t for t in app.text_input if t.label == "Chemin du fichier sur ce PC").set_value(str(path)).run()
            next(b for b in app.button if b.label == "Ouvrir le chemin local").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state.rows), 4)
            self.assertEqual(len(app.session_state.repairs), 2)
            self.assertEqual(len(app.session_state.errors), 1)
            self.assertIn("2 ligne(s) JSONL réparée(s)", app.success[0].value)
            self.assertIn("Certaines entrées", app.warning[0].value)
            next(c for c in app.checkbox if c.label == "Réparer les erreurs JSONL simples").uncheck().run()
            next(b for b in app.button if b.label == "Ouvrir le chemin local").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state.rows), 1)
            self.assertFalse(app.session_state.repairs)
            self.assertEqual(len(app.session_state.errors), 3)
            self.assertFalse(app.success)
            self.assertEqual(path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
