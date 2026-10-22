from copy import deepcopy
import io
import json
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from core import Entry, apply_filters, export_jsonl, export_project, load_dataset
from tagging import entry_tags
from tag_management import apply_tag_change, plan_tag_change, tag_catalogue


APP = Path(__file__).resolve().parents[1] / "app.py"


def sample_rows():
    return [Entry(1, {"prompt": "Question", "completion": "Answer"}, 1,
                  auto_tags=["SFW", "Romance", "Comedy"]),
            Entry(2, {"prompt": "Other", "completion": "Response"}, 2,
                  auto_tags=["NSFW", "Comedy", "Vore"]),
            Entry(3, {"text": "Notes"}, 3, auto_tags=["Needs Review", "Romance"]),
            Entry(4, {"text": "No annotation"}, 4),
            Entry(5, {"text": "Trash"}, 5, deleted=True, auto_tags=["SFW", "Comedy"])]


class TestTagManagement(unittest.TestCase):
    def ids(self, **kwargs):
        return [row.uid for row in apply_filters(sample_rows(), **kwargs)]

    def test_any_all_and_exclusion_are_exact_and_combined(self):
        self.assertEqual(self.ids(tags=["Comedy", "Romance"]), [1])
        self.assertEqual(self.ids(tags=["Comedy", "Romance"], tag_mode="any"), [1, 2, 3])
        self.assertEqual(self.ids(tags=["Comedy", "Romance"], tag_mode="any", excluded_tags=["nsfw"]), [1, 3])
        self.assertEqual(self.ids(tags=["SFW"], tag_mode="any"), [1])
        self.assertEqual(self.ids(tags=["SFW", "NSFW"], tag_mode="any"), [1, 2])
        self.assertEqual(self.ids(tags=["SFW", "NSFW"], tag_mode="all"), [])
        self.assertEqual(self.ids(excluded_tags=["SFW", "NSFW"]), [3, 4])
        self.assertEqual(self.ids(tags=["Rom"]), [])
        self.assertEqual(self.ids(tags=["comédie"], contains="Question"), [1])

    def test_tag_presence_empty_selection_and_deleted_rows(self):
        self.assertEqual(self.ids(tag_status="untagged"), [4])
        self.assertEqual(self.ids(tag_status="tagged", tag_mode="any"), [1, 2, 3])
        self.assertEqual(self.ids(tag_status="untagged", tags=["SFW"]), [])
        self.assertEqual(self.ids(tag_mode="any", excluded_tags=["Comedy"]), [3, 4])
        self.assertEqual(self.ids(tags=["Comedy"], include_deleted=True), [1, 2, 5])
        with self.assertRaises(ValueError):
            self.ids(tag_mode="unknown")
        with self.assertRaises(ValueError):
            self.ids(tag_status="unknown")

    def test_catalogue_counts_effective_tags_once_groups_case_and_ignores_trash(self):
        rows = [Entry(1, {"tags": ["custom", "CUSTOM", "comédie"]}, 1),
                Entry(2, {}, 2, auto_tags=["Custom", "Comedy"], tags_override=["CUSTOM", "Romance"]),
                Entry(3, {"tags": ["custom"]}, 3, tags_override=[]),
                Entry(4, {"tags": ["custom"]}, 4, deleted=True)]
        actual = {item["Tag"].casefold(): item for item in tag_catalogue(rows)}
        self.assertEqual(actual["custom"]["Entrées"], 2)
        self.assertEqual(actual["comedy"]["Entrées"], 1)
        self.assertEqual(actual["comedy"]["Famille"], "Genre")
        self.assertEqual(actual["custom"]["Famille"], "Personnage / source / tags libres")
        self.assertEqual(len(actual), 3)

    def test_bulk_merge_preserves_sources_trash_and_roundtrips(self):
        rows = [Entry(1, {"tags": ["Old", "Keep"]}, 1, auto_tags=["New"]),
                Entry(2, {}, 2, tags_override=["OLD", "Keep"]),
                Entry(3, {"tags": ["Old"]}, 3, deleted=True)]
        original = deepcopy(rows)
        changes = plan_tag_change(rows, "old", "New")
        self.assertTrue(all(row.tags_override == old.tags_override for row, old in zip(rows, original)))
        before = apply_tag_change(changes)
        self.assertEqual([entry_tags(row) for row in rows], [["New", "Keep"], ["New", "Keep"], ["Old"]])
        self.assertEqual([row.data for row in rows], [row.data for row in original])
        self.assertEqual(rows[0].auto_tags, ["New"])
        blob, count = export_jsonl(rows)
        self.assertEqual(count, 2)
        self.assertEqual(json.loads(blob.splitlines()[0])["tags"], ["New", "Keep"])
        restored, errors = load_dataset(io.BytesIO(export_project(rows, "test")), "test.rpwork.json")
        self.assertEqual(restored, rows)
        self.assertFalse(errors)
        for row, (uid, auto, override) in zip(rows, before):
            self.assertEqual(row.uid, uid)
            row.auto_tags, row.tags_override = auto, override
        self.assertEqual(rows, original)

    def test_remove_last_tag_is_explicit_and_rename_validates_single_tag(self):
        row = Entry(1, {"tags": ["Old"]}, 1)
        self.assertEqual(plan_tag_change([row], "Old", "Old"), [])
        for invalid in ("", "A; B", '[]'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                plan_tag_change([row], "Old", invalid)
        apply_tag_change(plan_tag_change([row], "OLD"))
        self.assertEqual(row.tags_override, [])
        self.assertEqual(entry_tags(row), [])
        self.assertEqual(row.data["tags"], ["Old"])


class TestTagBrowserUI(unittest.TestCase):
    def open(self, page="🔎 Explorer", rows=None):
        app = AppTest.from_file(str(APP), default_timeout=60)
        app.session_state.tagging_engine = "Règles rapides (sans LLM)"
        app.session_state.rows = sample_rows() if rows is None else rows
        app.session_state.dataset_name = "test.jsonl"
        app.run()
        if page != "🔎 Explorer":
            self.navigate(app, page)
        self.assertFalse(app.exception)
        return app

    def navigate(self, app, page):
        next(item for item in app.radio if item.label == "Navigation").set_value(page).run()
        self.assertFalse(app.exception)

    def click(self, app, label):
        next(item for item in app.button if item.label == label).click().run()
        self.assertFalse(app.exception)

    def test_any_exclusion_shortcuts_reset_and_form_synchronization(self):
        app = self.open()
        next(item for item in app.multiselect if item.label == "Tags à conserver").set_value(["Comedy", "Romance"])
        next(item for item in app.radio if item.label == "Combiner les tags à conserver").set_value("any")
        next(item for item in app.multiselect if item.label.startswith("Tags à exclure")).set_value(["NSFW"])
        self.click(app, "Appliquer les filtres")
        self.assertEqual(list(app.dataframe[0].value["ID"]), [1, 3])
        self.click(app, "NSFW")
        self.assertEqual(list(app.dataframe[0].value["ID"]), [2])
        self.assertEqual(next(item for item in app.multiselect if item.label == "Tags à conserver").value, ["NSFW"])
        self.click(app, "Appliquer les filtres")
        self.assertEqual(list(app.dataframe[0].value["ID"]), [2])
        self.click(app, "Sans tags")
        self.assertEqual(list(app.dataframe[0].value["ID"]), [4])
        self.click(app, "Réinitialiser les filtres")
        self.assertEqual(list(app.dataframe[0].value["ID"]), [1, 2, 3, 4])

    def test_missing_selected_tag_stays_selected_after_deletion(self):
        app = self.open(rows=[Entry(1, {"text": "only"}, 1, auto_tags=["Custom"])])
        next(item for item in app.multiselect if item.label == "Tags à conserver").set_value(["Custom"])
        self.click(app, "Appliquer les filtres")
        next(item for item in app.checkbox if item.label.startswith("Confirmer la suppression")).check().run()
        self.click(app, "Supprimer TOUS les résultats filtrés")
        self.assertEqual(next(item for item in app.multiselect if item.label == "Tags à conserver").value, ["Custom"])
        self.click(app, "Appliquer les filtres")
        self.assertEqual(app.session_state.filters["tags"], ["Custom"])
        self.click(app, "↶ Annuler la dernière action")
        self.assertEqual(list(app.dataframe[0].value["ID"]), [1])

    def test_catalogue_search_family_open_and_combine(self):
        app = self.open("🏷️ Tags")
        next(item for item in app.text_input if item.label == "Rechercher un tag").set_value("rom").run()
        self.assertEqual(list(app.dataframe[0].value["Tag"]), ["Romance"])
        self.click(app, "Voir ce tag dans Explorer")
        self.assertEqual(list(app.dataframe[0].value["ID"]), [1, 3])
        self.navigate(app, "🏷️ Tags")
        next(item for item in app.selectbox if item.label == "Famille de tags").set_value("Classification").run()
        self.assertTrue(set(app.dataframe[0].value["Tag"]) <= {"SFW", "NSFW", "Needs Review"})
        next(item for item in app.selectbox if item.label == "Tag à utiliser").set_value("Needs Review").run()
        self.click(app, "Ajouter au filtre à exclure")
        self.assertEqual(list(app.dataframe[0].value["ID"]), [1])
        self.assertEqual(app.session_state.filters["tags"], ["Romance"])

    def test_rename_and_remove_require_confirmation_and_undo(self):
        app = self.open("🏷️ Tags")
        original = deepcopy(app.session_state.rows)
        next(item for item in app.selectbox if item.label == "Tag à utiliser").set_value("Comedy").run()
        next(item for item in app.text_input if item.label == "Nouveau nom du tag").set_value("Romance").run()
        self.assertTrue(next(item for item in app.button if item.label == "Appliquer la modification du tag").disabled)
        next(item for item in app.checkbox if item.label.startswith("Confirmer la modification")).check().run()
        self.click(app, "Appliquer la modification du tag")
        self.assertNotIn("Comedy", entry_tags(app.session_state.rows[0]))
        self.assertIn("Romance", entry_tags(app.session_state.rows[1]))
        self.assertEqual(app.session_state.rows[4], original[4])
        self.click(app, "↶ Annuler la dernière action")
        self.assertEqual(app.session_state.rows, original)
        next(item for item in app.selectbox if item.label == "Tag à utiliser").set_value("Comedy").run()
        next(item for item in app.radio if item.label == "Action sur le tag").set_value("Retirer").run()
        next(item for item in app.checkbox if item.label.startswith("Confirmer la modification")).check().run()
        self.click(app, "Appliquer la modification du tag")
        self.assertTrue(all("Comedy" not in entry_tags(row) for row in app.session_state.rows[:4]))
        self.click(app, "↶ Annuler la dernière action")
        self.assertEqual(app.session_state.rows, original)

    def test_filtered_exports_formats_and_invalidations(self):
        app = self.open()
        next(item for item in app.multiselect if item.label == "Tags à conserver").set_value(["Romance"])
        self.click(app, "Appliquer les filtres")
        self.click(app, "Préparer l'export filtré")
        _, blob, count = app.session_state.export_filtered
        self.assertEqual(count, 2)
        self.assertTrue(all("Romance" in json.loads(line)["tags"] for line in blob.splitlines()))
        next(item for item in app.radio if item.label == "Format de l'export filtré").set_value("Unsloth / ChatML").run()
        self.assertEqual(len(app.get("download_button")), 0)
        self.click(app, "Préparer l'export filtré")
        self.assertEqual(app.session_state.export_filtered[2], 1)
        self.assertEqual(list(json.loads(app.session_state.export_filtered[1])), ["messages"])
        self.click(app, "NSFW")
        self.assertNotIn("export_filtered", app.session_state)
        self.click(app, "Préparer l'export filtré")
        next(item for item in app.text_input if item.label.startswith("Corriger les tags")).set_value("Custom")
        self.click(app, "Enregistrer les tags")
        self.assertNotIn("export_filtered", app.session_state)

    def test_export_contains_all_pages_and_does_not_remove_other_entries(self):
        rows = [Entry(uid, {"prompt": f"Q{uid}", "completion": "A"}, uid,
                      auto_tags=["Romance"] if uid <= 75 else ["Comedy"])
                for uid in range(1, 81)]
        app = self.open(rows=rows)
        next(item for item in app.multiselect if item.label == "Tags à conserver").set_value(["Romance"])
        self.click(app, "Appliquer les filtres")
        self.assertEqual(len(app.dataframe[0].value), 50)
        self.click(app, "Préparer l'export filtré")
        _, blob, count = app.session_state.export_filtered
        self.assertEqual(count, 75)
        self.assertEqual([json.loads(line)["prompt"] for line in blob.splitlines()],
                         [f"Q{uid}" for uid in range(1, 76)])
        self.assertEqual(app.session_state.rows, rows)

    def test_bulk_confirmation_resets_when_replacement_changes(self):
        app = self.open("🏷️ Tags")
        next(item for item in app.selectbox if item.label == "Tag à utiliser").set_value("Comedy").run()
        next(item for item in app.text_input if item.label == "Nouveau nom du tag").set_value("Custom").run()
        next(item for item in app.checkbox if item.label.startswith("Confirmer la modification")).check().run()
        self.assertFalse(next(item for item in app.button if item.label == "Appliquer la modification du tag").disabled)
        next(item for item in app.text_input if item.label == "Nouveau nom du tag").set_value("Other").run()
        self.assertFalse(next(item for item in app.checkbox if item.label.startswith("Confirmer la modification")).value)
        self.assertTrue(next(item for item in app.button if item.label == "Appliquer la modification du tag").disabled)


if __name__ == "__main__":
    unittest.main()
