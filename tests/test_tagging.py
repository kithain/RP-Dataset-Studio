from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from streamlit.testing.v1 import AppTest

from core import Entry, apply_filters, export_jsonl, export_project, load_dataset
from merge_datasets import analyze_merge, prepare_merge
from split_export import export_split_zip
from tagging import detect_tags, entry_tags, normalize_tags, source_tags, tag_entries


APP = Path(__file__).resolve().parents[1] / "app.py"


class TestTagDetection(unittest.TestCase):
    def test_character_summary_and_explicit_traits(self):
        text = """**Le personnage** : Hayase Nagatoro, une élève du secondaire tirée de
l'anime/manga *The Nagatoro Story* (Nagatoro Monogatari).
C'est la kouhai de son Senpai, une sadodere ; elle a des sentiments pour lui.
Le ton : dominante, menue mais forte, vorace, prédatrice, taquine,
qui ne quitte jamais son personnage et prend toujours le lead.
Thème déclaré : Vore.
"""
        actual = set(detect_tags({"description": text}))
        expected = {"NSFW", "Hayase Nagatoro", "The Nagatoro Story", "Nagatoro Monogatari",
                    "Anime", "Romance", "Kouhai", "Senpai dynamic", "Sadodere", "Dominant",
                    "Teasing", "Voracious", "Predatory", "Petite", "Female", "High Schooler",
                    "Vore", "Stays in Character"}
        self.assertTrue(expected.issubset(actual), expected - actual)
        # Unstated traits/genres are not guessed from character or franchise lore.
        self.assertFalse({"SFW", "Tsundere", "Athletic", "Popular", "Comedy", "Slice of Life"} & actual)

    def test_all_requested_groups_with_french_and_english_clues(self):
        tags = detect_tags({"description": "Comédie, romance et tranche de vie. Anime. "
                            "Tsundere, sadique, joueuse, séductrice, athlétique, énergique, "
                            "populaire, petite, femme, lycéenne. Roleplay immersif et descriptif."})
        self.assertTrue({"SFW", "Comedy", "Romance", "Slice of Life", "Anime", "Tsundere",
                         "Sadistic", "Playful", "Flirty", "Athletic", "Energetic", "Popular",
                         "Petite", "Female", "High Schooler", "Roleplay", "Immersive",
                         "Descriptive"}.issubset(tags))
        self.assertNotIn("NSFW", tags)

    def test_explicit_content_and_vore_take_precedence_over_sfw(self):
        for text in ("NSFW", "18+", "BDSM", "Contenu érotique", "Graphic violence",
                     "Vore", "SFW, but graphic violence is present."):
            with self.subTest(text=text):
                tags = detect_tags({"text": text, "tags": ["SFW"]})
                self.assertIn("NSFW", tags)
                self.assertNotIn("SFW", tags)
        self.assertIn("NSFW", detect_tags({"metadata": {"tags": ["nsfw"]}}))

    def test_negated_clues_and_word_boundaries(self):
        for text in ("SFW only. No NSFW. Without gore. No romance.",
                     "Sans violence explicite. Pas de contenu érotique. Vore interdit.",
                     "No graphic violence. NSFW is forbidden.",
                     "sexist Essex Essexville actionnaire carnivore."):
            with self.subTest(text=text):
                tags = detect_tags({"text": text})
                self.assertIn("SFW", tags)
                self.assertFalse({"NSFW", "Romance", "Vore", "Action"} & set(tags))

    def test_unrecognized_and_empty_records_are_unrated(self):
        for data in ({}, {"id": "NSFW"}, {"text": "12345"}, {"unknown": "Romance"},
                     {"image": "gore.png", "url": "https://example.com/nsfw"}):
            with self.subTest(data=data):
                self.assertEqual(detect_tags(data), [])

    def test_supported_conversation_schemas_and_nested_cards(self):
        for data in (
            {"messages": [{"role": "assistant", "content": "Comedy and romance"}]},
            {"conversations": [{"from": "gpt", "value": "Comedy and romance"}]},
            {"instruction": "Comedy", "input": "Context", "output": "Romance"},
            {"prompt": "Comedy", "completion": "Romance"},
            {"question": "Comedy", "answer": "Romance"},
            {"conversation": json.dumps([{"speaker": "bot", "text": "Comedy and romance"}])},
            {"data": {"description": "Comedy", "metadata": {"genre": "Romance"}}},
        ):
            with self.subTest(data=data):
                self.assertTrue({"SFW", "Comedy", "Romance"}.issubset(detect_tags(data)))

    def test_only_text_blocks_are_analyzed_and_tool_payloads_are_skipped(self):
        data = {"messages": [
            {"role": "user", "content": [{"type": "text", "text": "Comedy"},
                                          {"type": "image_url", "text": "NSFW",
                                           "image_url": {"url": "gore.png"}}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Romance"}],
             "tool_calls": [{"function": {"arguments": "Vore"}}]},
            {"role": "tool", "content": "NSFW"},
            {"from": "function", "value": "Gore"}]}
        self.assertEqual(set(detect_tags(data)), {"SFW", "Comedy", "Romance"})

    def test_explicit_names_and_sources_keep_unicode_without_a_lore_database(self):
        data = {"character": {"name": "Éloïse", "series": ["日本語", "Example Series"]},
                "text": 'Source : Univers local\nUn manga « Œuvre locale ».'}
        self.assertTrue({"Éloïse", "日本語", "Example Series", "Univers local", "Œuvre locale"}
                        .issubset(detect_tags(data)))

    def test_normalization_preserves_free_labels_and_deduplicates_aliases(self):
        self.assertEqual(normalize_tags(' #sfw ; NSFW ; comédie ; comedy ; Slice of Life ; A &amp; B'),
                         ["NSFW", "Comedy", "Slice of Life", "A & B"])
        self.assertEqual(normalize_tags('["romance", "Romance", 12, {"tag":"bad"}]'), ["Romance"])
        self.assertEqual(normalize_tags({"tag": "Comedy"}), [])

    def test_batch_is_idempotent_preserves_overrides_and_leaves_deleted_rows_untouched(self):
        rows = [Entry(1, {"text": "Comedy"}, 1, tags_override=["Custom"]),
                Entry(2, {"text": "Romance"}, 2, tags_override=[]),
                Entry(3, {"text": "Vore"}, 3, deleted=True, auto_tags=["Original"])]
        original = deepcopy(rows)
        before = tag_entries(rows)
        self.assertEqual(before, [(1, []), (2, [])])
        self.assertEqual([entry_tags(row) for row in rows], [["Custom"], [], ["Original"]])
        self.assertEqual([row.data for row in rows], [row.data for row in original])
        self.assertEqual(rows[2], original[2])
        self.assertEqual(tag_entries(rows), [])
        for uid, tags in before:
            rows[uid - 1].auto_tags = tags
        self.assertEqual(rows, original)


class TestTagStorage(unittest.TestCase):
    def test_source_tags_and_manual_clear_in_original_export(self):
        data = {"data": {"tags": ["Anime", "Comedy"]}, "text": "Hello"}
        self.assertEqual(source_tags(data), ["Anime", "Comedy"])
        row = Entry(1, deepcopy(data), 1, auto_tags=["SFW"], tags_override=[])
        blob, count = export_jsonl([row])
        loaded, errors = load_dataset(io.BytesIO(blob), "copy.jsonl")
        self.assertEqual((count, errors), (1, []))
        self.assertEqual(entry_tags(loaded[0]), [])
        self.assertEqual(loaded[0].data["data"], data["data"])
        self.assertEqual(row.data, data)

    def test_original_chatml_and_split_exports(self):
        row = Entry(1, {"prompt": "Q", "completion": "A", "tags": ["Original"]}, 1,
                    auto_tags=["SFW", "Comedy"])
        original = deepcopy(row)
        blob, _ = export_jsonl([row])
        self.assertEqual(json.loads(blob)["tags"], ["Original", "SFW", "Comedy"])
        chatml, count = export_jsonl([row], chatml=True)
        self.assertEqual(count, 1)
        self.assertEqual(list(json.loads(chatml)), ["messages"])
        archive, parts = export_split_zip([row], entries_per_file=1)
        with ZipFile(io.BytesIO(archive)) as zipped:
            self.assertEqual(zipped.read(parts[0][0]), blob)
        self.assertEqual(row, original)

    def test_project_roundtrip_including_deleted_rows_and_empty_override(self):
        rows = [Entry(1, {"text": "Comedy"}, 7, auto_tags=["SFW", "Comedy"], tags_override=[]),
                Entry(2, {"text": "Romance"}, 9, deleted=True, tags_override=["Custom"])]
        origins = {1: "a.jsonl", 2: "b.jsonl"}
        restored_origins = {}
        result, errors = load_dataset(io.BytesIO(export_project(rows, "test", origins)),
                                      "test.rpwork.json", entry_sources=restored_origins)
        self.assertEqual(result, rows)
        self.assertFalse(errors)
        self.assertEqual(restored_origins, origins)

    def test_old_project_still_loads_and_invalid_tag_annotations_are_rejected(self):
        project = {"format": "rp-dataset-studio-v1", "entries": [{"uid": 1, "data": {"text": "Q"}}]}
        rows, errors = load_dataset(io.BytesIO(json.dumps(project).encode()), "old.rpwork.json")
        self.assertEqual(rows[0].auto_tags, [])
        self.assertIsNone(rows[0].tags_override)
        self.assertFalse(errors)
        for key, value in (("auto_tags", None), ("auto_tags", "Comedy"), ("tags_override", {}),
                           ("tags_override", [1]), ("auto_tags", [""])):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                invalid = deepcopy(project)
                invalid["entries"][0][key] = value
                load_dataset(io.BytesIO(json.dumps(invalid).encode()), "bad.rpwork.json")

    def test_filters_require_all_selected_tags_and_exclude_any_forbidden_tag(self):
        rows = [Entry(1, {"text": "A"}, 1, auto_tags=["SFW", "Comedy", "Romance"]),
                Entry(2, {"text": "B"}, 2, auto_tags=["NSFW", "Comedy"]),
                Entry(3, {"text": "C", "tags": ["Romance"]}, 3),
                Entry(4, {"text": "D"}, 4, deleted=True, auto_tags=["SFW", "Comedy"])]
        ids = lambda result: [row.uid for row in result]
        self.assertEqual(ids(apply_filters(rows, tags=["comedy", "SFW"])), [1])
        self.assertEqual(ids(apply_filters(rows, excluded_tags=["NSFW", "Romance"])), [])
        self.assertEqual(ids(apply_filters(rows, contains="Comedy", excluded_tags=["NSFW"])), [1])
        self.assertEqual(ids(apply_filters(rows, contains=r"\bRomance$", use_regex=True)), [1, 3])

    def test_merge_preserves_annotations_and_does_not_share_mutable_lists(self):
        row = Entry(4, {"prompt": "Q", "completion": "A"}, 8,
                    auto_tags=["SFW", "Romance"], tags_override=["Custom"])
        sources = [("a.rpwork.json", io.BytesIO(export_project([row], "source"))),
                   ("b.jsonl", io.BytesIO(b'{"prompt":"Other Q","completion":"Other A"}\n'))]
        analysis = analyze_merge(sources)
        selection = prepare_merge(analysis, shuffle=False)
        self.assertEqual(entry_tags(selection.rows[0]), ["Custom"])
        selection.rows[0].auto_tags.append("Extra")
        selection.rows[0].tags_override.append("Edited")
        self.assertEqual(analysis.rows[0].auto_tags, ["SFW", "Romance"])
        self.assertEqual(analysis.rows[0].tags_override, ["Custom"])


class TestTaggingUI(unittest.TestCase):
    def open_page(self, rows, page="🔎 Explorer"):
        app = AppTest.from_file(str(APP), default_timeout=60)
        app.session_state.tagging_engine = "Règles rapides (sans LLM)"
        app.session_state.rows = rows
        app.session_state.dataset_name = "test.jsonl"
        app.run()
        if page != "🔎 Explorer":
            self.navigate(app, page)
        self.assertFalse(app.exception)
        return app

    def navigate(self, app, page):
        next(r for r in app.radio if r.label == "Navigation").set_value(page).run()

    def click(self, app, label):
        next(b for b in app.button if b.label == label).click().run()
        self.assertFalse(app.exception)

    def undo(self, app):
        self.click(app, "↶ Annuler la dernière action")

    def test_import_enabled_disabled_and_project_annotations_are_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            payload = b'{"prompt":"Comedy","completion":"Romance"}\n'
            path.write_bytes(payload)
            app = AppTest.from_file(str(APP), default_timeout=60)
            app.session_state.tagging_engine = "Règles rapides (sans LLM)"
            app.run()
            next(t for t in app.text_input if t.label == "Chemin du fichier sur ce PC").set_value(str(path)).run()
            self.click(app, "Ouvrir le chemin local")
            self.assertEqual(set(entry_tags(app.session_state.rows[0])), {"SFW", "Comedy", "Romance"})
            next(c for c in app.checkbox if c.key == "auto_tags_import").uncheck().run()
            self.click(app, "Ouvrir le chemin local")
            self.assertEqual(app.session_state.rows[0].auto_tags, [])
            saved = Entry(1, {"text": "Comedy"}, 1, auto_tags=["Saved"], tags_override=[])
            project_path = Path(directory) / "saved.rpwork.json"
            project_path.write_bytes(export_project([saved], "test"))
            next(c for c in app.checkbox if c.key == "auto_tags_import").check().run()
            next(t for t in app.text_input if t.label == "Chemin du fichier sur ce PC").set_value(str(project_path)).run()
            self.click(app, "Ouvrir le chemin local")
            self.assertEqual(app.session_state.rows, [saved])
            self.assertEqual(path.read_bytes(), payload)

    def test_batch_tags_undo_clear_exports_and_preserve_manual_annotations(self):
        rows = [Entry(1, {"text": "Comedy"}, 1, tags_override=["Custom"]),
                Entry(2, {"text": "Romance"}, 2),
                Entry(3, {"text": "Vore"}, 3, deleted=True)]
        original = deepcopy(rows)
        app = self.open_page(rows, "🧹 Qualité")
        app.session_state.export_original = b"stale"
        self.click(app, "Détecter / actualiser les tags")
        self.assertEqual(entry_tags(app.session_state.rows[0]), ["Custom"])
        self.assertIn("Romance", entry_tags(app.session_state.rows[1]))
        self.assertEqual(app.session_state.rows[2].auto_tags, [])
        self.assertNotIn("export_original", app.session_state)
        self.assertTrue(any("Lot de 2 entrées" in item.value for item in app.success))
        self.click(app, "Détecter / actualiser les tags")
        self.assertEqual(len(app.session_state.history), 1)
        self.undo(app)
        self.assertEqual(app.session_state.rows, original)

    def test_manual_correction_empty_override_reset_and_undo(self):
        row = Entry(1, {"text": "Comedy"}, 1, auto_tags=["SFW", "Comedy"])
        app = self.open_page([row])
        next(t for t in app.text_input if t.label.startswith("Corriger les tags")).set_value("Romance ; Custom")
        self.click(app, "Enregistrer les tags")
        self.assertEqual(entry_tags(app.session_state.rows[0]), ["Romance", "Custom"])
        self.navigate(app, "🧹 Qualité")
        self.click(app, "Détecter / actualiser les tags")
        self.assertEqual(entry_tags(app.session_state.rows[0]), ["Romance", "Custom"])
        self.navigate(app, "🔎 Explorer")
        next(t for t in app.text_input if t.label.startswith("Corriger les tags")).set_value("")
        self.click(app, "Enregistrer les tags")
        self.assertEqual(entry_tags(app.session_state.rows[0]), [])
        self.click(app, "Utiliser les tags automatiques")
        self.assertEqual(entry_tags(app.session_state.rows[0]), ["SFW", "Comedy"])
        self.assertIsNone(app.session_state.rows[0].tags_override)
        self.undo(app)
        self.assertEqual(app.session_state.rows[0].tags_override, [])
        self.undo(app)
        self.assertEqual(entry_tags(app.session_state.rows[0]), ["Romance", "Custom"])

    def test_filter_then_bulk_delete_and_undo(self):
        rows = [Entry(1, {"text": "A"}, 1, auto_tags=["SFW", "Comedy"]),
                Entry(2, {"text": "B"}, 2, auto_tags=["NSFW", "Comedy"]),
                Entry(3, {"text": "C"}, 3, auto_tags=["SFW", "Romance"])]
        app = self.open_page(rows)
        next(m for m in app.multiselect if m.label.startswith("Tags à conserver")).set_value(["Comedy"])
        next(m for m in app.multiselect if m.label.startswith("Tags à exclure")).set_value(["NSFW"])
        self.click(app, "Appliquer les filtres")
        self.assertEqual(list(app.dataframe[0].value["ID"]), [1])
        next(c for c in app.checkbox if c.label.startswith("Confirmer la suppression")).check().run()
        self.click(app, "Supprimer TOUS les résultats filtrés")
        self.assertEqual([row.deleted for row in app.session_state.rows], [True, False, False])
        self.undo(app)
        self.assertEqual([row.deleted for row in app.session_state.rows], [False, False, False])

    def test_conversation_edit_refreshes_tags_without_losing_manual_override_and_undo(self):
        row = Entry(1, {"text": "Comedy"}, 1, auto_tags=["SFW", "Comedy"], tags_override=["Custom"])
        app = self.open_page([row])
        next(r for r in app.radio if r.label == "Affichage de l'entrée").set_value("✏️ Modifier les messages").run()
        app.text_area[0].set_value("Romance")
        self.click(app, "Enregistrer la conversation")
        self.assertEqual(app.session_state.rows[0].auto_tags, ["SFW", "Romance"])
        self.assertEqual(entry_tags(app.session_state.rows[0]), ["Custom"])
        self.undo(app)
        self.assertEqual(app.session_state.rows[0].auto_tags, ["SFW", "Comedy"])
        self.assertEqual(app.session_state.rows[0].data, {"text": "Comedy"})

    def test_merge_can_add_tags_or_keep_only_existing_annotations(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                first, second = Path(directory) / "a.jsonl", Path(directory) / "b.jsonl"
                first.write_text('{"prompt":"Comedy","completion":"Hello"}\n', encoding="utf-8")
                second.write_text('{"prompt":"Romance","completion":"Bye"}\n', encoding="utf-8")
                app = AppTest.from_file(str(APP), default_timeout=60)
                app.session_state.tagging_engine = "Règles rapides (sans LLM)"
                app.run()
                self.navigate(app, "🧩 Fusionner")
                next(t for t in app.text_area if t.label == "Ou chemins locaux, un par ligne").set_value(f"{first}\n{second}").run()
                self.click(app, "Analyser les fichiers à fusionner")
                if not enabled:
                    next(c for c in app.checkbox if c.key == "merge_auto_tags").uncheck().run()
                self.click(app, "Fusionner et préparer l'export Unsloth")
                self.assertEqual([bool(row.auto_tags) for row in app.session_state.rows], [enabled, enabled])
                self.assertEqual(app.session_state.merge_report["tags_automatiques"], enabled)
                self.assertTrue(all(list(json.loads(line)) == ["messages"]
                                    for line in app.session_state.export_chatml[0].splitlines()))


if __name__ == "__main__":
    unittest.main()
