from dataclasses import asdict, fields, make_dataclass
import io
import json
import logging
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import activity_log
from activity_log import Activity, event, shared_lock
from core import Entry
from finetuning import Runner, create_job, request_pause, run_logged_job
from llm_tagging import LocalQwen, QwenConfig, QwenUnavailable, generate_json
from qwen_ui import ClientPool, configured_client


class TestActivityLog(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.logger = logging.Logger("activity-test", level=logging.INFO)
        self.handler = logging.StreamHandler(self.output)
        self.logger.addHandler(self.handler)
        self.patch = patch("activity_log.get_logger", return_value=self.logger)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.handler.close)

    def test_heartbeat_stops_when_the_operation_finishes(self):
        seen = threading.Event()
        original = self.handler.emit
        def capture(record):
            original(record)
            if "EN COURS" in record.getMessage():
                seen.set()
        self.handler.emit = capture
        with Activity("Réponse en attente", interval=0.01, entree=7) as operation:
            self.assertTrue(seen.wait(2))
        operation.thread.join(2)
        self.assertFalse(operation.thread.is_alive())
        lines = self.output.getvalue().splitlines()
        self.assertIn("DÉBUT", lines[0])
        self.assertTrue(any("EN COURS" in line and "entree=7" in line for line in lines))
        self.assertIn("FIN", lines[-1])

    def test_errors_are_logged_and_still_propagate(self):
        with self.assertRaisesRegex(TimeoutError, "Délai dépassé"):
            with Activity("Requête", heartbeat=False):
                raise TimeoutError("Délai dépassé")
        self.assertIn("ÉCHEC", self.output.getvalue())
        self.assertIn("TimeoutError", self.output.getvalue())
        self.assertNotIn("FIN — Requête", self.output.getvalue())

    def test_nested_operations_keep_task_and_entry_context(self):
        with Activity("Lot", heartbeat=False, lot="demo", entree=18):
            with Activity("Génération", heartbeat=False, etape="generation"):
                event("Réponse reçue", sortie_tokens=12)
        line = next(line for line in self.output.getvalue().splitlines() if "Réponse reçue" in line)
        self.assertIn("lot=demo", line)
        self.assertIn("entree=18", line)
        self.assertIn("etape=generation", line)
        self.assertIn("sortie_tokens=12", line)

    def test_queued_request_is_identified_before_lock_is_available(self):
        queued, entered = threading.Event(), threading.Event()
        original = self.handler.emit
        def capture(record):
            original(record)
            if "DÉBUT — Attente du moteur" in record.getMessage():
                queued.set()
        self.handler.emit = capture
        lock = threading.Lock()
        lock.acquire()
        def worker():
            with shared_lock(lock):
                event("Requête lancée")
                entered.set()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            self.assertTrue(queued.wait(2))
            self.assertFalse(entered.is_set())
        finally:
            lock.release()
            thread.join(2)
        self.assertTrue(entered.is_set())
        output = self.output.getvalue()
        self.assertLess(output.index("FIN — Attente du moteur"), output.index("Requête lancée"))

    def test_request_logs_loading_usage_and_duration_without_prompt_or_response(self):
        client = LocalQwen(QwenConfig(model="demo.gguf"))
        result = {"choices": [{"finish_reason": "stop", "message": {"content": "PRIVATE_RESPONSE"}}],
                  "usage": {"prompt_tokens": 314, "completion_tokens": 17}}
        with patch.object(client, "load"), patch.object(client, "_json_http", return_value=result):
            self.assertEqual(generate_json(client, "PRIVATE_SYSTEM", "PRIVATE_SOURCE", schema={}, max_tokens=64),
                             "PRIVATE_RESPONSE")
        output = self.output.getvalue()
        for expected in ("Chargement du modèle", "Modèle prêt", "Réponse du modèle en attente",
                         "entree_tokens=314", "sortie_tokens=17", "duree_s="):
            self.assertIn(expected, output)
        for private in ("PRIVATE_SYSTEM", "PRIVATE_SOURCE", "PRIVATE_RESPONSE"):
            self.assertNotIn(private, output)

    def test_request_failure_closes_child_and_reports_the_error(self):
        client = LocalQwen(QwenConfig())
        with patch.object(client, "load"), patch.object(client, "close") as close, \
                patch.object(client, "_json_http", side_effect=TimeoutError("Serveur sans réponse")):
            with self.assertRaises(QwenUnavailable):
                generate_json(client, "system", "source", schema={}, max_tokens=64)
            close.assert_called_once()
        self.assertIn("ÉCHEC — Réponse du modèle", self.output.getvalue())
        self.assertIn("Serveur sans réponse", self.output.getvalue())

    def test_batch_preview_pause_and_resume_report_progress_without_reprocessing(self):
        text = ("Le voyageur observe la grande horloge de la gare. Il attend son train et replie sa lettre. "
                "Le quai reste vide, puis les premiers passagers arrivent avec leurs bagages. "
                "La lumière du matin traverse les verrières. ")
        rows = [Entry(i, {"prompt": "Raconte.", "completion": text + f" Il compte {i} bagages."}, i)
                for i in (1, 2)]
        def generate(system, prompt, *, schema, max_tokens):
            if "consigne" in schema["properties"]:
                return json.dumps({"consigne": "Raconte comment le voyageur attend son train près de la grande horloge de la gare en relisant sa lettre."})
            return json.dumps({"verdict": "OK", "motif": "Fidèle."})
        fake = Mock(side_effect=generate)
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(create_job(rows, "demo.jsonl", "m"), Path(directory) / "state.json")
            run_logged_job(runner, fake, 1)
            self.assertEqual(fake.call_count, 2)
            request_pause(runner)
            run_logged_job(runner, fake)
            self.assertEqual(fake.call_count, 2)
            runner.stop.clear()
            run_logged_job(runner, fake)
            self.assertEqual(fake.call_count, 4)
        output = self.output.getvalue()
        for expected in ("Lot de narration", "Génération de la consigne", "Vérification de la consigne",
                         "Progression sauvegardée", "Pause demandée", "etat=pause", "etat=terminé",
                         "Lot inactif"):
            self.assertIn(expected, output)
        self.assertNotIn(text, output)

    def test_config_reloaded_as_new_dataclass_reuses_the_loaded_client(self):
        config = QwenConfig()
        legacy_type = make_dataclass("LegacyQwenConfig", [(field.name, field.type) for field in fields(config)])
        client = SimpleNamespace(config=legacy_type(**asdict(config)), close=Mock())
        pool = ClientPool()
        pool.client = client
        with patch("qwen_ui.LocalQwen") as create:
            self.assertIs(configured_client(pool, config), client)
            create.assert_not_called()
            client.close.assert_not_called()


class TestActivityHandlers(unittest.TestCase):
    def test_console_and_rotating_file_are_configured_only_once(self):
        logger = logging.Logger("isolated-activity", level=logging.INFO)
        console = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "activity.log"
            try:
                with patch("activity_log.logging.getLogger", return_value=logger), \
                        patch("activity_log.sys.stdout", console), patch("activity_log.LOG_PATH", path):
                    self.assertIs(activity_log.get_logger(), activity_log.get_logger())
                    event("Test de journal")
                    self.assertEqual(len(logger.handlers), 2)
                    self.assertEqual(console.getvalue().count("Test de journal"), 1)
                    self.assertEqual(path.read_text(encoding="utf-8").count("Test de journal"), 1)
                    journal = next(h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler))
                    self.assertGreater(journal.maxBytes, 0)
                    self.assertGreater(journal.backupCount, 0)
            finally:
                for handler in logger.handlers:
                    handler.close()

    def test_unwritable_journal_does_not_interrupt_the_task(self):
        logger = logging.Logger("isolated-failure", level=logging.INFO)
        console = io.StringIO()
        try:
            with patch("activity_log.logging.getLogger", return_value=logger), \
                    patch("activity_log.sys.stdout", console), \
                    patch("activity_log.RotatingFileHandler", side_effect=OSError("lecture seule")):
                with Activity("Tâche", heartbeat=False):
                    event("Traitement effectué")
            output = console.getvalue()
            self.assertIn("Journal sur disque indisponible", output)
            self.assertIn("Traitement effectué", output)
            self.assertIn("FIN", output)
        finally:
            for handler in logger.handlers:
                handler.close()


if __name__ == "__main__":
    unittest.main()
