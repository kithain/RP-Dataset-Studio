from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from llm_tagging import LocalQwen, QwenConfig, QwenContextError, QwenError, QwenUnavailable, generate_json
from memory_guard import MemoryGuard, MemorySnapshot, shortage


class TestMemoryGuard(unittest.TestCase):
    def test_thresholds_and_missing_sensors(self):
        self.assertIsNone(shortage(MemorySnapshot(8., 4000), 5, 1536))
        self.assertIn("RAM", shortage(MemorySnapshot(3., 4000), 5, 1536))
        self.assertIn("VRAM", shortage(MemorySnapshot(8., 1000), 5, 1536))
        self.assertIn("indisponible", shortage(MemorySnapshot(None, None), 5, 1536))
        self.assertIsNone(shortage(MemorySnapshot(8., None), 5, 1536))

    def test_monitor_signals_shortage_without_taking_request_lock(self):
        signal = threading.Event()
        guard = MemoryGuard(5, 1536, lambda reason: signal.set(), probe=lambda: MemorySnapshot(2., 8000))
        guard.start()
        self.assertTrue(signal.wait(3))
        guard.close()

    def test_large_gguf_command_and_preflight_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            model, server = Path(directory) / "m.gguf", Path(directory) / "llama-server.exe"
            model.touch()
            server.touch()
            config = QwenConfig(model=str(model), server=str(server), low_memory=True,
                                gpu_layers=32, context=4096)
            client = LocalQwen(config)
            with patch("llm_tagging.MemoryGuard") as factory, patch("llm_tagging.subprocess.Popen") as popen:
                factory.return_value.check.return_value = "Réserve RAM atteinte"
                with self.assertRaisesRegex(QwenUnavailable, "Réserve RAM"):
                    client.load()
                popen.assert_not_called()
            child = Mock()
            child.poll.return_value = None
            with patch("llm_tagging.MemoryGuard") as factory, \
                    patch("llm_tagging.subprocess.Popen", return_value=child) as popen, \
                    patch.object(client, "_json_http", return_value={}):
                factory.return_value.check.return_value = None
                client.load()
                command = popen.call_args.args[0]
                for flag in ("--no-repack", "--offline", "--no-webui", "--no-context-shift"):
                    self.assertIn(flag, command)
                for flag, value in (("--load-mode", "none"), ("-ngl", "32"), ("-c", "4096"),
                                    ("-b", "128"), ("-ub", "64"), ("--cache-ram", "1024"), ("-np", "1")):
                    self.assertEqual(command[command.index(flag) + 1], value)
                self.assertNotIn("--spec-type", command)
                client._memory_stop("Réserve VRAM atteinte")
                child.terminate.assert_called_once()
                client.close()

    def test_invalid_limits_rejected(self):
        for values in ({"gpu_layers": -1}, {"context": 100}, {"ubatch": 256, "batch": 128},
                       {"cache_mib": -1}, {"reserve_ram_gib": -1}):
            with self.subTest(values=values), self.assertRaisesRegex(QwenError, "mémoire"):
                QwenConfig(**values).validate()

    def test_context_checked_before_generation_without_truncating_source(self):
        client = LocalQwen(QwenConfig(low_memory=True, context=2048))
        with patch.object(client, "load"), patch.object(client, "close") as close, patch.object(client, "_json_http", side_effect=[
                {"prompt": "rendered"}, {"tokens": list(range(2000))}]) as http:
            with self.assertRaisesRegex(QwenContextError, "aucune troncature"):
                generate_json(client, "system", "full source", schema={}, max_tokens=320)
            self.assertEqual([call.args[0] for call in http.call_args_list], ["/apply-template", "/tokenize"])
            close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
