from contextlib import redirect_stdout
import io
import subprocess
import unittest
from unittest.mock import Mock, patch

import launch_local as launcher


class TestLauncher(unittest.TestCase):
    def setUp(self):
        self.stop_patch = patch.object(launcher, "stop_previous_instances", return_value=False)
        self.stop = self.stop_patch.start()
        self.addCleanup(self.stop_patch.stop)

    def test_existing_instance_is_stopped_before_a_new_server_starts(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        order = []
        self.stop.side_effect = lambda: order.append("stop") or True
        with patch.object(launcher, "server_ready", return_value=True), \
                patch.object(launcher, "port_in_use", return_value=False), \
                patch.object(launcher.subprocess, "Popen", side_effect=lambda *a, **k: order.append("start") or process), \
                patch.object(launcher.webbrowser, "open", return_value=True):
            self.assertEqual(launcher.launch(), 0)
        self.assertEqual(order, ["stop", "start"])

    def test_busy_non_streamlit_port_never_opens_browser_or_stops_processes(self):
        with patch.object(launcher, "server_ready", return_value=False), \
                patch.object(launcher, "port_in_use", return_value=True), \
                patch.object(launcher.subprocess, "Popen") as spawn, \
                patch.object(launcher.webbrowser, "open") as browser:
            self.assertEqual(launcher.launch(), 1)
        spawn.assert_not_called()
        browser.assert_not_called()

    def test_new_server_opens_browser_only_after_readiness(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        with patch.object(launcher, "server_ready", side_effect=[False, True]), \
                patch.object(launcher, "port_in_use", return_value=False), \
                patch.object(launcher.subprocess, "Popen", return_value=process) as spawn, \
                patch.object(launcher.time, "sleep"), \
                patch.object(launcher.webbrowser, "open", return_value=True) as browser:
            self.assertEqual(launcher.launch(), 0)
        command = spawn.call_args.args[0]
        self.assertEqual(command[:4], [launcher.sys.executable, "-m", "streamlit", "run"])
        self.assertEqual(command[4], str(launcher.BASE / "app.py"))
        self.assertEqual(spawn.call_args.kwargs["cwd"], launcher.BASE)
        self.assertEqual(command[command.index("--server.headless") + 1], "true")
        browser.assert_called_once()
        process.wait.assert_called_once()
        process.terminate.assert_not_called()

    def test_startup_failure_does_not_open_browser(self):
        process = Mock(returncode=2)
        process.poll.return_value = 2
        with patch.object(launcher, "server_ready", return_value=False), \
                patch.object(launcher, "port_in_use", return_value=False), \
                patch.object(launcher.subprocess, "Popen", return_value=process), \
                patch.object(launcher.webbrowser, "open") as browser:
            self.assertEqual(launcher.launch(), 2)
        browser.assert_not_called()

    def test_slow_start_keeps_waiting_and_opens_browser_when_ready(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        output = io.StringIO()
        with patch.object(launcher, "server_ready", side_effect=[False, True]), \
                patch.object(launcher, "port_in_use", return_value=False), \
                patch.object(launcher.subprocess, "Popen", return_value=process), \
                patch.object(launcher.time, "sleep"), \
                patch.object(launcher.webbrowser, "open", return_value=True) as browser, redirect_stdout(output):
            self.assertEqual(launcher.launch(startup_timeout=0), 0)
        browser.assert_called_once()
        process.terminate.assert_not_called()
        self.assertIn(launcher.URL, output.getvalue())

    def test_ctrl_c_only_stops_the_child_created_by_launcher(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [KeyboardInterrupt, subprocess.TimeoutExpired("test", 5), 0]
        with patch.object(launcher, "server_ready", return_value=True), \
                patch.object(launcher, "port_in_use", return_value=False), \
                patch.object(launcher.subprocess, "Popen", return_value=process), \
                patch.object(launcher.webbrowser, "open", return_value=True):
            self.assertEqual(launcher.launch(), 0)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()

    def test_browser_failure_displays_url(self):
        for result in (False, OSError("No browser")):
            with self.subTest(result=result):
                output = io.StringIO()
                browser = Mock(side_effect=result) if isinstance(result, OSError) else Mock(return_value=result)
                with patch.object(launcher.webbrowser, "open", browser), redirect_stdout(output):
                    launcher.open_interface()
                self.assertIn("Ouvre cette adresse", output.getvalue())
                self.assertIn(launcher.URL, output.getvalue())

    def test_health_requires_ok_and_bypasses_proxy(self):
        for status, body, expected in ((200, b"ok", True), (200, b"other application", False), (503, b"ok", False)):
            with self.subTest(status=status, body=body):
                response = Mock(status=status)
                response.read.return_value = body
                response.__enter__ = Mock(return_value=response)
                response.__exit__ = Mock(return_value=False)
                opener = Mock()
                opener.open.return_value = response
                with patch.object(launcher.urllib.request, "build_opener", return_value=opener) as build:
                    self.assertEqual(launcher.server_ready(), expected)
                self.assertEqual(build.call_args.args[0].proxies, {})
                opener.open.assert_called_once_with(launcher.URL + "/_stcore/health", timeout=1)


class TestProcessCleanup(unittest.TestCase):
    def record(self, pid, parent, args, executable=None):
        return {"ProcessId": pid, "ParentProcessId": parent,
                "ExecutablePath": executable or args[0], "CommandLine": subprocess.list2cmdline(args)}

    def test_only_current_project_instances_are_selected(self):
        local_python = str(launcher.BASE / ".venv" / "Scripts" / "python.exe")
        external_python = r"C:\Python\python.exe"
        rows = [self.record(101, 0, [local_python, "-m", "streamlit", "run", "app.py"]),
                self.record(102, 101, [external_python, "-m", "streamlit", "run", "app.py"]),
                self.record(201, 0, [external_python, "-m", "streamlit", "run", str(launcher.BASE / "app.py")]),
                self.record(301, 0, [external_python, "-m", "streamlit", "run", r"C:\Other project\app.py"]),
                self.record(302, 0, [external_python, "-m", "streamlit", "run", "app.py"]),
                self.record(303, 0, [local_python, "-m", "streamlit", "run", "examples/app.py"]),
                self.record(401, 0, [local_python, "launch_local.py"]),
                self.record(launcher.os.getpid(), 0, [local_python, "launch_local.py"])]
        self.assertEqual(launcher.project_process_ids(rows), {101, 201, 401})

    def test_paths_with_spaces_and_accents_are_parsed(self):
        with patch.object(launcher, "BASE", launcher.BASE / "copie avec espaces é"):
            args = [r"C:\Python\python.exe", "-m", "streamlit", "run", str(launcher.BASE / "app.py")]
            self.assertEqual(launcher.project_process_ids([self.record(901, 0, args)]), {901})

    def test_taskkill_targets_old_root_and_its_tree_once(self):
        python = str(launcher.BASE / ".runtimes" / "python" / "python.exe")
        rows = [self.record(101, 0, [python, "launch_local.py"]),
                self.record(102, 101, [python, "-m", "streamlit", "run", str(launcher.BASE / "app.py")]),
                self.record(103, 102, [str(launcher.BASE / ".runtimes" / "llama-server.exe"), "-m", "model.gguf"])]
        with patch.object(launcher, "process_snapshot", return_value=rows), \
                patch.object(launcher.subprocess, "run", return_value=Mock(returncode=0)) as terminate:
            self.assertTrue(launcher.stop_previous_instances())
        self.assertEqual(terminate.call_count, 1)
        self.assertEqual(terminate.call_args.args[0], ["taskkill.exe", "/PID", "101", "/T", "/F"])

    def test_failed_stop_blocks_restart_if_old_instance_is_still_present(self):
        python = str(launcher.BASE / ".runtimes" / "python" / "python.exe")
        rows = [self.record(101, 0, [python, "launch_local.py"])]
        with patch.object(launcher, "process_snapshot", return_value=rows), \
                patch.object(launcher.subprocess, "run", return_value=Mock(returncode=1)):
            with self.assertRaisesRegex(RuntimeError, "Impossible de fermer"):
                launcher.stop_previous_instances()

    def test_instance_that_exits_during_cleanup_does_not_block_restart(self):
        python = str(launcher.BASE / ".runtimes" / "python" / "python.exe")
        rows = [self.record(101, 0, [python, "launch_local.py"])]
        with patch.object(launcher, "process_snapshot", side_effect=[rows, []]), \
                patch.object(launcher.subprocess, "run", return_value=Mock(returncode=1)):
            self.assertTrue(launcher.stop_previous_instances())


if __name__ == "__main__":
    unittest.main()
