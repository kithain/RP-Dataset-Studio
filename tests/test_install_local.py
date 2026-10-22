import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import install_local as installer
from install_assets import model_path, resolve_setting, runtime_path


def digest(data):
    return hashlib.sha256(data).hexdigest()


class TestDownloads(unittest.TestCase):
    def test_valid_asset_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.bin"
            path.write_bytes(b"valid")
            with patch("install_local.urllib.request.urlopen") as http:
                installer.download("https://example.invalid", path, digest(b"valid"), 5)
            http.assert_not_called()

    def test_download_is_verified_before_replacing_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            source.write_bytes(b"new verified bytes")
            target.write_bytes(b"old asset")
            installer.download(source.as_uri(), target, digest(source.read_bytes()), source.stat().st_size)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertFalse(target.with_name("target.part").exists())

    def test_bad_hash_or_short_download_keeps_original_asset(self):
        for expected, size in ((digest(b"other"), 3), (digest(b"bad"), 100)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, target = root / "source", root / "target"
                source.write_bytes(b"bad")
                target.write_bytes(b"original")
                with self.assertRaises(RuntimeError):
                    installer.download(source.as_uri(), target, expected, size, attempts=1)
                self.assertEqual(target.read_bytes(), b"original")

    def test_transient_network_failure_is_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "asset"
            with patch("install_local.urllib.request.urlopen", side_effect=[OSError("offline"), io.BytesIO(b"ok")]) as http, \
                    patch("install_local.time.sleep"):
                installer.download("https://example.invalid", target, digest(b"ok"), 2)
            self.assertEqual(http.call_count, 2)
            self.assertEqual(target.read_bytes(), b"ok")


class TestRuntimeInstall(unittest.TestCase):
    def package(self, root, extra=None):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("llama-server.exe", b"test executable")
            archive.writestr("ggml-cpu.dll", b"required library")
            if extra:
                archive.writestr(*extra)
        payload = data.getvalue()
        path = root / ".runtimes/downloads/test.zip"
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        return patch.dict(installer.RUNTIME_ASSETS, {"vulkan": ("test.zip", digest(payload), len(payload))})

    def test_installs_all_libraries_and_reuses_verified_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.package(root):
                server = installer.install_runtime(root, "vulkan")
                self.assertTrue(server.is_file())
                self.assertTrue((server.parent / "ggml-cpu.dll").is_file())
                with patch("install_local.download") as download:
                    self.assertEqual(installer.install_runtime(root, "vulkan"), server)
                download.assert_not_called()

    def test_missing_library_is_repaired_from_cached_archive_without_removing_user_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.package(root):
                server = installer.install_runtime(root, "vulkan")
                (server.parent / "ggml-cpu.dll").unlink()
                (server.parent / "my-notes.txt").write_text("keep", encoding="utf-8")
                with patch("install_local.urllib.request.urlopen") as http:
                    installer.install_runtime(root, "vulkan")
                http.assert_not_called()
                self.assertEqual((server.parent / "ggml-cpu.dll").read_bytes(), b"required library")
                self.assertEqual((server.parent / "my-notes.txt").read_text(), "keep")

    def test_archive_cannot_escape_root_or_create_links(self):
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        for name in ("../outside", "/absolute", "C:/outside", "..\\outside", link):
            with self.subTest(name=str(name)), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.package(root, (name, b"bad")), self.assertRaises(ValueError):
                    installer.install_runtime(root, "vulkan")
                self.assertFalse((runtime_path(root) / "llama-server.exe").exists())

    def test_tampered_or_invalid_manifest_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / ".install_manifest.json"
            for value in ("bad json", "[]", '{}', '{"archive_sha256":"expected","files":{"../outside":"x"}}'):
                manifest.write_text(value, encoding="utf-8")
                self.assertFalse(installer.runtime_valid(root, "expected"))

    def test_identical_locked_runtime_files_do_not_need_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.package(root):
                server = installer.install_runtime(root, "vulkan")
                (server.parent / ".install_manifest.json").unlink()
                replace = installer.os.replace
                def locked(source, target):
                    if Path(target).suffix in (".exe", ".dll"):
                        raise PermissionError("runtime in use")
                    return replace(source, target)
                with patch("install_local.os.replace", side_effect=locked):
                    installer.install_runtime(root, "vulkan")
                self.assertTrue((server.parent / ".install_manifest.json").exists())


class TestCppRuntime(unittest.TestCase):
    def test_cpp_libraries_are_bundled_locally_and_verified_on_reinstall(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / ".runtimes/python"
            numpy = python / "Lib/site-packages/numpy.libs"
            numpy.mkdir(parents=True)
            (numpy / "msvcp140-example.dll").write_bytes(b"cpp")
            (python / "vcruntime140.dll").write_bytes(b"vc")
            (python / "vcruntime140_1.dll").write_bytes(b"vc1")
            server = runtime_path(root) / "llama-server.exe"
            server.parent.mkdir(parents=True)
            installer.write_json(server.parent / ".install_manifest.json", {"files": {}, "archive_sha256": "test"})
            installer.ensure_cpp_runtime(root, server)
            self.assertEqual((server.parent / "msvcp140.dll").read_bytes(), b"cpp")
            self.assertEqual((server.parent / "vcruntime140.dll").read_bytes(), b"vc")
            manifest = json.loads((server.parent / ".install_manifest.json").read_text())
            self.assertEqual(manifest["files"]["vcruntime140_1.dll"], digest(b"vc1"))
            with patch("install_local.shutil.copyfile") as copy:
                installer.ensure_cpp_runtime(root, server)
            copy.assert_not_called()

    def test_missing_cpp_library_stops_installation_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "relancer install.bat"):
                installer.ensure_cpp_runtime(root, root / "server.exe")


class TestStandaloneConfiguration(unittest.TestCase):
    def test_default_paths_are_relative_and_survive_folder_move(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            installer.configure(root, model_path(root), runtime_path(root) / "llama-server.exe")
            settings = json.loads((root / "qwen_settings.json").read_text(encoding="utf-8"))
            self.assertFalse(Path(settings["qwen_model_2b"]).is_absolute())
            moved = root / "another location"
            self.assertEqual(Path(resolve_setting(settings["qwen_model_2b"], moved)), model_path(moved))
            self.assertEqual(Path(resolve_setting(settings["qwen_server"], moved)), runtime_path(moved) / "llama-server.exe")

    def test_existing_custom_configuration_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = {"tagging_engine": "Custom engine", "qwen_model_2b": "custom-model.gguf",
                        "qwen_server": "custom-server.exe", "extra": "keep"}
            (root / "qwen_settings.json").write_text(json.dumps(settings), encoding="utf-8")
            installer.configure(root, model_path(root), runtime_path(root) / "llama-server.exe")
            self.assertEqual(json.loads((root / "qwen_settings.json").read_text()), settings)

    def test_absolute_managed_paths_are_migrated_to_portable_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = runtime_path(root) / "llama-server.exe"
            path = root / "qwen_settings.json"
            path.write_text(json.dumps({"qwen_model_2b": str(model_path(root)), "qwen_server": str(server)}))
            installer.configure(root, model_path(root), server)
            settings = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(Path(settings["qwen_model_2b"]).is_absolute())
            self.assertFalse(Path(settings["qwen_server"]).is_absolute())

    def test_invalid_configuration_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "qwen_settings.json"
            for content in ("invalid", "[]"):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    installer.configure(root, model_path(root), runtime_path(root) / "llama-server.exe")
                self.assertEqual(path.read_text(), content)

    def test_vulkan_failure_installs_and_configures_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gpu = runtime_path(root) / "llama-server.exe"
            cpu = runtime_path(root, "cpu") / "llama-server.exe"
            installer.configure(root, model_path(root), gpu)
            with patch("install_local.download"), patch("install_local.ensure_cpp_runtime"), \
                    patch("install_local.install_runtime", side_effect=[gpu, cpu]) as runtime, \
                    patch("install_local.check_model", side_effect=[RuntimeError("GPU unavailable"), None]) as check:
                installer.install(root)
            self.assertEqual([call.args[1] for call in runtime.call_args_list], ["vulkan", "cpu"])
            self.assertEqual(check.call_count, 2)
            self.assertIn("-cpu/", json.loads((root / "qwen_settings.json").read_text())["qwen_server"])

    def test_cpu_inference_failure_does_not_announce_configured_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("install_local.download"), patch("install_local.ensure_cpp_runtime"), \
                    patch("install_local.install_runtime", return_value=root / "server"), \
                    patch("install_local.check_model", side_effect=RuntimeError("model failed")), \
                    patch("install_local.configure") as configure:
                with self.assertRaises(RuntimeError):
                    installer.install(root, backend="cpu")
            configure.assert_not_called()

    def test_cached_model_is_copied_into_app_and_has_no_external_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "external-model"
            source.write_bytes(b"valid model")
            with patch("install_local.MODEL_SIZE", 11), patch("install_local.MODEL_SHA256", digest(b"valid model")), \
                    patch("install_local.ensure_cpp_runtime"), \
                    patch("install_local.install_runtime", return_value=runtime_path(root) / "llama-server.exe"), \
                    patch("install_local.check_model"), patch("install_local.urllib.request.urlopen") as http:
                installer.install(root, model_cache=source)
            http.assert_not_called()
            source.write_bytes(b"changed")
            self.assertEqual(model_path(root).read_bytes(), b"valid model")


if __name__ == "__main__":
    unittest.main()
