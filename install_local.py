"""Download and configure the standalone local Qwen installation (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time
import urllib.request
import zipfile

from install_assets import (LLAMA_VERSION, MODEL_SHA256, MODEL_SIZE, MODEL_URL, RUNTIME_ASSETS,
                            model_path, resolve_setting, runtime_path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected: str, size: int, *, attempts: int = 3):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size and sha256(destination) == expected:
        print(f"Deja valide : {destination.name}", flush=True)
        return
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RP-Dataset-Studio-installer"})
            count, previous, digest = 0, -1, hashlib.sha256()
            with urllib.request.urlopen(req, timeout=60) as source, temporary.open("wb") as target:
                while block := source.read(4 * 1024 * 1024):
                    target.write(block)
                    digest.update(block)
                    count += len(block)
                    step = min(10, count * 10 // size)
                    if step != previous:
                        print(f"{destination.name} : {count / 1e6:.0f} / {size / 1e6:.0f} Mo", flush=True)
                        previous = step
            if count != size or digest.hexdigest() != expected:
                raise ValueError(f"Taille ou SHA-256 incorrect : {destination.name}")
            os.replace(temporary, destination)
            return
        except (OSError, ValueError) as exc:
            if attempt == attempts:
                raise RuntimeError(f"Telechargement incomplet apres {attempts} essais : {exc}") from exc
            print(f"Nouvel essai {attempt + 1}/{attempts} : {exc}", flush=True)
            time.sleep(1)


def checked_target(root: Path, name: str) -> Path:
    if "\\" in name or ":" in name:
        raise ValueError("Chemin d'archive invalide")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Chemin d'archive hors du dossier d'installation")
    return target


def runtime_valid(directory: Path, expected_archive: str) -> bool:
    try:
        manifest = json.loads((directory / ".install_manifest.json").read_text(encoding="utf-8"))
        files = manifest["files"]
        return (manifest["archive_sha256"] == expected_archive and isinstance(files, dict) and
                "llama-server.exe" in files and all(
                    checked_target(directory, name).is_file() and sha256(checked_target(directory, name)) == digest
                    for name, digest in files.items()))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def install_runtime(root: Path, backend: str) -> Path:
    filename, digest, size = RUNTIME_ASSETS[backend]
    directory = runtime_path(root, backend)
    if runtime_valid(directory, digest):
        print(f"Moteur {backend} deja valide.", flush=True)
        return directory / "llama-server.exe"
    archive = root / ".runtimes/downloads" / filename
    download(f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_VERSION}/{filename}",
             archive, digest, size)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=directory.parent) as temporary:
        staging = Path(temporary).resolve()
        if not staging.is_relative_to(root.resolve()):
            raise ValueError("Dossier temporaire hors de l'installation")
        with zipfile.ZipFile(archive) as package:
            for info in package.infolist():
                target = checked_target(staging, info.filename)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Lien symbolique refuse dans l'archive")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        if not (staging / "llama-server.exe").is_file():
            raise ValueError("llama-server.exe absent de l'archive")
        files = {}
        for source in staging.rglob("*"):
            if source.is_file():
                name = source.relative_to(staging).as_posix()
                target = checked_target(directory, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                files[name] = sha256(source)
                if not target.is_file() or sha256(target) != files[name]:
                    os.replace(source, target)
        write_json(directory / ".install_manifest.json", {"archive_sha256": digest, "files": files})
    return directory / "llama-server.exe"


def write_json(path: Path, value: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def ensure_cpp_runtime(root: Path, server: Path):
    """Bundle the MSVC DLLs from our portable Python and installed NumPy wheel."""
    python = root / ".runtimes/python"
    candidates = sorted((python / "Lib/site-packages/numpy.libs").glob("msvcp140*.dll"))
    if not candidates:
        raise RuntimeError("Bibliotheque MSVCP140 absente du Python portable ; relancer install.bat.")
    sources = {"msvcp140.dll": candidates[0], "vcruntime140.dll": python / "vcruntime140.dll",
               "vcruntime140_1.dll": python / "vcruntime140_1.dll"}
    manifest_path = server.parent / ".install_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, source in sources.items():
        if not source.is_file():
            raise RuntimeError(f"Bibliotheque du Python portable absente : {source.name}")
        expected = sha256(source)
        target = server.parent / name
        if not target.is_file() or sha256(target) != expected:
            temporary = target.with_name(name + ".tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        manifest["files"][name] = expected
    write_json(manifest_path, manifest)


def configure(root: Path, model: Path, server: Path):
    path = root / "qwen_settings.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(saved, dict):
        raise ValueError("qwen_settings.json invalide ; fichier conserve, corriger puis relancer.")
    saved.setdefault("tagging_engine", "Qwen 2B Heretic · rapide (GGUF)")
    current_model = saved.get("qwen_model_2b")
    if not current_model or (isinstance(current_model, str) and
                            resolve_setting(current_model, root).casefold() == str(model).casefold()):
        saved["qwen_model_2b"] = model.relative_to(root).as_posix()
    current_server = saved.get("qwen_server")
    managed_servers = {str(runtime_path(root, backend) / "llama-server.exe").casefold()
                       for backend in ("vulkan", "cpu")}
    if not current_server or (isinstance(current_server, str) and
                             resolve_setting(current_server, root).casefold() in managed_servers):
        saved["qwen_server"] = server.relative_to(root).as_posix()
    write_json(path, saved)


def check_model(model: Path, server: Path):
    from llm_tagging import LocalQwen, QwenConfig, analyze_record
    client = LocalQwen(QwenConfig(model=str(model), server=str(server)), timeout=180)
    try:
        tags = analyze_record({"text": "Two friends quietly drink tea and discuss their daily routine."}, client.generate)
        if not tags or not any(tag in tags for tag in ("SFW", "NSFW", "Needs Review")):
            raise RuntimeError("Qwen n'a pas renvoye de classification valide.")
        print("Qwen charge et inference JSON verifiee.", flush=True)
    finally:
        client.close()


def install(root: Path, *, backend: str = "vulkan", model_cache: Path | None = None):
    root = root.resolve()
    model = model_path(root)
    if (not model.exists() and model_cache is not None and model_cache.is_file() and
            model_cache.stat().st_size == MODEL_SIZE and sha256(model_cache) == MODEL_SHA256):
        model.parent.mkdir(parents=True, exist_ok=True)
        temporary = model.with_name(model.name + ".part")
        shutil.copyfile(model_cache, temporary)
        os.replace(temporary, model)
    download(MODEL_URL, model, MODEL_SHA256, MODEL_SIZE)
    server = install_runtime(root, backend)
    ensure_cpp_runtime(root, server)
    try:
        check_model(model, server)
    except Exception as exc:
        if backend != "vulkan":
            raise
        print(f"Moteur Vulkan indisponible ({exc}). Installation du moteur CPU...", flush=True)
        server = install_runtime(root, "cpu")
        ensure_cpp_runtime(root, server)
        check_model(model, server)
    configure(root, model, server)
    print("Qwen autonome installe et configure dans le dossier de l'application.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--backend", choices=["vulkan", "cpu"], default="vulkan")
    parser.add_argument("--model-cache", type=Path)
    args = parser.parse_args()
    try:
        install(args.root, backend=args.backend, model_cache=args.model_cache)
    except PermissionError as exc:
        print(f"Fichier utilise ou non accessible : {exc}. Fermer l'application puis relancer install.bat.",
              file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"Installation Qwen interrompue : {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
