"""Restart this project's old instances, then open a fresh local Streamlit session."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser


BASE = Path(__file__).resolve().parent
HOST, PORT = "127.0.0.1", 8501
URL = f"http://{HOST}:{PORT}"


def process_snapshot():
    script = ("$ErrorActionPreference='Stop'; "
              "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
              "@(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,"
              "ExecutablePath,CommandLine) | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, encoding="utf-8", errors="replace", timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError("Impossible de verifier les anciennes instances de l'application.")
    processes = json.loads(result.stdout) or []
    return [processes] if isinstance(processes, dict) else processes


def windows_arguments(command):
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    parse = shell32.CommandLineToArgvW
    parse.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    parse.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    argv = parse(command, ctypes.byref(count))
    if not argv:
        return []
    try:
        return [argv[index] for index in range(count.value)]
    finally:
        kernel32.LocalFree(argv)


def project_process_ids(processes):
    """Recognize absolute project scripts, or legacy relative scripts with a local Python."""
    selected = set()
    for process in processes:
        pid = process["ProcessId"]
        command = process.get("CommandLine") or ""
        if pid == os.getpid() or not any(name in command.casefold() for name in ("streamlit", "launch_local.py")):
            continue
        args = windows_arguments(command)
        script, expected = None, None
        for index, arg in enumerate(args):
            if arg.casefold() == "run" and index and index + 1 < len(args):
                if args[index - 1].casefold() in {"streamlit", "streamlit.exe"} or Path(args[index - 1]).name.casefold() == "streamlit.exe":
                    script, expected = args[index + 1], "app.py"
                    break
        if script is None and len(args) > 1 and Path(args[1]).name.casefold() == "launch_local.py":
            script, expected = args[1], "launch_local.py"
        if script is None:
            continue
        path = Path(script)
        executable = process.get("ExecutablePath")
        if path.is_absolute() and path.resolve() == BASE / expected:
            selected.add(pid)
        elif script.replace("\\", "/").casefold() in {expected, f"./{expected}"} and executable:
            if Path(executable).resolve().parent in {BASE / ".venv" / "Scripts", BASE / ".runtimes" / "python"}:
                selected.add(pid)
    return selected


def stop_previous_instances():
    processes = process_snapshot()
    selected = project_process_ids(processes)
    parents = {process["ProcessId"]: process["ParentProcessId"] for process in processes}
    roots = set(selected)
    for pid in selected:
        ancestor, seen = parents.get(pid), set()
        while ancestor and ancestor not in seen:
            if ancestor in selected:
                roots.discard(pid)
                break
            seen.add(ancestor)
            ancestor = parents.get(ancestor)
    for pid in sorted(roots):
        print(f"Fermeture de l'ancienne instance et de ses sous-processus (PID {pid})...", flush=True)
        result = subprocess.run(["taskkill.exe", "/PID", str(pid), "/T", "/F"], capture_output=True,
                                timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode and pid in project_process_ids(process_snapshot()):
            raise RuntimeError(f"Impossible de fermer l'ancienne instance PID {pid}. "
                               "Ferme sa console avant de relancer.")
    return bool(roots)


def server_ready():
    # Local requests must not depend on the machine's configured HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{URL}/_stcore/health", timeout=1) as response:
            return response.status == 200 and response.read(32).strip() == b"ok"
    except (OSError, urllib.error.URLError):
        return False


def port_in_use():
    try:
        with socket.create_connection((HOST, PORT), timeout=1):
            return True
    except OSError:
        return False


def open_interface():
    print(f"Ouverture de l'interface : {URL}", flush=True)
    try:
        opened = webbrowser.open(URL, new=2)
    except (OSError, webbrowser.Error):
        opened = False
    if not opened:
        print(f"Le navigateur n'a pas pu etre ouvert. Ouvre cette adresse : {URL}", flush=True)


def launch(*, startup_timeout=60):
    try:
        stopped = stop_previous_instances()
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"Redemarrage impossible : {exc}", flush=True)
        return 1
    deadline = time.monotonic() + (5 if stopped else 0)
    while port_in_use() and time.monotonic() < deadline:
        time.sleep(0.1)
    if port_in_use():
        print(f"Le port {PORT} reste occupe par un processus non identifie comme appartenant "
              "a cette application. Il n'a pas ete ferme.", flush=True)
        return 1
    command = [sys.executable, "-m", "streamlit", "run", str(BASE / "app.py"),
               "--server.address", HOST, "--server.port", str(PORT),
               "--server.headless", "true", "--server.maxUploadSize", "1024",
               "--browser.gatherUsageStats", "false"]
    print(f"Demarrage du serveur local : {URL}", flush=True)
    print("Pour fermer : CTRL+C dans cette fenetre.", flush=True)
    try:
        process = subprocess.Popen(command, cwd=BASE)
    except OSError as exc:
        print(f"Impossible de demarrer l'application : {exc}", flush=True)
        return 1
    try:
        deadline = time.monotonic() + startup_timeout
        warned = False
        while process.poll() is None:
            if server_ready():
                open_interface()
                return process.wait()
            if not warned and time.monotonic() >= deadline:
                print(f"Le demarrage prend plus de temps que prevu. "
                      f"L'interface {URL} s'ouvrira quand le serveur sera pret.", flush=True)
                warned = True
            time.sleep(0.25)
        print("Le serveur s'est arrete avant d'etre pret. Consulte l'erreur ci-dessus.", flush=True)
        return process.returncode or 1
    except KeyboardInterrupt:
        # Ctrl+C stops the server created by this launch.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return 0


if __name__ == "__main__":
    raise SystemExit(launch())
