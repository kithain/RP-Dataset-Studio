"""Small, optional resource watchdog for a child owned by the application."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import threading


@dataclass(frozen=True)
class MemorySnapshot:
    ram_gib: float | None
    vram_mib: int | None


def memory_snapshot():
    ram = None
    if os.name == "nt":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available", "virtual_total",
                 "virtual_available", "extended")]
        value = Status()
        value.length = ctypes.sizeof(value)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            ram = value.available / 1024 ** 3
    else:
        try:
            values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            ram = int(values["MemAvailable"].split()[0]) / 1024 ** 2
        except (OSError, ValueError, KeyError):
            pass
    vram = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode == 0:
            vram = min(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return MemorySnapshot(ram, vram)


def shortage(snapshot, reserve_ram_gib, reserve_vram_mib):
    if reserve_ram_gib and snapshot.ram_gib is None:
        return "Mesure de RAM indisponible : chargement interrompu."
    if reserve_ram_gib and snapshot.ram_gib < reserve_ram_gib:
        return f"Réserve RAM atteinte ({snapshot.ram_gib:.1f} Gio libres, seuil {reserve_ram_gib:g} Gio)."
    if reserve_vram_mib and snapshot.vram_mib is not None and snapshot.vram_mib < reserve_vram_mib:
        return f"Réserve VRAM atteinte ({snapshot.vram_mib} Mio libres, seuil {reserve_vram_mib} Mio)."
    return None


class MemoryGuard:
    def __init__(self, reserve_ram_gib, reserve_vram_mib, on_shortage, probe=memory_snapshot):
        self.ram, self.vram, self.on_shortage = reserve_ram_gib, reserve_vram_mib, on_shortage
        self.probe, self.event = probe, threading.Event()
        self.thread = None

    def check(self):
        return shortage(self.probe(), self.ram, self.vram)

    def start(self):
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _watch(self):
        while not self.event.wait(0.5):
            try:
                reason = self.check()
            except Exception:
                reason = "Mesure mémoire interrompue : arrêt du modèle par précaution."
            if reason and not self.event.is_set():
                self.on_shortage(reason)
                return

    def close(self):
        self.event.set()
