"""Export the active dataset as numbered JSONL files in a ZIP archive."""
from __future__ import annotations

import io
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from core import Entry, as_messages, export_jsonl


def export_split_zip(rows: list[Entry], base_name: str = "dataset", *,
                     entries_per_file: int = 2000, chatml: bool = False
                     ) -> tuple[bytes, list[tuple[str, int]]]:
    """Keep record order and export rules, buffering at most one uncompressed part."""
    if isinstance(entries_per_file, bool) or not isinstance(entries_per_file, int) or entries_per_file < 1:
        raise ValueError("Le nombre d'entrées par fichier doit être un entier supérieur à zéro.")
    name = re.split(r"[/\\]", base_name)[-1]
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(name).stem).strip(" .") or "dataset"
    format_name = "unsloth" if chatml else "nettoye"
    output = io.BytesIO()
    parts = []
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        def write_part(batch: list[Entry]):
            payload, count = export_jsonl(batch, chatml=chatml)
            filename = f"{stem}_{format_name}_partie_{len(parts) + 1:04d}.jsonl"
            archive.writestr(filename, payload)
            parts.append((filename, count))

        batch = []
        for row in rows:
            if row.deleted or (chatml and as_messages(row.data) is None):
                continue
            batch.append(row)
            if len(batch) == entries_per_file:
                write_part(batch)
                batch = []
        if batch:
            write_part(batch)
    return output.getvalue(), parts
