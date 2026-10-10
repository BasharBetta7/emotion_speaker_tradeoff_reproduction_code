#!/usr/bin/env python3
"""Rewrite workstation-specific ESD/CREMA-D paths in packaged metadata."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def relocated(value: str, esd_root: Path | None, cremad_root: Path | None) -> str:
    normalized = value.replace("\\", "/")
    esd_marker = "/ESD_DATA_24KHz/"
    crema_marker = "/AudioWAV/"
    if esd_root is not None and esd_marker in normalized:
        return str(esd_root / normalized.split(esd_marker, 1)[1])
    if cremad_root is not None and crema_marker in normalized:
        return str(cremad_root / "AudioWAV" / normalized.split(crema_marker, 1)[1])
    return value


def rewrite_csv(path: Path, esd_root: Path | None, cremad_root: Path | None) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
        except csv.Error:
            delimiter = "\t" if sample.splitlines()[0].count("\t") > sample.splitlines()[0].count(",") else ","
        rows = list(csv.reader(handle, delimiter=delimiter))

    changed = 0
    for row in rows:
        for index, value in enumerate(row):
            new_value = relocated(value, esd_root, cremad_root)
            changed += new_value != value
            row[index] = new_value

    if changed:
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, delimiter=delimiter, lineterminator="\n").writerows(rows)
    return changed


def rewrite_text(path: Path, esd_root: Path | None, cremad_root: Path | None) -> int:
    original = path.read_text(encoding="utf-8")
    parts = original.splitlines(keepends=True)
    changed = 0
    output: list[str] = []
    for line in parts:
        ending = "\n" if line.endswith("\n") else ""
        fields = line.rstrip("\r\n").split("|")
        new_fields = []
        for value in fields:
            new_value = relocated(value, esd_root, cremad_root)
            changed += new_value != value
            new_fields.append(new_value)
        output.append("|".join(new_fields) + ending)
    if changed:
        path.write_text("".join(output), encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--esd-root", type=Path, help="Extracted 24-kHz ESD root")
    parser.add_argument("--cremad-root", type=Path, help="CREMA-D root containing AudioWAV/")
    args = parser.parse_args()
    if args.esd_root is None and args.cremad_root is None:
        parser.error("provide --esd-root, --cremad-root, or both")

    # Preserve relative roots so metadata can remain portable inside a clone.
    esd_root = args.esd_root.expanduser() if args.esd_root else None
    cremad_root = args.cremad_root.expanduser() if args.cremad_root else None
    files_changed = fields_changed = 0
    for path in sorted(args.data_dir.glob("*")):
        if path.suffix.lower() in (".csv", ".tsv") and path.stat().st_size:
            count = rewrite_csv(path, esd_root, cremad_root)
        elif path.suffix.lower() == ".txt" and path.stat().st_size:
            count = rewrite_text(path, esd_root, cremad_root)
        else:
            continue
        files_changed += bool(count)
        fields_changed += count
    print(f"Rewrote {fields_changed} path fields in {files_changed} files under {args.data_dir}.")


if __name__ == "__main__":
    main()
