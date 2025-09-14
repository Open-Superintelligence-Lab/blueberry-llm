from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable


def append_row_csv(path: Path, row: Dict[str, object], field_order: Iterable[str] | None = None) -> None:
    """Append a row to CSV, creating file with header if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if field_order is None:
        fieldnames = list(row.keys())
    else:
        fieldnames = list(field_order)

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

