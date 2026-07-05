import json
import sys


def wants_json(flags: list[str]) -> bool:
    for i, f in enumerate(flags):
        if f in ("-o", "--output"):
            return i + 1 < len(flags) and flags[i + 1] == "json"
        if f in ("-o=json", "--output=json", "-ojson"):
            return True
    return False


def emit(data, flags: list[str], columns: list[str] | None = None) -> None:
    if wants_json(flags):
        json.dump(data, sys.stdout, indent=2, ensure_ascii=False, default=str)
        sys.stdout.write("\n")
        return

    rows = data if isinstance(data, list) else [data]
    if not rows:
        print("(none)")
        return
    if not isinstance(rows[0], dict):
        for r in rows:
            print(r)
        return

    cols = columns or list(rows[0].keys())
    table = [[_cell(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in table)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for row in table:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


def _cell(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)
