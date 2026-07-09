"""Target serialization codecs, named by Task.codec.

Ported from mycelium's JSON codec: base models occasionally emit a few stray
tokens (garbled unicode, leaked pretraining text) before "snapping into" the
fine-tuned completion, so decode tolerates leading junk before the first
balanced {...} object rather than requiring a strict parse from character 0.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable


def _encode_json(d: dict[str, object]) -> str:
    return json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _extract_first_json_object(s: str) -> str | None:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        c = s[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def _decode_json(s: str) -> dict[str, object]:
    try:
        result = json.loads(s)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    obj_str = _extract_first_json_object(s)
    if obj_str is None:
        return {}
    try:
        result = json.loads(obj_str)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


@dataclass(frozen=True)
class Codec:
    encode: Callable[[Any], str]
    # decode is Any, not dict: codecs serve any output type (e.g. text tasks
    # return a str), not only extraction records.
    decode: Callable[[str], Any]


_CODECS = {
    "json": Codec(_encode_json, _decode_json),
    "text": Codec(lambda s: s, lambda s: s),
}


def get_codec(name: str) -> Codec:
    if name not in _CODECS:
        raise ValueError(
            f"unknown codec '{name}' (registered: {', '.join(sorted(_CODECS))})"
        )
    return _CODECS[name]
