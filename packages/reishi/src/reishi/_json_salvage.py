"""Salvage a JSON object from noisy model output.

Kept out of the codec: recovering the first balanced {...} from generation that
carries junk around it is an inference-output concern, not part of the JSON wire
format. The json codec's decode delegates here when a strict parse fails.
"""


def extract_first_json_object(s: str) -> str | None:
    """Return the first balanced {...} substring, or None if there isn't one.

    Tracks brace depth while respecting string literals and escapes, so braces
    inside JSON string values don't throw the count off.
    """
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
