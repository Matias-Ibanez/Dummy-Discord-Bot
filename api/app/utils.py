import re
import json
from typing import Optional

_LEADING_RE = re.compile(r"^\s*(here('?s| you go| you)|as requested[:,\s]|here[:\s])+", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r'(?<=[\.\?\!])\s+')
_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)


def _clean_leading(s: str) -> str:
    return _LEADING_RE.sub("", s).strip()


def _take_two_sentences(s: str) -> str:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(s) if p.strip()]
    if not parts:
        return ""
    return " ".join(parts[:2])


def extract_json_block(text: str) -> Optional[dict]:
    if not text:
        return None
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    block = m.group(0)
    try:
        return json.loads(block)
    except Exception:
        # Try to make minor corrections: remove trailing commas
        try:
            cleaned = re.sub(r',\s*}', '}', block)
            cleaned = re.sub(r',\s*\]', ']', cleaned)
            return json.loads(cleaned)
        except Exception:
            return None
