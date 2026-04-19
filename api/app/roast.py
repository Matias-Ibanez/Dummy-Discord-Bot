import json
import logging
import os
import re
from typing import Optional

import httpx

from .prompts import ANALYZER_PROMPT, GENERATOR_PROMPT
from .utils import _clean_leading, _take_two_sentences, extract_json_block

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_PATH = os.getenv("OLLAMA_GENERATE_PATH", "/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "phi")
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "70"))

GEN_TEMPERATURE = float(os.getenv("OLLAMA_GEN_TEMPERATURE", "0.9"))
GEN_TOP_P = float(os.getenv("OLLAMA_GEN_TOP_P", "0.88"))
GEN_REPEAT_PENALTY = float(os.getenv("OLLAMA_GEN_REPEAT_PENALTY", "1.22"))
GEN_MAX_TOKENS = int(os.getenv("OLLAMA_GEN_MAX_TOKENS", "90"))

ENGLISH_WORDS_RE = re.compile(r"\b(the|and|you|your|with|for|that|this|are|is|it|of|to)\b", re.IGNORECASE)

RIOPLATENSE_MARKERS = (
    "vos",
    "che",
    "sos",
    "tenes",
    "boludo",
    "pelotudo",
    "forro",
    "salame",
    "nabo",
    "bardo",
    "bardear",
    "cabeza de pija",
    "cerra el orto",
)

AGGRESSIVE_MARKERS = (
    "boludo",
    "pelotudo",
    "forro",
    "salame",
    "nabo",
    "gil",
    "payaso",
    "fantasma",
    "cara rota",
    "cabeza de pija",
    "cerra el orto",
    "chota",
    "puto",
    "trolo",
)

POETIC_MARKERS = (
    "espiritu",
    "suenos",
    "ancestros",
    "paz",
    "alma",
    "poesia",
    "metafora",
    "epico",
    "vibrante",
    "corazon",
)

BANNED_PHRASES = (
    "system-reminder",
    "poesia",
    "metafora",
    "vibrante",
    "ancestros",
    "alma",
)

SYSTEM_REMINDER_BLOCK_RE = re.compile(r"<system-reminder>[\s\S]*?</system-reminder>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
FENCE_RE = re.compile(r"```[\s\S]*?```")


def _parse_ndjson_text(text: str) -> Optional[dict]:
    if not text:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    chunks = []
    last_obj = None
    for line in lines:
        try:
            obj = json.loads(line)
        except Exception:
            return None

        if isinstance(obj, dict):
            last_obj = obj
            part = obj.get("response")
            if isinstance(part, str):
                chunks.append(part)

    if last_obj is None:
        return None

    combined = "".join(chunks)
    if combined:
        last_obj["response"] = combined
    return last_obj


def _extract_text_from_response(data: object) -> Optional[str]:
    if isinstance(data, dict):
        for key in ("response", "result", "output", "text", "generated", "content"):
            if key in data and isinstance(data[key], str):
                return data[key]

        if "choices" in data:
            try:
                first = data["choices"][0]
                if isinstance(first, dict):
                    text = first.get("text") or first.get("message")
                    if isinstance(text, dict) and "content" in text:
                        return text["content"]
                    if isinstance(text, str):
                        return text
            except Exception:
                return None

    if isinstance(data, str):
        return data

    return None


def _sanitize_roast_output(text: str) -> str:
    if not text:
        return ""

    cleaned = text.replace("\r", " ")
    cleaned = SYSTEM_REMINDER_BLOCK_RE.sub(" ", cleaned)

    lower_raw = cleaned.lower()
    if "<system-reminder>" in lower_raw:
        cleaned = cleaned[: lower_raw.index("<system-reminder>")]

    cleaned = FENCE_RE.sub(" ", cleaned)
    cleaned = TAG_RE.sub(" ", cleaned)

    for stopper in ("note:", "explicacion:", "explanation:"):
        idx = cleaned.lower().find(stopper)
        if idx != -1:
            cleaned = cleaned[:idx]

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _ensure_closed_sentence(text: str) -> str:
    if not text:
        return ""

    t = text.strip()
    if t.endswith((".", "!", "?")):
        return t

    last_end = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
    if last_end != -1:
        return t[: last_end + 1].strip()

    return t + "."


def _needs_argentinization(text: str) -> bool:
    if not text:
        return True

    lowered = text.lower()
    english_hits = len(ENGLISH_WORDS_RE.findall(lowered))
    has_marker = any(marker in lowered for marker in RIOPLATENSE_MARKERS)
    return english_hits >= 3 or not has_marker


def _needs_punch_up(text: str) -> bool:
    if not text:
        return True

    lowered = text.lower()
    lacks_aggressive = not any(marker in lowered for marker in AGGRESSIVE_MARKERS)
    sounds_poetic = any(marker in lowered for marker in POETIC_MARKERS)
    return lacks_aggressive or sounds_poetic


def _has_direct_insult(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in AGGRESSIVE_MARKERS)


def _is_bad_output(text: str) -> bool:
    if not text:
        return True

    lowered = text.lower()
    if any(phrase in lowered for phrase in BANNED_PHRASES):
        return True

    if _needs_argentinization(text) or _needs_punch_up(text):
        return True

    if not _has_direct_insult(text):
        return True

    if not text.strip().endswith((".", "!", "?")):
        return True

    return False


async def call_ollama(
    prompt: str,
    model: str = MODEL,
    temperature: float = 0.8,
    max_tokens: int = 120,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
) -> dict:
    base = OLLAMA_URL.rstrip("/")
    configured_path = OLLAMA_PATH if OLLAMA_PATH.startswith("/") else f"/{OLLAMA_PATH}"
    paths_to_try = [configured_path]
    if configured_path != "/api/generate":
        paths_to_try.append("/api/generate")

    model_candidates = [model]
    if isinstance(model, str) and model.lower() == "phi":
        model_candidates.append("phi3")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        last_exc = None
        for current_model in model_candidates:
            payload = {
                "model": current_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty,
                },
            }

            for path in paths_to_try:
                url = base + path
                try:
                    resp = await client.post(url, json=payload)
                except Exception as e:
                    last_exc = e
                    logging.debug("Failed POST %s (model=%s): %s", url, current_model, e)
                    continue

                if resp.status_code == 404:
                    logging.debug("Endpoint/model not found at %s (model=%s)", url, current_model)
                    continue

                try:
                    resp.raise_for_status()
                except Exception as e:
                    last_exc = e
                    logging.debug("Request to %s failed (model=%s): %s", url, current_model, e)
                    continue

                try:
                    return resp.json()
                except Exception:
                    ndjson = _parse_ndjson_text(resp.text)
                    if ndjson is not None:
                        return ndjson
                    return {"result": resp.text}

        logging.error("All Ollama endpoint attempts failed; last_exc=%s", last_exc)
        if last_exc:
            raise last_exc
        raise httpx.HTTPError("No working Ollama endpoint/model found")


async def check_ollama_health() -> bool:
    try:
        await call_ollama("ping", temperature=0.0, max_tokens=1, top_p=0.2, repeat_penalty=1.0)
        return True
    except Exception:
        return False


async def analyze_request(target: Optional[str], raw: Optional[str], invoker: Optional[str]) -> dict:
    context = {"target": target, "raw": raw, "invoker": invoker}
    prompt = ANALYZER_PROMPT + "\n\n" + json.dumps(context)

    try:
        data = await call_ollama(prompt, temperature=0.0, max_tokens=80, top_p=0.2, repeat_penalty=1.0)
    except Exception:
        return {"action": "clarify", "target": None, "reason": "analysis_timeout"}

    text = _extract_text_from_response(data)
    if text is None:
        if isinstance(data, dict) and "action" in data:
            return data
        text = json.dumps(data)

    parsed = extract_json_block(text)
    if not parsed:
        return {"action": "clarify", "target": None, "reason": "no_json"}

    action = parsed.get("action")
    if isinstance(action, str):
        action = action.strip().lower()
        if action == "insult":
            action = "roast"

    return {
        "action": action or "clarify",
        "target": parsed.get("target"),
        "reason": parsed.get("reason"),
    }


async def generate_response(action: str, target: Optional[str], raw: Optional[str], invoker: Optional[str]) -> str:
    if action == "respond":
        context = f'Action: respond\nTarget: "{target}"\nOriginal message: "{raw or ""}"'
    else:
        context = f'Action: roast\nTarget: "{target or invoker}"'

    style_boost = (
        "REGLAS FINALES OBLIGATORIAS: Espanol rioplatense puro con voseo, nada de ingles. "
        "Tono de barrio: directo, sucio, agresivo y sin poesia. "
        "Prohibido romantizar, prohibido metaforas largas, prohibido estilo epico. "
        "Menciona al objetivo por nombre en la primera oracion. "
        "Maximo 2 oraciones cortas, completas y cerradas con punto o signo final. "
        "Sin introducciones, sin explicaciones, sin disculpas. "
        "Usa al menos dos puteadas argentinas de esta lista: boludo, pelotudo, forro, salame, nabo, cabeza de pija, cerra el orto, chota, puto, trolo."
    )
    prompt = GENERATOR_PROMPT + "\n\n" + style_boost + "\n\n" + context

    try:
        data = await call_ollama(
            prompt,
            temperature=GEN_TEMPERATURE,
            max_tokens=max(GEN_MAX_TOKENS, 80),
            top_p=GEN_TOP_P,
            repeat_penalty=GEN_REPEAT_PENALTY,
        )
    except Exception:
        return "No mood to roast right now."

    text = _extract_text_from_response(data)
    if text is None:
        text = str(data)

    if not text:
        return "No mood to roast right now."

    text = _sanitize_roast_output(text)
    text = _clean_leading(text)
    text = _take_two_sentences(text)
    text = _ensure_closed_sentence(text)

    attempts = 0
    while _is_bad_output(text) and attempts < 3:
        attempts += 1
        rewrite_prompt = (
            "Genera de cero un roast argentino MUY bardero y agresivo. "
            "Usa voseo y modismos argentinos reales. "
            "Menciona al objetivo en la primera oracion. "
            "Maximo 2 oraciones cortas, completas y cerradas. "
            "Sin metaforas, sin tono literario, sin sentimentalismo. "
            "Prohibido ingles. Prohibido markdown. Prohibido explicaciones. "
            "Inclui minimo dos insultos directos de esta lista: boludo, pelotudo, forro, salame, nabo, cabeza de pija, cerra el orto, chota, puto, trolo. "
            f"Objetivo: {target or invoker or 'el objetivo'}"
        )
        try:
            rewrite_data = await call_ollama(
                rewrite_prompt,
                temperature=GEN_TEMPERATURE,
                max_tokens=max(GEN_MAX_TOKENS, 80),
                top_p=GEN_TOP_P,
                repeat_penalty=GEN_REPEAT_PENALTY,
            )
        except Exception:
            break

        rewrite_text = _extract_text_from_response(rewrite_data)
        if not rewrite_text:
            continue

        rewrite_text = _sanitize_roast_output(rewrite_text)
        rewrite_text = _clean_leading(rewrite_text)
        rewrite_text = _take_two_sentences(rewrite_text)
        rewrite_text = _ensure_closed_sentence(rewrite_text)
        if rewrite_text:
            text = rewrite_text

    if not text:
        return "No mood to roast right now."
    return text
