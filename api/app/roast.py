import os
import json
import httpx
import logging
import re
from typing import Optional
from .utils import extract_json_block, _clean_leading, _take_two_sentences
from .prompts import ANALYZER_PROMPT, GENERATOR_PROMPT

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_PATH = os.getenv("OLLAMA_GENERATE_PATH", "/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "phi")
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "8"))
GEN_TEMPERATURE = float(os.getenv("OLLAMA_GEN_TEMPERATURE", "1.05"))
GEN_TOP_P = float(os.getenv("OLLAMA_GEN_TOP_P", "0.92"))
GEN_REPEAT_PENALTY = float(os.getenv("OLLAMA_GEN_REPEAT_PENALTY", "1.12"))
GEN_MAX_TOKENS = int(os.getenv("OLLAMA_GEN_MAX_TOKENS", "95"))

ENGLISH_WORDS_RE = re.compile(r"\b(the|and|you|your|with|for|that|this|are|is|it|of|to)\b", re.IGNORECASE)
RIOPLATENSE_MARKERS = (
    "vos",
    "che",
    "boludo",
    "pelotudo",
    "forro",
    "nabo",
    "salame",
    "mamarracho",
    "vendehumo",
    "sos",
    "tenes",
    "laburo",
    "bardear",
    "gil",
    "culiado",
    "hdp",
)

AGGRESSIVE_MARKERS = (
    "boludo",
    "pelotudo",
    "forro",
    "salame",
    "nabo",
    "gil",
    "mamarracho",
    "payaso",
    "fantasma",
    "cara rota",
    "vendehumo",
)


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
    return not any(marker in lowered for marker in AGGRESSIVE_MARKERS)


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

            for p in paths_to_try:
                url = base + p
                try:
                    resp = await client.post(url, json=payload)
                except Exception as e:
                    last_exc = e
                    logging.debug("Failed POST %s (model=%s): %s", url, current_model, e)
                    continue

                # If endpoint not found, try next path.
                if resp.status_code == 404:
                    logging.debug("Endpoint %s returned 404, trying next", url)
                    continue

                try:
                    resp.raise_for_status()
                except Exception as e:
                    last_exc = e
                    logging.debug("Request to %s failed (model=%s): %s", url, current_model, e)
                    continue

                try:
                    parsed = resp.json()
                    logging.info("Ollama %s returned %s with model=%s", url, resp.status_code, current_model)
                    return parsed
                except Exception:
                    ndjson = _parse_ndjson_text(resp.text)
                    if ndjson is not None:
                        logging.info("Ollama %s returned NDJSON with model=%s", url, current_model)
                        return ndjson

                    logging.info("Ollama %s returned %s; text preview=%.200s", url, resp.status_code, resp.text)
                    return {"result": resp.text}

        # If we get here, all attempts failed
        logging.error("All Ollama endpoint attempts failed; last_exc=%s", last_exc)
        if last_exc:
            raise last_exc
        raise httpx.HTTPError("No working Ollama endpoint found")


async def check_ollama_health() -> bool:
    try:
        # light call
        await call_ollama("ping", temperature=0.0, max_tokens=1)
        return True
    except Exception:
        return False


async def analyze_request(target: Optional[str], raw: Optional[str], invoker: Optional[str]) -> dict:
    # Build analyzer input
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

    # normalize
    action = parsed.get("action")
    if isinstance(action, str):
        action = action.strip().lower()
        if action == "insult":
            action = "roast"

    target_res = parsed.get("target")
    reason = parsed.get("reason")
    return {"action": action or "clarify", "target": target_res, "reason": reason}


async def generate_response(action: str, target: Optional[str], raw: Optional[str], invoker: Optional[str]) -> str:
    # Build generation prompt depending on action
    if action == "respond":
        context = f'Action: respond\nTarget: "{target}"\nOriginal message: "{raw or ""}"'
    else:
        context = f'Action: roast\nTarget: "{target or invoker}"'

    style_boost = (
        "REGLAS FINALES OBLIGATORIAS: Espanol rioplatense puro con voseo, nada de ingles. "
        "Agresividad 9/10, lenguaje crudo y callejero, sarcasmo filoso. "
        "Menciona al objetivo por nombre en la primera oracion. "
        "Maximo 2 oraciones. Sin introducciones, sin explicaciones, sin disculpas."
    )
    prompt = GENERATOR_PROMPT + "\n\n" + style_boost + "\n\n" + context

    try:
        data = await call_ollama(
            prompt,
            temperature=GEN_TEMPERATURE,
            max_tokens=GEN_MAX_TOKENS,
            top_p=GEN_TOP_P,
            repeat_penalty=GEN_REPEAT_PENALTY,
        )
    except Exception:
        return "No mood to roast right now."

    text = _extract_text_from_response(data)
    if text is None:
        text = str(data)

    if not text:
        # fallback to stringify
        try:
            text = json.dumps(data)
        except Exception:
            return "No mood to roast right now."

    # post-process
    text = _clean_leading(text)
    text = _take_two_sentences(text)

    if _needs_argentinization(text) or _needs_punch_up(text):
        rewrite_prompt = (
            "Reescribi este roast para que suene bien argentino y mucho mas agresivo. "
            "Usa voseo, lunfardo y modismos argentinos naturales. "
            "Tono: brutal, picante, callejero, sin filtro pero con ingenio. "
            "Menciona al objetivo por nombre en la primera oracion. "
            "Maximo 2 oraciones. Prohibido ingles. Prohibido introducciones o explicaciones. "
            "Texto base:\n"
            f"{text}\n"
            f"Objetivo: {target or invoker or 'el objetivo'}"
        )
        try:
            rewrite_data = await call_ollama(
                rewrite_prompt,
                temperature=GEN_TEMPERATURE,
                max_tokens=GEN_MAX_TOKENS,
                top_p=GEN_TOP_P,
                repeat_penalty=GEN_REPEAT_PENALTY,
            )
            rewrite_text = _extract_text_from_response(rewrite_data)
            if rewrite_text:
                rewrite_text = _clean_leading(rewrite_text)
                rewrite_text = _take_two_sentences(rewrite_text)
                if rewrite_text:
                    text = rewrite_text
        except Exception:
            pass

    if not text:
        return "No mood to roast right now."
    return text
