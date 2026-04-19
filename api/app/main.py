from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from .roast import analyze_request, generate_response, check_ollama_health
from .prompts import ANALYZER_PROMPT, GENERATOR_PROMPT

app = FastAPI(title="Roast API")


class RoastIn(BaseModel):
    target: Optional[str] = None
    raw: Optional[str] = None
    invoker: Optional[str] = None


@app.post("/v1/roast")
async def roast_endpoint(payload: RoastIn):
    # If target is explicitly provided, prioritize direct roasting.
    if payload.target and payload.target.strip():
        text = await generate_response("roast", payload.target.strip(), payload.raw, payload.invoker)
        return {"text": text}

    # Step 1: analyze
    analysis = await analyze_request(payload.target, payload.raw, payload.invoker)
    action = analysis.get("action")
    target = analysis.get("target")

    # If clarify: per decision return 200 with clarify payload
    if action == "clarify":
        return {"action": "clarify", "reason": analysis.get("reason", "unclear")}

    # If action is missing or unknown, treat as clarify
    if action not in ("roast", "respond"):
        return {"action": "clarify", "reason": "unknown_action"}

    # If no target provided by analyzer, per policy default to invoker
    if not target:
        target = payload.invoker

    # Generate
    text = await generate_response(action, target, payload.raw, payload.invoker)
    return {"text": text}


@app.get("/health")
async def health():
    ok = await check_ollama_health()
    if not ok:
        raise HTTPException(status_code=503, detail="Ollama unavailable")
    return {"status": "ok"}
