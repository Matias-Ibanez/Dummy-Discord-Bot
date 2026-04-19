import os
from dotenv import load_dotenv
import logging

load_dotenv()

ANALYZER_PROMPT = os.getenv("ANALYZER_PROMPT")
GENERATOR_PROMPT = os.getenv("GENERATOR_PROMPT")

if not ANALYZER_PROMPT:
    logging.warning("ANALYZER_PROMPT not set in env; using default analyzer prompt.")
    ANALYZER_PROMPT = (
        "You are an assistant that only analyzes user requests about insults. Output MUST be valid JSON ONLY. "
        "Return an object with keys: action (one of 'roast','respond','clarify','ignore'), target (string|null), reason (string). "
        "Prefer 'clarify' if uncertain. Be deterministic."
    )

if not GENERATOR_PROMPT:
    logging.warning("GENERATOR_PROMPT not set in env; using default generator prompt.")
    GENERATOR_PROMPT = (
        "You are a Roast Comedian. Produce short, acerbic, sarcastic insults with dark humor. "
        "Responses must be at most 2 sentences, never include leading phrases like 'Here you go' or 'As requested'. "
        "Be creative, witty, and succinct."
    )
