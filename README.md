# Dummy Discord Bot

## Model selection (Ollama)

You can switch models without code changes by setting `OLLAMA_MODEL` in `.env`.

Recommended for better Argentine-style roasts:

- `mistral:7b`
- `llama3`

After changing the model, recreate services so Ollama pulls and API/bot reload env:

```bash
docker compose up -d --force-recreate api bot ollama-setup
```

Then verify the active model inside API:

```bash
docker compose exec api sh -c "printenv OLLAMA_MODEL"
```
