# Scouting retrieval-augmented assistant

The FastAPI app indexes supplied reports as embeddings in Postgres/pgvector, retrieves relevant passages, and asks an LLM to answer with citations. It includes a small keyword-based evaluation harness and a fictional example report. The harness is a basic automated check, not proof of factual accuracy; review answers and citations manually. No actual scouting reports, evaluation results, deployed service, or agent tool calls are included.

```sh
export OPENAI_API_KEY=your-key
# requires Docker daemon and permission to pull images
docker compose up --build
curl -X POST http://localhost:8000/documents -H 'Content-Type: application/json' -d '{"source":"example","text":"Pitcher Alex Example favors a fastball early in the count."}'
curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' -d '{"question":"What pitch does Alex favor?"}'
python evals/evaluate.py evals/example.jsonl
```

OpenAI API calls incur charges. Bind the local service to localhost and do not index confidential reports without appropriate authorization. The example has no authentication or prompt-injection defenses sufficient for public deployment. The Docker/pgvector/LLM integration requires external services and has not been verified here. `assistant.py` remains a separate offline TF-IDF retrieval baseline.

## Verified locally (Nov 24, 2026), with no paid API calls

`test_integration_local.py` runs the real FastAPI routes against real Postgres/pgvector, replacing only the OpenAI embedding call (deterministic hashed embedder) and the chat model (a scripted client that makes a real `search_reports` tool call through `agent.py`). With three fictional reports (5 chunks), the correct report ranked first for all 3 test questions and every answer cited its source; empty questions are rejected with 422. The Docker image builds. Answers from a real OpenAI model have not been run or evaluated.

```sh
DATABASE_URL=postgresql://USER:PASSWORD@127.0.0.1:5434/scouting python -m unittest -v test_integration_local
```
