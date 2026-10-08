# EU AI Act Risk Classifier Agent

A LangGraph + FastAPI service that classifies software products under the [EU AI Act](https://www.europarl.europa.eu/topics/en/article/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence) risk framework.

> **Educational tool only.** This project is intended for learning and demonstration purposes. It does not constitute legal advice and should not be used as the basis for EU AI Act compliance decisions.

## Risk Categories

| Category | Description |
|----------|-------------|
| `prohibited` | Banned systems — social scoring, real-time biometric surveillance, subliminal manipulation |
| `high_risk` | Critical infrastructure, employment, law enforcement, medical devices, education |
| `limited_risk` | Transparency obligations required — chatbots, emotion recognition, deepfakes |
| `minimal_risk` | All other AI — spam filters, recommendation systems, game AI |

## Architecture

```
POST /classify
      │
      ▼
LangGraph StateGraph
  START → classify (LLM) → conditional routing
                               ├── prohibited
                               ├── high_risk
                               ├── limited_risk
                               └── minimal_risk → END
```

## Setup

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
```

## Running

```bash
uv run uvicorn app:app --port 8000
```

## API

### `POST /classify`

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"product_name": "HireBot", "description": "AI that ranks job candidates based on resume and video analysis."}'
```

```json
{
  "product_name": "HireBot",
  "classification": "high_risk",
  "reasoning": "HireBot is used in employment decisions, which falls under Annex III of the EU AI Act as a high-risk application. Systems that evaluate candidates for jobs are subject to strict requirements around transparency, accuracy, and human oversight.",
  "confidence": 0.95
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

### `GET /graph`

Returns a Mermaid diagram of the classification graph.

```bash
curl http://localhost:8000/graph
```

## Observability (OpenTelemetry)

Traces are exported over OTLP/HTTP to Bastyn. FastAPI requests, LangGraph/LangChain runs (including LLM calls) and outbound HTTPX calls are instrumented automatically in `app.py`. Add these to `.env`:

```
OTEL_EXPORTER_OTLP_ENDPOINT=https://ingest.staging.bastyn.ai
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_RESOURCE_ATTRIBUTES=bastyn.agent_id=<agent-uuid>,bastyn.api_key=<bastyn-api-key>
```

If these are unset, the exporter falls back to `http://localhost:4318` and spans are silently dropped.
