# Reto3 — Agente ReAct + RAG sobre Y Combinator

Agente inteligente basado en el patrón **ReAct (Reason + Act)** que responde
preguntas sobre **Y Combinator** (historia, proceso de aplicación, deal estándar,
batches, ensayos de Paul Graham, alumni notables) usando **Retrieval-Augmented
Generation** sobre una base vectorial ChromaDB, con **trazabilidad en LangSmith**
y **evaluación automática con RAGAS**.

> El sistema es **agnóstico al dominio** — puedes vaciar `data/yc/` y meter tus
> propios documentos (PDF/TXT/MD/JSON) para hacer RAG sobre lo que quieras.

```
┌────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  FastAPI   │───▶│  QueryService    │───▶│  ReActAgent     │
│  /query    │    │  (orchestration) │    │  (LangGraph)    │
└────────────┘    └──────────────────┘    └────────┬────────┘
                                                   │
                              ┌────────────────────┼───────────────┐
                              ▼                    ▼               ▼
                       retrieve_context        memo tool       ChatOpenAI
                              │
                              ▼
                       ┌─────────────┐    ┌─────────────────┐
                       │  Retriever  │───▶│  ChromaVectorStore │
                       └─────────────┘    └─────────────────┘
                                                   ▲
                                                   │
                                          ┌────────┴────────┐
                                          │  Embeddings     │
                                          │  (OpenAI / ST)  │
                                          └─────────────────┘
```

## Características

- **Patrón ReAct** con LangGraph: `Thought → Action → Observation → Final Answer`.
- **RAG completo**: ingesta (PDF, TXT, MD, JSON) → chunking configurable →
  embeddings → Chroma → retrieval semántico → inyección de contexto.
- **Observabilidad** con LangSmith (tracing automático de cada paso, latencia,
  tools y chains).
- **Evaluación RAGAS**: `faithfulness`, `answer_relevancy`, `context_precision`,
  `context_recall`.
- **API FastAPI** async con streaming de respuestas (`/query/stream`).
- **Tooling avanzado del agente**: `retrieve_context` y `memo` (scratchpad
  persistente por sesión).
- **MCP (Model Context Protocol)**: el agente puede consumir tools externas vía
  `langchain-mcp-adapters`. Incluye integración con **Tavily** (búsqueda web en
  vivo) para responder lo que el corpus estático no cubre. Opt-in y con
  degradación robusta (si el MCP falla, el agente sigue con sus tools nativas).
- **Robustez de producción**: reintentos exponenciales, manejo de timeouts,
  errores tipados, logs estructurados (structlog), dependency injection,
  configuración 12-factor.
- **Docker-ready** con `Dockerfile` + `docker-compose.yml`.
- **Tests** con `pytest` incluyendo mocks, casos edge y evaluación RAGAS.

## Estructura del proyecto

```
app/
├── agents/         # ReActAgent (LangGraph) + tools + prompts
├── api/            # FastAPI routes + dependency injection
├── config/         # pydantic-settings (12-factor)
├── core/           # logging estructurado + excepciones tipadas
├── embeddings/     # provider con retry/backoff (OpenAI o SentenceTransformers)
├── evaluation/     # harness RAGAS
├── models/         # schemas pydantic (DTOs)
├── rag/            # loaders + chunker + retriever
├── services/       # IngestionService + QueryService
├── vectorstore/    # wrapper sobre ChromaDB
└── main.py         # ASGI app + lifespan
tests/              # pytest (unit + integración con embeddings fake)
scripts/            # ingest_example, query_example, evaluate_example
data/sample/        # documentos de ejemplo
```

## Quick start

### 1) Configuración

```bash
cp .env.example .env
# editar OPENAI_API_KEY y (opcional) LANGCHAIN_API_KEY
```

### 2) Instalación local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Ingestar documentos de Y Combinator

`data/yc/` ya viene con contenido curado sobre YC (qué es YC, proceso de
aplicación, batch program, deal estándar, ensayos de Paul Graham, alumni,
FAQ). Para ingerirlo:

```bash
python -m scripts.ingest_example data/yc
```

**Opcional — descargar ensayos completos de Paul Graham** (de paulgraham.com):

```bash
python -m scripts.fetch_yc_docs           # baja 10 ensayos
python -m scripts.fetch_yc_docs --limit 3 # sólo 3
python -m scripts.ingest_example data/yc/essays
```

### 4) Levantar la API

```bash
uvicorn app.main:app --reload --port 8000
```

### 5) Consultar sobre YC

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "How much does Y Combinator invest in each startup?"}'
```

Respuesta (resumida):

```json
{
  "question": "How much does Y Combinator invest in each startup?",
  "answer": "Since 2022, YC invests $500,000 per company: $125,000 for 7% equity on a post-money SAFE, plus $375,000 on an uncapped MFN SAFE that converts at the next priced round (source: 01_about_yc.md).",
  "contexts": [
    {"source": "data/yc/01_about_yc.md", "score": 0.91, "content": "YC invests $500,000 per company..."}
  ],
  "trace": [
    {"step": 1, "thought": "I should look up YC's standard deal terms.", "action": "retrieve_context", "action_input": "Y Combinator standard deal investment amount"},
    {"step": 2, "observation": "[1] source=01_about_yc.md ... YC invests $500,000 per company under a two-part instrument ..."},
    {"step": 3, "thought": "The context covers both tranches; I can answer."}
  ],
  "elapsed_ms": 842.3,
  "run_id": "..."
}
```

Otras preguntas para probar:
- `"Who founded Y Combinator and when?"`
- `"What is Bookface?"`
- `"Summarize Paul Graham's essay 'Do Things That Don't Scale'."`
- `"Is OpenAI a YC company?"`
- `"What is default alive vs default dead?"`
- `"Which YC companies have gone public?"`

### 6) Streaming

```bash
curl -N -X POST http://localhost:8000/query/stream \
     -H "Content-Type: application/json" \
     -d '{"question": "What happens during a YC batch?"}'
```

## Endpoints

| Método | Ruta               | Descripción                                              |
|--------|--------------------|----------------------------------------------------------|
| GET    | `/health`          | Health check + estado del vector store + LangSmith       |
| POST   | `/ingest`          | Sube un archivo (multipart) — pdf/txt/md/json            |
| POST   | `/ingest/path`     | Ingesta una ruta local del servidor (batch jobs)         |
| POST   | `/query`           | Pregunta al agente; devuelve respuesta + contextos + trace |
| POST   | `/query/stream`    | Streaming token-a-token del razonamiento final           |
| POST   | `/evaluate`        | Corre RAGAS sobre un conjunto de samples                 |

Docs interactivos: `http://localhost:8000/docs`.

## Docker

```bash
docker compose up --build
```

El volumen `chroma_data` persiste los embeddings entre reinicios.

## Tests

```bash
pytest -v
```

Los tests usan **embeddings fake deterministas** (hashing) para no depender de
red ni costo de API. Cubren:

- Loaders por formato (PDF/TXT/MD/JSON) y casos edge (archivo faltante, JSON
  inválido, formato no soportado).
- Configuración inválida del chunker.
- Recuperación con vector store vacío.
- Extracción de traza ReAct desde mensajes LangGraph.
- Tool `memo` (aislamiento por sesión, limpieza).
- Agente con LLM mockeado (respuesta correcta + timeout).
- API completa con dependencies sobrescritas.
- RAGAS evaluator (validaciones + propagación de errores).

## Evaluación RAGAS

```bash
python -m scripts.evaluate_example
```

O vía API:

```bash
curl -X POST http://localhost:8000/evaluate \
     -H "Content-Type: application/json" \
     -d '{
           "samples": [
             {"question": "How much does YC invest per startup?",
              "ground_truth": "$500k total: $125k for 7% + $375k uncapped MFN SAFE."},
             {"question": "Is OpenAI a YC company?",
              "ground_truth": "No. Sam Altman cofounded OpenAI while running YC, but OpenAI never went through a YC batch."}
           ]
         }'
```

Devuelve:

```json
{
  "n_samples": 1,
  "metrics": {
    "faithfulness": 0.93,
    "answer_relevancy": 0.91,
    "context_precision": 0.85,
    "context_recall": 0.78
  },
  "per_sample": [...]
}
```

## MCP — búsqueda web con Tavily

El agente puede consumir servidores **MCP** externos como tools adicionales,
junto a `retrieve_context` y `memo`. La integración usa `langchain-mcp-adapters`
(`MultiServerMCPClient`) y carga las tools **una sola vez** al arranque (en el
lifespan de FastAPI), inyectándolas en el `ReActAgent`.

Servidor incluido: **Tavily** (búsqueda/extracción web en vivo). El prompt del
agente le instruye usar el corpus de YC primero y recurrir a la web sólo cuando
el corpus no tiene la respuesta (eventos recientes, funding actual, empresas
nuevas), citando siempre la URL.

### Activar

1. Consigue una API key en [tavily.com](https://tavily.com).
2. En tu `.env`:

```bash
MCP_ENABLED=true
TAVILY_API_KEY=tvly-tu-key
MCP_TAVILY_TRANSPORT=http   # "http" (remoto, sin Node) | "stdio" (npx tavily-mcp)
```

3. Levanta la API. En los logs verás `mcp.tools_loaded` con las tools cargadas.

**Transportes:**

- `http` (default): endpoint remoto `https://mcp.tavily.com/mcp/` — no requiere
  Node, ideal para el contenedor Docker (python-only).
- `stdio`: ejecuta `npx -y tavily-mcp@latest` localmente (requiere Node/npx).

**Robustez:** MCP es opt-in. Si `MCP_ENABLED=false`, no hay key, o la conexión
falla, el agente arranca igual y opera sólo con sus tools nativas — un fallo de
MCP nunca tumba el servicio.

> **Nota de versiones:** integrar MCP requirió `langchain-core>=0.3.36` (subido a
> `0.3.63`) y `langgraph` 0.2.76 (provee `langgraph.types.Command`). `sse-starlette`
> queda fijado en `2.1.3` porque las 3.x exigen un `starlette` incompatible con
> FastAPI 0.115.0. Todo dentro de las series existentes — sin cambios de major.

## LangSmith

Si `LANGCHAIN_TRACING_V2=true` y `LANGCHAIN_API_KEY` está configurada, **cada
ejecución del agente se traza automáticamente**:

- Árbol completo `LLM → tools → LLM → ...`
- Latencia por nodo
- Inputs/outputs de cada tool call
- Metadata custom (`run_id`, `session_id`)

Project: `LANGCHAIN_PROJECT` (default: `reto3-react-rag`).

## Decisiones de diseño

- **LangGraph** sobre LangChain Expression Language: la API `create_react_agent`
  ya implementa el bucle ReAct correctamente y produce streams de mensajes
  fáciles de instrumentar.
- **ChromaDB** local con persistencia en disco: cero infra para arrancar, fácil
  de cambiar a Qdrant/Pinecone reemplazando el wrapper en `app/vectorstore/`.
- **Composition root** en `app/api/dependencies.py`: cada singleton se construye
  con `lru_cache`, fácil de overridear en tests.
- **DTOs pydantic** separados de modelos de dominio LangChain: la API es estable
  aun si cambia la versión de LangChain.
- **Errores tipados** (`AppError` + subclases con `status_code`) traducidos a
  HTTP por un handler global.
- **Reintentos con tenacity** sólo en frontera (embeddings); la lógica de
  negocio falla rápido y limpia.

## Variables de entorno

Ver `.env.example`. Las más relevantes:

| Variable                  | Default                  | Notas                                      |
|---------------------------|--------------------------|--------------------------------------------|
| `OPENAI_API_KEY`          | —                        | requerido para LLM + embeddings OpenAI     |
| `LLM_MODEL`               | `gpt-4o-mini`            | cualquier modelo chat compatible           |
| `EMBEDDING_PROVIDER`      | `openai`                 | o `sentence-transformers` para local       |
| `CHUNK_SIZE`              | `800`                    | tamaño en caracteres                       |
| `CHUNK_OVERLAP`           | `120`                    | overlap entre chunks                       |
| `TOP_K`                   | `4`                      | chunks recuperados por consulta            |
| `CHROMA_PERSIST_DIR`      | `./.chroma`              | directorio de persistencia                 |
| `LANGCHAIN_TRACING_V2`    | `false`                  | activa LangSmith                           |
| `LANGCHAIN_API_KEY`       | —                        | api key LangSmith                          |
| `LANGCHAIN_PROJECT`       | `reto3-react-rag`        | proyecto LangSmith                         |
| `MCP_ENABLED`             | `false`                  | activa tools MCP externas en el agente     |
| `TAVILY_API_KEY`          | —                        | api key Tavily (requerida para activar MCP)|
| `MCP_TAVILY_TRANSPORT`    | `http`                   | `http` (remoto) o `stdio` (npx tavily-mcp) |

## Extender

- **Otro vector store**: implementa una clase con la misma interfaz que
  `ChromaVectorStore` y cámbiala en `app/api/dependencies.py`.
- **Otro LLM**: cambia el `_build_llm()` en `app/agents/react_agent.py` por
  cualquier modelo de chat de LangChain.
- **Más tools**: añade `StructuredTool`s en `app/agents/tools.py` y regístralos
  en `_build_graph()`.
- **Más servidores MCP**: añade su conexión en `build_mcp_connections()`
  (`app/agents/mcp_client.py`); `load_mcp_tools()` las cargará y el agente las
  usará automáticamente. `MultiServerMCPClient` soporta varios servidores a la vez.

## Licencia
MIT.
