# Arquitectura — Reto3

## Capas

```
┌────────────────────────────────────────────────────────────────┐
│ API (FastAPI)                                                   │
│  routes ─┬─ /health   /ingest   /query   /query/stream  /evaluate│
│          └─ exception handlers (AppError → JSONResponse)        │
├────────────────────────────────────────────────────────────────┤
│ Services                                                        │
│  IngestionService   QueryService    RagasEvaluator              │
├────────────────────────────────────────────────────────────────┤
│ Agents                                                          │
│  ReActAgent (LangGraph) + tools(retrieve_context, memo)         │
├────────────────────────────────────────────────────────────────┤
│ RAG                                                             │
│  Loaders ─▶ Chunker ─▶ Retriever                                │
├────────────────────────────────────────────────────────────────┤
│ Infra                                                           │
│  EmbeddingProvider   ChromaVectorStore   structlog   tenacity   │
└────────────────────────────────────────────────────────────────┘
```

## Flujos

### Ingesta
```
upload/file/path
   │
   ▼
DocumentLoader  ──▶  Chunker  ──▶  EmbeddingProvider  ──▶  Chroma
```

### Consulta (ReAct loop)
```
question
   │
   ▼
Retriever.pre-fetch (devuelve `contexts` siempre, decoupled del agente)
   │
   ▼
LangGraph create_react_agent
   │
   │  ┌──────────────────────────────────────┐
   ▼  ▼                                      │
LLM ─── tool_calls? ─── yes ──▶ run tool ────┘
   │
   no
   ▼
final answer + trace extraído de los messages
```

### Evaluación
```
samples ─▶ QueryService.answer (per sample) ─▶ Dataset(question, answer, contexts, gt)
                                                       │
                                                       ▼
                                                 RAGAS evaluate
                                                       │
                                                       ▼
                                  {faithfulness, answer_relevancy,
                                   context_precision, context_recall}
```

## Decisiones clave

1. **Singletons en composition root** (`app/api/dependencies.py`) construidos
   con `lru_cache`. Reemplazables en tests vía `app.dependency_overrides`.

2. **Pre-retrieval explícito** en `ReActAgent.arun`: la API siempre devuelve
   los chunks usados para grounding, independientemente de lo que el agente
   decida hacer internamente. Esto facilita RAGAS y debugging.

3. **DTOs pydantic** (`app/models/schemas.py`) desacoplados de LangChain.
   Si la versión de LangChain cambia internamente, la API no cambia.

4. **Traza ReAct** se reconstruye desde la lista de `messages` del state final
   de LangGraph (`_extract_answer_and_trace`). Cada `AIMessage` con `tool_calls`
   produce un `ReActStep` con Thought/Action; el siguiente `ToolMessage`
   lo completa con la Observation. El `AIMessage` final sin tool_calls es la
   Final Answer.

5. **Timeouts** se manejan a dos niveles:
   - Nivel LLM: `ChatOpenAI(timeout=...)` con retries.
   - Nivel agente: `asyncio.wait_for` sobre el grafo completo (2× el timeout
     del LLM) para protegernos de loops infinitos.

6. **LangSmith** se activa exclusivamente vía variables de entorno; el código
   de aplicación no llama a la API de LangSmith directamente — LangChain lo
   hace por nosotros.

## Manejo de errores

| Excepción               | HTTP | Cuándo                                  |
|-------------------------|------|------------------------------------------|
| `UnsupportedFormatError`| 415  | Extensión de archivo no soportada        |
| `IngestionError`        | 422  | Archivo inválido o ausente               |
| `EmbeddingError`        | 502  | Falla del provider de embeddings         |
| `RetrievalError`        | 502  | Falla del vector store                   |
| `LLMTimeoutError`       | 504  | Agente excedió el timeout                |
| `EvaluationError`       | 500  | RAGAS o entrada inválida                 |
| `AppError` (base)       | 500  | Cualquier otro caso de negocio           |

Todas son atrapadas por el handler global en `app/main.py`.
