# Korean MCQ Evaluation Adapter

Custom EvalHub adapter for evaluating Korean LLMs on multiple-choice benchmarks with per-question accuracy tracking.

## Supported Benchmarks

| Benchmark | Dataset | Questions | Choices | Categories |
|-----------|---------|-----------|---------|------------|
| `click` | [EunsuKim/CLIcK](https://huggingface.co/datasets/EunsuKim/CLIcK) | 1,995 | 5 | Culture (8) + Language (3) |
| `haerae` | [HAERAE-HUB/HAE_RAE_BENCH_1.1](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH_1.1) | ~4,900 | 5 | 13 categories |
| `kmmlu` | [HAERAE-HUB/KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) | 35,030 | 4 | 45 subjects |
| `kmmlu_hard` | [HAERAE-HUB/KMMLU-HARD](https://huggingface.co/datasets/HAERAE-HUB/KMMLU-HARD) | ~1,260 | 4 | 45 subjects |
| `kobest_boolq` | [skt/kobest_v1 (boolq)](https://huggingface.co/datasets/skt/kobest_v1) | 1,842 | 2 | Binary (참/거짓) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ EvalHub                                                      │
│   ├── submits job → K8s Pod (korean-mcq-adapter image)      │
│   │                    ├── loads HF dataset                  │
│   │                    ├── formats MCQ prompts               │
│   │                    ├── async parallel LLM calls (httpx)  │
│   │                    ├── parses answers                    │
│   │                    └── reports metrics + artifacts       │
│   └── logs to MLflow                                         │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- **Async parallel processing** via `asyncio` + `httpx.AsyncClient` with configurable concurrency
- **Robust error handling**: retries with exponential backoff + jitter for 429, 503, timeouts, connection errors
- **Per-question tracking**: every question's answer, prediction, and correctness saved to CSV
- **Category-level metrics**: accuracy broken down by category and supercategory
- **OCI artifact export**: CSV, JSON summary, and Markdown report pushed as OCI artifacts

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.0 | LLM sampling temperature |
| `max_tokens` | 16 | Max tokens for LLM response |
| `concurrency` | 20 | Number of parallel LLM requests |

Example job submission:

```python
BenchmarkConfig(
    id="haerae",
    provider_id=KOREAN_PROVIDER_ID,
    parameters={"temperature": 0.0, "max_tokens": 16, "concurrency": 20},
    num_examples=2000,
)
```

## Performance

| Dataset | Questions | Sequential (~) | Parallel (concurrency=20) |
|---------|-----------|----------------|---------------------------|
| KoBEST BoolQ | 100 | ~30s | ~3s |
| HAE-RAE | 4,900 | ~15 min | ~42s |
| CLIcK | 1,995 | ~6 min | ~18s |

## MLflow Integration

### What Gets Recorded

| MLflow Tab | Recorded | Details |
|------------|----------|---------|
| **Overview** | Yes | Job ID, model name, benchmark, parameters, duration |
| **Model Metrics** | Yes | `overall_accuracy`, `category_accuracy.*`, `supercategory_accuracy.*` |
| **Artifacts** | Yes | `detailed_results.csv`, `results.json`, `DETAILED_RESULTS.md` |
| **Trace** | No | Not instrumented (adapter uses raw httpx, not OpenAI SDK) |
| **System Metrics** | No | Not enabled (`psutil` not installed in container) |

### Why Trace / System Metrics Are Not Recorded

- **Trace**: MLflow Tracing requires either the OpenAI Python SDK (auto-instrumented) or explicit `mlflow.trace()` decorators. This adapter uses `httpx.AsyncClient` directly for maximum throughput with async parallelism, which MLflow cannot auto-trace.
- **System Metrics**: Requires `psutil` package + `mlflow.enable_system_metrics_logging()`. The adapter container is kept lightweight for fast cold starts in K8s. The evaluation is I/O-bound (waiting for LLM responses), not CPU/Memory-bound, so system metrics provide limited value.

The recorded **Overview + Model Metrics** are sufficient for:
- Comparing model accuracy across benchmarks
- Tracking performance over time
- Analyzing per-category strengths/weaknesses

## File Structure

```
adapters/korean-mcq/
├── main.py              # Adapter entry point (FrameworkAdapter implementation)
├── datasets_config.py   # Dataset configurations and normalization
├── prompts.py           # MCQ prompt formatting
├── parser.py            # LLM response parsing (answer extraction)
├── provider.yaml        # EvalHub provider definition
├── requirements.txt     # Python dependencies
├── Containerfile        # Container image definition
└── mapping/
    ├── id_to_category.json    # CLIcK ID → category mapping
    └── kmmlu_category.json    # KMMLU category → supercategory mapping
```

## Building the Image

```bash
# OpenShift internal registry build
oc start-build korean-mcq-adapter --from-dir=. --follow

# Or local build
podman build -t korean-mcq-adapter:latest -f Containerfile .
```

## Error Handling

The adapter handles these failure modes gracefully:

| Error | Behavior |
|-------|----------|
| Rate Limit (HTTP 429) | Retry up to 5x with increasing delay (5s → 60s) + jitter |
| Server Overload (HTTP 503) | Retry up to 5x with exponential backoff |
| Server Error (HTTP 500) | Retry up to 5x with exponential backoff |
| Request Timeout | Retry up to 5x with exponential backoff |
| Connection Error | Retry up to 5x with exponential backoff |
| Malformed Response | Log error, mark question as FAILED, continue |
| All retries exhausted | Mark question as FAILED, continue with remaining questions |

Individual question failures never crash the entire evaluation -- they are recorded as `FAILED` in the results CSV.
