# Evaluation Results

This directory stores evaluation results from LMEvalJob runs on OpenShift AI.

## Result Format

Results are stored per model in JSON format:

```
results/
├── gemma-4-E2B-it/
│   ├── kmmlu_direct_law.json
│   ├── click.json
│   └── kobest_wic.json
├── llama-3.1-8b-instruct/
│   ├── kmmlu_direct_law.json
│   └── click.json
└── ...
```

Each JSON file contains the full lm-evaluation-harness output including:
- `results`: Aggregated scores per task
- `configs`: Task configuration used
- `n-samples`: Number of samples evaluated
- `config`: Model configuration

## Extracting Results from LMEvalJob

```bash
# Get results JSON from a completed job
oc get lmevaljob <job-name> -n <namespace> -o jsonpath='{.status.results}' | python -m json.tool > results.json
```

## Comprehensive Results

For accumulated benchmark results across major open-weight models (GPT, Gemma, Llama, Phi, Qwen, etc.) with radar chart visualizations, see:

> **[evaluate-llm-on-korean-dataset](https://github.com/hyogrin/evaluate-llm-on-korean-dataset)**

That repository tracks:
- KMMLU (45 subjects)
- CLIcK (11 categories)
- HAE-RAE (6 categories)
- HRM8K (math reasoning)
- KoBALT (linguistic phenomena)
- KorMedMCQA (medical QA)
