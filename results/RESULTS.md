## Korean LLM Benchmark Results

### Accuracy (Korean MCQ)

| Benchmark | qwen3-14b | exaone4-32b | gemma4-12b | qwen36-27b |
|:---|---:|---:|---:|---:|
| click | 66.82% | 68.30% | 73.88% | 75.90% |
| haerae | 54.64% | 63.20% | 69.87% | 60.43% |
| kmmlu | 48.30% | 52.24% | 57.51% | 62.50% |
| kmmlu_hard | 27.95% | 29.48% | 33.80% | 43.06% |
| kobest_boolq | 93.23% | 91.52% | 96.08% | 96.65% |

### Performance (GuideLLM)

| Metric | qwen3-14b | exaone4-32b | gemma4-12b | qwen36-27b |
|:---|---:|---:|---:|---:|
| Output tokens/sec | 26.65 | 47.74 | 22.86 | 11.40 |
| Prompt tokens/sec | 52.05 | 107.43 | 51.19 | 25.41 |
| Requests/sec | 0.19 | 0.74 | 0.36 | 0.18 |
| Mean TTFT (ms) | - | 168.33 | 98.78 | 235.40 |
| Mean ITL (ms) | - | 18.66 | 42.88 | 85.46 |