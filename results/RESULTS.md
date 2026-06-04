## Korean LLM Benchmark Results

### Accuracy (Korean MCQ)

| Benchmark | qwen3-14b | exaone4-32b |
|:---|---:|---:|
| click | 66.82% | 68.30% |
| haerae | 54.64% | 63.20% |
| kmmlu | 48.30% | 52.24% |
| kmmlu_hard | 27.95% | 29.48% |
| kobest_boolq | 93.23% | 91.52% |

### Performance (GuideLLM)

| Metric | qwen3-14b | exaone4-32b |
|:---|---:|---:|
| Output tokens/sec | 26.65 | 47.74 |
| Prompt tokens/sec | 52.05 | 107.43 |
| Requests/sec | 0.19 | 0.74 |
| Mean TTFT (ms) | - | 168.33 |
| Mean ITL (ms) | - | 18.66 |