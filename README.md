# RHOAI LMEval Builder Lab

A hands-on workshop for running **Korean language evaluation benchmarks** on **Red Hat OpenShift AI** using the TrustyAI `LMEvalJob` Custom Resource. This lab guides you through evaluating open-weight LLMs on datasets like KMMLU, CLIcK, KoBEST, and HAE-RAE directly from your OpenShift AI environment.

## Architecture

```mermaid
flowchart LR
    User[User / Workbench] -->|oc apply| CRD[LMEvalJob CR]
    User -->|SDK / REST| EvalHub[EvalHub Service]
    CRD --> Operator[TrustyAI Operator]
    EvalHub --> Operator
    Operator --> Pod[lm-eval Pod]
    Pod -->|API call| vLLM[vLLM InferenceService]
    Pod -->|download| HF[HuggingFace Datasets]
    EvalHub -->|tracking| MLflow[MLflow]
```

**How it works:**

- **LMEvalJob (Phase 1–3):** Submit an `LMEvalJob` YAML directly to OpenShift. The TrustyAI Operator creates a Pod running [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) that downloads datasets and sends inference requests to your model.
- **EvalHub (Phase 4):** Use the [EvalHub](https://github.com/eval-hub/eval-hub) REST API / Python SDK to orchestrate evaluations across multiple frameworks (lm-eval, RAGAS, LightEval, GuideLLM, etc.) with built-in **MLflow** experiment tracking.

## Model

This workshop uses **Gemma 4 (E2B-it)** deployed on OpenShift AI via a custom vLLM ServingRuntime as the target model for evaluation. The setup notebook includes instructions for deploying the model with GPU support.

## What's Included

### 0. Setup

- **0_setup/0_model_deploy.ipynb**: Deploy a Gemma 4 model using a custom vLLM ServingRuntime with NVIDIA GPU support.
- **0_setup/1_LMEval_setup.ipynb**: Configure RBAC permissions, create secrets (HF token, SA token for OAuth), and verify cluster access for LMEvalJob.
- **0_setup/2_eval_hub_setup.ipynb**: Deploy the EvalHub service and MLflow on OpenShift, install the eval-hub-sdk, and verify connectivity.

### 1. Built-in Tasks (Phase 1)

- **1_builtin_tasks/1_builtin_task_eval.ipynb**: Run evaluations using `taskNames` — the simplest approach leveraging TrustyAI's built-in task support.

### 2. Custom Tasks (Phase 2)

- **2_custom_tasks/1_custom_task_eval.ipynb**: Run evaluations using `customTasks` with a Git source — pull any task from the lm-evaluation-harness repository for full flexibility.

### 3. LMEvalJob Benchmark (Phase 3)

- **3_lmeval_job_benchmark/1_lmeval_benchmark.ipynb**: Analyze and compare LMEvalJob evaluation results across models. Load results from JSON, aggregate by category/supercategory, and export to Markdown.
- **3_lmeval_job_benchmark/generate_report.py**: Generate a standalone HTML report with interactive charts and comparison tables from evaluation result logs.

### 4. EvalHub Benchmark (Phase 4)

- **4_eval_hub_benchmark/1_eval_hub_benchmark.ipynb**: Run evaluations through the EvalHub REST API using the Python SDK. Supports multi-benchmark jobs, automatic MLflow tracking, and centralized result management.

## Korean Benchmark Datasets

| Dataset | Description | Categories | Samples |
|---------|-------------|------------|---------|
| **KMMLU** | Korean Massive Multi-task Language Understanding | 45 subjects (STEM, HUMSS, Applied Science) | 35,030 |
| **CLIcK** | Cultural and Linguistic Intelligence in Korean | 11 categories (Culture + Language) | 1,995 |
| **KoBEST** | Korean Balanced Evaluation of Significant Tasks | WiC, CoPA, BoolQ, HellaSwag, SentiNeg | 6,100+ |
| **HAE-RAE** | Korean Language Proficiency Benchmark | 6 categories (General Knowledge, History, etc.) | 1,538 |
| **HRM8K** | HAE-RAE Math 8K (bilingual math reasoning) | Korean School Math + Prior Sets | 8,011 |

## Evaluation Results

Accumulated benchmark results across major open-weight models are maintained at:

> **[evaluate-llm-on-korean-dataset](https://github.com/hyogrin/evaluate-llm-on-korean-dataset)**

This companion repository tracks performance of models like Gemma, Llama, Phi, Qwen, and others on Korean evaluation datasets with detailed per-category breakdowns and radar chart visualizations.

## Prerequisites

- Red Hat OpenShift AI cluster with TrustyAI Operator installed
- A model deployed via KServe (vLLM runtime) with OAuth auth enabled
- `oc` CLI access to the cluster
- A Hugging Face account with access to gated models/tokenizers (e.g., `google/gemma-2b`)
- Hugging Face API token
- (Phase 4) EvalHub service and MLflow deployed on the cluster

## Quick Start

1. Clone this repo into your OpenShift AI Workbench:
   ```bash
   git clone https://github.com/hyogrin/lm-eval-builder-lab.git
   cd lm-eval-builder-lab
   ```

2. Install dependencies with [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync
   source .venv/bin/activate
   ```

3. Copy and fill in environment variables:
   ```bash
   cp sample.env .env
   # Edit .env with your values
   ```

4. Run notebooks in order:
   - `0_setup/0_model_deploy.ipynb` — Deploy Gemma 4 model (skip if already deployed)
   - `0_setup/1_LMEval_setup.ipynb` — One-time RBAC and secrets setup for LMEvalJob
   - `0_setup/2_eval_hub_setup.ipynb` — Deploy EvalHub + MLflow (for Phase 4)
   - `1_builtin_tasks/1_builtin_task_eval.ipynb` — Quick eval with built-in tasks
   - `2_custom_tasks/1_custom_task_eval.ipynb` — Full eval with any task from Git
   - `3_lmeval_job_benchmark/1_lmeval_benchmark.ipynb` — Analyze LMEvalJob results and generate reports
   - `4_eval_hub_benchmark/1_eval_hub_benchmark.ipynb` — Run evaluations via EvalHub SDK with MLflow tracking

> **Note:** `uv sync` creates a `.venv` virtual environment and installs all dependencies defined in `pyproject.toml`. To run scripts directly, use `uv run python <script>` or activate the venv with `source .venv/bin/activate`.

## Phase Comparison

| | Phase 1: Built-in Tasks | Phase 2: Custom Tasks (Git) | Phase 3: LMEvalJob Benchmark | Phase 4: EvalHub Benchmark |
|---|---|---|---|---|
| **Approach** | `taskList.taskNames` | `taskList.customTasks.source.git` + `taskNames` | Post-processing result JSONs | EvalHub SDK / REST API |
| **Task Source** | TrustyAI built-in (Tier 1/2) | Any task from lm-evaluation-harness | N/A (consumes results) | Multiple frameworks (lm-eval, RAGAS, etc.) |
| **Available Korean Tasks** | kmmlu_direct_law, kobest_wic, haerae_history, etc. | Full kmmlu (45 subjects), click (11 categories), kobest, etc. | All tasks from Phase 1 & 2 | All tasks + multi-framework support |
| **Setup Complexity** | Minimal | Requires git URL + path | Requires result JSON files | Requires EvalHub + MLflow deployment |
| **Experiment Tracking** | Manual (Pod logs) | Manual (Pod logs) | Manual (JSON export) | Built-in MLflow integration |
| **Best For** | Quick smoke tests, CI/CD | Full benchmark suites, custom evaluations | Visualization, reporting, comparison | Production workflows, multi-framework eval |

## About

This workshop was built through real debugging and iteration on OpenShift AI. Key learnings documented:

- LMEvalJob CRD uses `spec.pod.container.env` (singular), not `spec.pod.containers[].env`
- OAuth-protected InferenceServices require RBAC + SA token via `OPENAI_API_KEY` env var
- The `served_model_name` in vLLM equals the InferenceService metadata name
- SSL verification must be disabled for self-signed certs (`verify_certificate: "False"`)
