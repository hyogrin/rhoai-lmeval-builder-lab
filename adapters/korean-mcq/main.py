"""Korean MCQ Evaluation Adapter for EvalHub.

Implements FrameworkAdapter to evaluate Korean LLMs on multiple-choice benchmarks
(CLIcK, HAE-RAE, KMMLU, KMMLU-HARD) with per-question accuracy tracking.
"""

import asyncio
import csv
import json
import logging
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets

from evalhub.adapter import (
    ErrorInfo,
    EvaluationResult,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
    OCIArtifactSpec,
)

from datasets_config import DATASET_CONFIGS, DatasetConfig, normalize_answer, get_choices_from_example
from parser import parse_answer
from prompts import format_prompt

logger = logging.getLogger(__name__)

MAPPING_DIR = Path(__file__).parent / "mapping"


class KoreanMCQAdapter(FrameworkAdapter):
    """Korean Multiple-Choice Question evaluation adapter.

    Evaluates LLMs on Korean benchmarks by:
    1. Loading datasets from HuggingFace
    2. Formatting MCQ prompts
    3. Calling vLLM-compatible API (OpenAI chat/completions)
    4. Parsing answer letters from responses
    5. Computing per-category accuracy
    6. Reporting results to EvalHub/MLflow
    """

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        start_time = time.time()
        benchmark_id = config.benchmark_id
        logger.info(f"Starting Korean MCQ job {config.id} for benchmark {benchmark_id}")

        # Initialize tracing
        self._trace_enabled = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self._trace_records: list[dict] = []

        try:
            # Phase 1: Initialize
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message=f"Initializing Korean MCQ evaluation for {benchmark_id}",
                        message_code="initializing",
                    ),
                )
            )

            dataset_key = benchmark_id.lower().replace("-", "_")
            if dataset_key not in DATASET_CONFIGS:
                raise ValueError(
                    f"Unknown benchmark: {benchmark_id}. "
                    f"Available: {list(DATASET_CONFIGS.keys())}"
                )

            ds_config = DATASET_CONFIGS[dataset_key]
            model_url = config.model.url
            model_name = config.model.name
            parameters = config.parameters or {}
            limit = config.num_examples or parameters.get("limit")
            temperature = parameters.get("temperature", 0.0)
            max_tokens = parameters.get("max_tokens", 16)
            concurrency = parameters.get("concurrency", 20)

            # Phase 2: Load dataset
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.1,
                    message=MessageInfo(
                        message=f"Loading {benchmark_id} dataset from HuggingFace",
                        message_code="loading_data",
                    ),
                )
            )

            dataset = self._load_dataset(ds_config, limit)
            total = len(dataset)
            logger.info(f"Loaded {total} examples for {benchmark_id}")

            # Load category mapping for CLIcK
            id_to_category = None
            if dataset_key == "click":
                mapping_file = MAPPING_DIR / "id_to_category.json"
                if mapping_file.exists():
                    with open(mapping_file) as f:
                        id_to_category = json.load(f)

            # Phase 3: Run evaluation
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.2,
                    message=MessageInfo(
                        message=f"Evaluating {total} questions via {model_name}",
                        message_code="running_evaluation",
                    ),
                )
            )

            results_rows = self._evaluate_dataset(
                dataset=dataset,
                ds_config=ds_config,
                model_url=model_url,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                id_to_category=id_to_category,
                callbacks=callbacks,
                total=total,
                concurrency=concurrency,
            )

            # Phase 4: Post-processing
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.9,
                    message=MessageInfo(
                        message="Computing accuracy metrics",
                        message_code="post_processing",
                    ),
                )
            )

            evaluation_results, overall_acc = self._compute_metrics(
                results_rows, dataset_key
            )

            # Save CSV and markdown artifacts
            output_files = self._save_artifacts(
                config.id, benchmark_id, model_name, results_rows, evaluation_results, overall_acc
            )

            # Phase 5: Persist OCI artifacts
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.95,
                    message=MessageInfo(
                        message="Persisting evaluation artifacts",
                        message_code="persisting_artifacts",
                    ),
                )
            )

            oci_artifact = None
            oci_exports = config.exports.oci if config.exports else None
            if oci_exports is not None and output_files:
                coords = oci_exports.coordinates.model_copy(deep=True)
                coords.annotations.update(
                    {
                        "org.opencontainers.image.created": datetime.now(UTC).isoformat(),
                        "io.github.eval-hub.benchmark": benchmark_id,
                        "io.github.eval-hub.model": model_name,
                        "io.github.eval-hub.job_id": config.id,
                    }
                )
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files_path=output_files[0].parent,
                        coordinates=coords,
                    )
                )
                logger.info(f"OCI artifact created: {oci_artifact.reference}")

            duration = time.time() - start_time

            return JobResults(
                id=config.id,
                benchmark_id=benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=model_name,
                results=evaluation_results,
                overall_score=overall_acc / 100.0 if overall_acc else None,
                num_examples_evaluated=total,
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "korean-mcq",
                    "dataset": dataset_key,
                    "num_choices": ds_config.num_choices,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "concurrency": concurrency,
                    "limit": limit,
                },
                oci_artifact=oci_artifact,
            )

        except Exception as e:
            logger.exception("Korean MCQ evaluation failed")
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    message=MessageInfo(message=str(e), message_code="failed"),
                    error=ErrorInfo(
                        message=str(e), message_code="evaluation_error"
                    ),
                )
            )
            raise

    def _load_dataset(self, config: DatasetConfig, limit: int | None):
        """Load dataset from HuggingFace."""
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        if config.load_all_configs:
            configs = get_dataset_config_names(config.hf_path, token=hf_token)
            logger.info(f"Loading {len(configs)} configs from {config.hf_path}")

            all_splits = []
            for cfg_name in configs:
                try:
                    kwargs: dict[str, Any] = {
                        "path": config.hf_path,
                        "name": cfg_name,
                        "split": config.split,
                    }
                    if hf_token:
                        kwargs["token"] = hf_token
                    ds = load_dataset(**kwargs)
                    # Add category column from config name if not present
                    cat_col = config.category_col or "category"
                    if cat_col not in ds.column_names:
                        ds = ds.add_column(cat_col, [cfg_name] * len(ds))
                    all_splits.append(ds)
                except Exception as e:
                    logger.warning(f"Skipping config {cfg_name}: {e}")

            if not all_splits:
                raise ValueError(f"No configs loaded from {config.hf_path}")

            ds = concatenate_datasets(all_splits)
        else:
            kwargs: dict[str, Any] = {"path": config.hf_path, "split": config.split}
            if config.hf_name:
                kwargs["name"] = config.hf_name
            if hf_token:
                kwargs["token"] = hf_token
            ds = load_dataset(**kwargs)

        if limit and limit < len(ds):
            ds = ds.select(range(limit))

        return ds

    def _evaluate_dataset(
        self,
        dataset,
        ds_config: DatasetConfig,
        model_url: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        id_to_category: dict | None,
        callbacks: JobCallbacks,
        total: int,
        concurrency: int = 20,
    ) -> list[dict]:
        """Evaluate all examples using async parallel HTTP calls."""
        return asyncio.run(
            self._evaluate_dataset_async(
                dataset=dataset,
                ds_config=ds_config,
                model_url=model_url,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                id_to_category=id_to_category,
                callbacks=callbacks,
                total=total,
                concurrency=concurrency,
            )
        )

    async def _evaluate_dataset_async(
        self,
        dataset,
        ds_config: DatasetConfig,
        model_url: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        id_to_category: dict | None,
        callbacks: JobCallbacks,
        total: int,
        concurrency: int = 20,
    ) -> list[dict]:
        """Async evaluation with concurrent LLM calls."""
        # Determine API endpoint
        base_url = model_url.rstrip("/")
        if "/v1/completions" in base_url:
            base_url = base_url.replace("/v1/completions", "")
        elif "/v1/chat/completions" in base_url:
            base_url = base_url.replace("/v1/chat/completions", "")
        elif base_url.endswith("/v1"):
            base_url = base_url[:-3]

        # Prepare all items upfront
        prepared: list[dict] = []
        for idx, example in enumerate(dataset):
            question = example[ds_config.question_col]
            choices = get_choices_from_example(example, ds_config)
            raw_answer = example[ds_config.answer_col]
            context = example.get(ds_config.context_col) if ds_config.context_col else None
            if context is not None and not context.strip():
                context = None

            answer_letter = normalize_answer(raw_answer, ds_config, choices)

            category = None
            if ds_config.category_col and ds_config.category_col in example:
                category = example[ds_config.category_col]
            elif id_to_category and ds_config.id_col and ds_config.id_col in example:
                category = id_to_category.get(str(example[ds_config.id_col]))

            if ds_config.preformatted_prompt:
                prompt = question
            else:
                prompt = format_prompt(question, choices, context)

            item = {
                "idx": idx,
                "prompt": prompt,
                "answer_letter": answer_letter,
                "category": category,
            }
            if ds_config.id_col and ds_config.id_col in example:
                item["id"] = example[ds_config.id_col]
            prepared.append(item)

        logger.info(f"Running async evaluation: {total} items, concurrency={concurrency}")

        sem = asyncio.Semaphore(concurrency)
        completed_count = 0
        last_reported = 0

        async def eval_one(client: httpx.AsyncClient, item: dict) -> dict:
            nonlocal completed_count, last_reported
            try:
                async with sem:
                    response_text = await self._call_llm_async(
                        client, model_name, item["prompt"], temperature, max_tokens
                    )
            except Exception as e:
                logger.error(f"Unhandled error in eval_one (idx={item['idx']}): {e}")
                response_text = ""

            pred = parse_answer(response_text, ds_config.num_choices)
            row = {
                "index": item["idx"],
                "answer": item["answer_letter"],
                "pred": pred if pred else "FAILED",
                "response": response_text,
                "correct": pred == item["answer_letter"] if pred else False,
            }
            if item["category"]:
                row["category"] = item["category"]
            if "id" in item:
                row["id"] = item["id"]

            completed_count += 1
            report_interval = max(1, total // 10)
            if completed_count - last_reported >= report_interval:
                last_reported = completed_count
                progress = 0.2 + 0.7 * (completed_count / total)
                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.RUNNING,
                        phase=JobPhase.RUNNING_EVALUATION,
                        progress=progress,
                        message=MessageInfo(
                            message=f"Evaluated {completed_count}/{total} questions",
                            message_code="running_evaluation",
                        ),
                    )
                )

            return row

        async with httpx.AsyncClient(
            base_url=base_url,
            verify=False,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=30.0),
            limits=httpx.Limits(
                max_connections=concurrency + 5,
                max_keepalive_connections=concurrency,
            ),
        ) as client:
            tasks = [eval_one(client, item) for item in prepared]
            results = await asyncio.gather(*tasks)

        return list(results)

    async def _call_llm_async(
        self,
        client: httpx.AsyncClient,
        model_name: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Async LLM call with robust retry logic.

        Handles rate limits (429), server overload (503), timeouts,
        and connection errors with exponential backoff + jitter.
        """
        max_retries = 5
        request_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if os.getenv("DISABLE_THINKING", "true").lower() == "true":
            request_payload["chat_template_kwargs"] = {"enable_thinking": False}

        for attempt in range(max_retries):
            try:
                response = await client.post(
                    "/v1/chat/completions",
                    json=request_payload,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                if self._trace_enabled:
                    self._trace_records.append({
                        "prompt": prompt[:500],
                        "response": content,
                        "model": model_name,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "status": "success",
                    })

                return content

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                if status_code == 429:
                    if attempt < max_retries - 1:
                        delay = min(60, (attempt + 1) * 5) + random.uniform(0, 2)
                        logger.warning(
                            f"Rate limited (429), retry {attempt+1}/{max_retries} in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.error("Rate limit exceeded after all retries")
                    return ""

                if status_code == 503:
                    if attempt < max_retries - 1:
                        delay = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"Service unavailable (503), retry {attempt+1}/{max_retries} in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.error("Service unavailable after all retries")
                    return ""

                if status_code == 500 and attempt < max_retries - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Server error (500), retry {attempt+1}/{max_retries} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.error(f"LLM API error (HTTP {status_code}): {e}")
                return ""

            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Request timeout, retry {attempt+1}/{max_retries} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("LLM call timed out after all retries")
                return ""

            except (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError, OSError) as e:
                if attempt < max_retries - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Connection error ({type(e).__name__}), retry {attempt+1}/{max_retries} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Connection failed after all retries: {e}")
                return ""

            except KeyError as e:
                logger.error(f"Unexpected response format (missing key {e})")
                return ""

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Unexpected error ({type(e).__name__}: {e}), "
                        f"retry {attempt+1}/{max_retries} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"LLM call failed after all retries: {e}")
                return ""

        return ""

    def _compute_metrics(
        self, results: list[dict], dataset_key: str
    ) -> tuple[list[EvaluationResult], float]:
        """Compute accuracy metrics from results."""
        eval_results: list[EvaluationResult] = []

        # Filter out FAILED predictions
        valid = [r for r in results if r["pred"] != "FAILED"]
        if not valid:
            return [], 0.0

        # Overall accuracy
        correct_count = sum(1 for r in valid if r["correct"])
        overall_acc = round(correct_count / len(valid) * 100, 2)

        eval_results.append(
            EvaluationResult(
                metric_name="overall_accuracy",
                metric_value=overall_acc,
                metric_type="float",
                num_samples=len(valid),
                metadata={"total": len(results), "valid": len(valid), "failed": len(results) - len(valid)},
            )
        )

        # Per-category accuracy
        categories: dict[str, list[bool]] = {}
        for r in valid:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r["correct"])

        for cat, correctness in sorted(categories.items()):
            cat_acc = round(sum(correctness) / len(correctness) * 100, 2)
            eval_results.append(
                EvaluationResult(
                    metric_name=f"category_accuracy.{cat}",
                    metric_value=cat_acc,
                    metric_type="float",
                    num_samples=len(correctness),
                    metadata={"category": cat},
                )
            )

        # Supercategory accuracy for CLIcK and KMMLU
        if dataset_key == "click":
            supercats = self._compute_click_supercategories(valid)
            for sc, acc in supercats.items():
                eval_results.append(
                    EvaluationResult(
                        metric_name=f"supercategory_accuracy.{sc}",
                        metric_value=acc,
                        metric_type="float",
                        metadata={"supercategory": sc},
                    )
                )
        elif dataset_key in ("kmmlu", "kmmlu_hard"):
            supercats = self._compute_kmmlu_supercategories(valid)
            for sc, acc in supercats.items():
                eval_results.append(
                    EvaluationResult(
                        metric_name=f"supercategory_accuracy.{sc}",
                        metric_value=acc,
                        metric_type="float",
                        metadata={"supercategory": sc},
                    )
                )

        return eval_results, overall_acc

    def _compute_click_supercategories(self, valid: list[dict]) -> dict[str, float]:
        culture_cats = {
            "Economy", "Geography", "History", "Law", "Politics",
            "Society", "Tradition", "Pop Culture",
        }
        language_cats = {"Functional", "Textual", "Grammar"}

        groups: dict[str, list[bool]] = {"Culture": [], "Language": [], "Other": []}
        for r in valid:
            cat = r.get("category", "")
            if cat in culture_cats:
                groups["Culture"].append(r["correct"])
            elif cat in language_cats:
                groups["Language"].append(r["correct"])
            else:
                groups["Other"].append(r["correct"])

        return {
            k: round(sum(v) / len(v) * 100, 2)
            for k, v in groups.items()
            if v
        }

    def _compute_kmmlu_supercategories(self, valid: list[dict]) -> dict[str, float]:
        mapping_file = MAPPING_DIR / "kmmlu_category.json"
        if not mapping_file.exists():
            return {}

        with open(mapping_file) as f:
            cat_to_super = json.load(f)

        groups: dict[str, list[bool]] = {}
        for r in valid:
            cat = r.get("category", "")
            sc = cat_to_super.get(cat, "Other")
            if sc not in groups:
                groups[sc] = []
            groups[sc].append(r["correct"])

        return {
            k: round(sum(v) / len(v) * 100, 2)
            for k, v in groups.items()
            if v
        }

    def _save_artifacts(
        self,
        job_id: str,
        benchmark_id: str,
        model_name: str,
        results_rows: list[dict],
        evaluation_results: list[EvaluationResult],
        overall_acc: float,
    ) -> list[Path]:
        """Save CSV results and markdown summary."""
        if self.local_jobs_base_path is not None:
            output_dir = self.local_jobs_base_path / "results"
        else:
            output_dir = Path("/tmp/korean-mcq-results")
        output_dir.mkdir(parents=True, exist_ok=True)

        files = []

        # CSV with per-question results
        csv_path = output_dir / "detailed_results.csv"
        if results_rows:
            fieldnames = list(results_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_rows)
            files.append(csv_path)

        # JSON summary
        summary_json = output_dir / "results.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "benchmark_id": benchmark_id,
                    "model_name": model_name,
                    "framework": "korean-mcq",
                    "overall_accuracy": overall_acc,
                    "results": [
                        {
                            "metric_name": r.metric_name,
                            "metric_value": r.metric_value,
                            "num_samples": r.num_samples,
                        }
                        for r in evaluation_results
                    ],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        files.append(summary_json)

        # Markdown report
        md_path = output_dir / "DETAILED_RESULTS.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {benchmark_id} Evaluation Results\n\n")
            f.write(f"**Model**: {model_name}\n\n")
            f.write(f"**Overall Accuracy**: {overall_acc}%\n\n")
            f.write(f"**Total Questions**: {len(results_rows)}\n\n")

            # Category table
            cat_results = [r for r in evaluation_results if r.metric_name.startswith("category_accuracy.")]
            if cat_results:
                f.write("## Category Accuracy\n\n")
                f.write("| Category | Accuracy (%) | Samples |\n")
                f.write("|----------|-------------|--------|\n")
                for r in cat_results:
                    cat_name = r.metric_name.replace("category_accuracy.", "")
                    f.write(f"| {cat_name} | {r.metric_value} | {r.num_samples} |\n")

            # Supercategory table
            sc_results = [r for r in evaluation_results if r.metric_name.startswith("supercategory_accuracy.")]
            if sc_results:
                f.write("\n## Supercategory Accuracy\n\n")
                f.write("| Supercategory | Accuracy (%) |\n")
                f.write("|--------------|-------------|\n")
                for r in sc_results:
                    sc_name = r.metric_name.replace("supercategory_accuracy.", "")
                    f.write(f"| {sc_name} | {r.metric_value} |\n")

        files.append(md_path)

        logger.info(f"Saved {len(files)} artifact files to {output_dir}")
        return files


def main() -> None:
    """Main entry point for Korean MCQ adapter."""
    import sys
    from evalhub.adapter import DefaultCallbacks

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = KoreanMCQAdapter(job_spec_path=job_spec_path)
        logger.info(f"Loaded job {adapter.job_spec.id}")
        logger.info(f"Benchmark: {adapter.job_spec.benchmark_id}")
        logger.info(f"Model: {adapter.job_spec.model.name}")

        callbacks = DefaultCallbacks.from_adapter(adapter)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        logger.info(f"Job completed: overall_accuracy={results.overall_score}")

        # Save to MLflow
        run_id = callbacks.mlflow.save(results, adapter.job_spec)
        if run_id:
            results.mlflow_run_id = run_id
            logger.info(f"MLflow run: {run_id}")

        # Log traces directly to MLflow Traces tab (bypassing EvalHub proxy)
        mlflow_direct_uri = os.getenv("MLFLOW_DIRECT_URI", "https://mlflow.redhat-ods-applications.svc:8443")
        if adapter._trace_enabled and adapter._trace_records and run_id:
            try:
                import mlflow
                from mlflow import MlflowClient
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

                os.environ.pop("MLFLOW_TRACKING_SERVER_CERT_PATH", None)
                os.environ["MLFLOW_TRACKING_URI"] = mlflow_direct_uri
                os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

                sa_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
                if Path(sa_token_path).exists():
                    sa_token = Path(sa_token_path).read_text().strip()
                    os.environ["MLFLOW_TRACKING_TOKEN"] = sa_token

                mlflow.set_tracking_uri(mlflow_direct_uri)
                mlflow_client = MlflowClient(tracking_uri=mlflow_direct_uri)
                logger.info(f"Logging {len(adapter._trace_records)} traces to MLflow at {mlflow_direct_uri}")

                exp_id = mlflow_client.get_run(run_id).info.experiment_id
                mlflow.set_experiment(experiment_id=exp_id)

                logged_count = 0
                for i, record in enumerate(adapter._trace_records):
                    try:
                        root_span = mlflow_client.start_trace(
                            name=f"llm_call_{i}",
                            experiment_id=exp_id,
                            inputs={"prompt": record["prompt"], "model": record["model"]},
                            attributes={
                                "temperature": str(record["temperature"]),
                                "max_tokens": str(record["max_tokens"]),
                            },
                        )
                        trace_id = root_span.request_id
                        mlflow_client.end_trace(
                            trace_id=trace_id,
                            outputs={"response": record["response"]},
                            attributes={"status": record["status"]},
                        )
                        logged_count += 1
                    except Exception as e:
                        if i == 0:
                            logger.warning(f"Trace logging failed on first record: {e}")
                            break
                        continue

                logger.info(f"Logged {logged_count}/{len(adapter._trace_records)} traces to MLflow Traces tab")
            except Exception as e:
                logger.warning(f"Failed to connect to MLflow for tracing: {e}")
                import traceback
                logger.warning(traceback.format_exc())

        callbacks.report_results(results)
        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"Job spec not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
