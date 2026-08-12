"""Standalone Korean MCQ evaluation against a MaaS (Model-as-a-Service) endpoint.

Runs all 5 Korean benchmarks (CLIcK, HAE-RAE, KMMLU, KMMLU-HARD, KoBEST BoolQ)
directly against an OpenAI-compatible API endpoint without requiring EvalHub.

Usage:
    python run_eval_maas.py \
        --endpoint https://maas.example.com/model/v1/chat/completions \
        --api-key sk-xxx \
        --model-name my-model \
        --limit 10000 \
        --concurrency 20
"""

import argparse
import asyncio
import ast
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAPPING_DIR = Path(__file__).parent / "utils" / "mapping"

# ---------------------------------------------------------------------------
# Dataset config (same as adapters/korean-mcq/datasets_config.py)
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    hf_path: str
    hf_name: str | None = None
    split: str = "test"
    question_col: str = "question"
    choices_col: str = "choices"
    answer_col: str = "answer"
    context_col: str | None = None
    category_col: str | None = None
    id_col: str | None = None
    num_choices: int = 5
    answer_offset: int = 0
    choice_labels: list[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E"])
    choices_in_separate_cols: bool = False
    load_all_configs: bool = False
    fixed_choices: list[str] | None = None
    preformatted_prompt: bool = False


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "click": DatasetConfig(
        hf_path="EunsuKim/CLIcK",
        split="train",
        question_col="question",
        choices_col="choices",
        answer_col="answer",
        context_col="paragraph",
        id_col="id",
        num_choices=4,
        answer_offset=0,
        choice_labels=["A", "B", "C", "D"],
    ),
    "haerae": DatasetConfig(
        hf_path="HAERAE-HUB/HAE_RAE_BENCH_1.1",
        split="test",
        question_col="query",
        choices_col="options",
        answer_col="answer",
        category_col="category",
        num_choices=5,
        load_all_configs=True,
        preformatted_prompt=True,
    ),
    "kmmlu": DatasetConfig(
        hf_path="HAERAE-HUB/KMMLU",
        split="test",
        question_col="question",
        answer_col="answer",
        category_col="Category",
        num_choices=4,
        answer_offset=1,
        choice_labels=["A", "B", "C", "D"],
        choices_in_separate_cols=True,
        load_all_configs=True,
    ),
    "kmmlu_hard": DatasetConfig(
        hf_path="HAERAE-HUB/KMMLU-HARD",
        split="test",
        question_col="question",
        answer_col="answer",
        category_col="Category",
        num_choices=4,
        answer_offset=1,
        choice_labels=["A", "B", "C", "D"],
        choices_in_separate_cols=True,
        load_all_configs=True,
    ),
    "kobest_boolq": DatasetConfig(
        hf_path="skt/kobest_v1",
        hf_name="boolq",
        split="test",
        question_col="question",
        answer_col="label",
        context_col="paragraph",
        num_choices=2,
        answer_offset=0,
        choice_labels=["A", "B"],
        fixed_choices=["참", "거짓"],
    ),
}


# ---------------------------------------------------------------------------
# Helpers (prompt, parser, dataset utils)
# ---------------------------------------------------------------------------

def get_choices_from_example(example: dict, config: DatasetConfig) -> list[str]:
    if config.fixed_choices:
        return config.fixed_choices
    if config.choices_in_separate_cols:
        return [example[label] for label in config.choice_labels]
    raw = example[config.choices_col]
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return [c.strip() for c in raw.split(",")]
    return raw


def normalize_answer(raw_answer, config: DatasetConfig, choices: list[str] | None = None) -> str:
    if config.fixed_choices and config.fixed_choices == ["참", "거짓"]:
        try:
            label = int(raw_answer)
            return "A" if label == 1 else "B"
        except (ValueError, TypeError):
            pass
    if isinstance(raw_answer, str):
        stripped = raw_answer.strip().strip("()")
        if len(stripped) == 1 and stripped.upper() in "ABCDE":
            return stripped.upper()
    try:
        idx = int(raw_answer) - config.answer_offset
        if 0 <= idx < config.num_choices:
            return chr(65 + idx)
    except (ValueError, TypeError):
        pass
    if choices and isinstance(raw_answer, str):
        for i, choice in enumerate(choices):
            if choice.strip() == raw_answer.strip():
                return chr(65 + i)
    return str(raw_answer).upper()


def format_prompt(question: str, choices: list[str], context: str | None = None) -> str:
    num_choices = len(choices)
    letters = [chr(65 + i) for i in range(num_choices)]
    options_str = ", ".join(letters[:-1]) + f", {letters[-1]}" if num_choices > 1 else letters[0]
    options_format = ", ".join([f"{chr(65 + i)}: {choices[i]}" for i in range(num_choices)])

    if context:
        return (
            f"You are taking a multiple choice exam. "
            f"Read the context carefully, then select the single correct answer "
            f"from the choices provided and respond with ONLY the letter ({options_str}). "
            f"Do not include any explanation or additional text.\n\n"
            f"{context}\n\n"
            f"{question}\n\n"
            f"{options_format}\n"
        )
    return (
        f"You are taking a multiple choice exam. "
        f"Select the single correct answer from the choices provided and "
        f"respond with ONLY the letter ({options_str}). "
        f"Do not include any explanation or additional text.\n\n"
        f"{question}\n\n"
        f"{options_format}\n"
    )


_THINK_END_PATTERN = re.compile(r"</think>\s*", re.IGNORECASE)
_ANSWER_PATTERN = re.compile(
    r"""(?:^[(\[]?\s*([A-E])\s*[)\]]?\s*$|(?:answer|정답)[^A-E]*([A-E])|^([A-E])\s*[.:)\]]|(?:^|\s)([A-E])(?:\s*$))""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def parse_answer(response: str, num_choices: int = 5) -> str | None:
    if not response or not response.strip():
        return None
    text = response.strip()
    match = _THINK_END_PATTERN.search(text)
    if match:
        text = text[match.end():].strip()
    if not text:
        return None
    valid_letters = {chr(65 + i) for i in range(num_choices)}
    if len(text) == 1 and text.upper() in valid_letters:
        return text.upper()
    cleaned = re.sub(r"[^A-Za-z]", "", text)
    if len(cleaned) == 1 and cleaned.upper() in valid_letters:
        return cleaned.upper()
    for m in _ANSWER_PATTERN.finditer(text):
        for group in m.groups():
            if group and group.upper() in valid_letters:
                return group.upper()
    for char in text:
        if char.upper() in valid_letters:
            return char.upper()
    return None


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

async def call_llm(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
) -> str:
    max_retries = 5
    messages = [
        {"role": "system", "content": "/no_think\nYou are an exam assistant. Output ONLY the answer letter. No explanation."},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.post(
                    "/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code in (429, 503, 500) and attempt < max_retries - 1:
                delay = min(60, (attempt + 1) * 5) + random.uniform(0, 2)
                logger.warning(f"HTTP {code}, retry {attempt+1}/{max_retries} in {delay:.1f}s")
                await asyncio.sleep(delay)
                continue
            logger.error(f"HTTP {code} after retries")
            return ""
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"{type(e).__name__}, retry {attempt+1}/{max_retries} in {delay:.1f}s")
                await asyncio.sleep(delay)
                continue
            logger.error(f"Connection failed: {e}")
            return ""
        except Exception as e:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Unexpected {type(e).__name__}: {e}, retry {attempt+1}")
                await asyncio.sleep(delay)
                continue
            logger.error(f"Failed: {e}")
            return ""
    return ""


def load_dataset_for_benchmark(config: DatasetConfig, limit: int | None) -> list:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    if config.load_all_configs:
        configs = get_dataset_config_names(config.hf_path, token=hf_token)
        logger.info(f"Loading {len(configs)} configs from {config.hf_path}")
        all_splits = []
        for cfg_name in configs:
            try:
                kwargs = {"path": config.hf_path, "name": cfg_name, "split": config.split}
                if hf_token:
                    kwargs["token"] = hf_token
                ds = load_dataset(**kwargs)
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
        kwargs = {"path": config.hf_path, "split": config.split}
        if config.hf_name:
            kwargs["name"] = config.hf_name
        if hf_token:
            kwargs["token"] = hf_token
        ds = load_dataset(**kwargs)

    if limit and limit < len(ds):
        ds = ds.select(range(limit))

    return ds


async def run_benchmark(
    benchmark_id: str,
    endpoint: str,
    api_key: str,
    model_name: str,
    limit: int | None,
    concurrency: int,
    temperature: float = 0.0,
    max_tokens: int = 16,
) -> dict:
    """Run a single benchmark and return results dict."""
    ds_config = DATASET_CONFIGS[benchmark_id]
    logger.info(f"=== Starting {benchmark_id} ===")

    dataset = load_dataset_for_benchmark(ds_config, limit)
    total = len(dataset)
    logger.info(f"Loaded {total} examples")

    # Load CLIcK category mapping
    id_to_category = None
    if benchmark_id == "click":
        mapping_file = MAPPING_DIR / "id_to_category.json"
        if mapping_file.exists():
            with open(mapping_file) as f:
                id_to_category = json.load(f)

    # Prepare items
    prepared = []
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

        prepared.append({
            "idx": idx,
            "prompt": prompt,
            "answer_letter": answer_letter,
            "category": category,
        })

    # Run evaluation
    sem = asyncio.Semaphore(concurrency)
    completed = 0

    async def eval_one(client: httpx.AsyncClient, item: dict) -> dict:
        nonlocal completed
        response_text = await call_llm(client, sem, model_name, item["prompt"], temperature, max_tokens, api_key)
        pred = parse_answer(response_text, ds_config.num_choices)
        completed += 1
        if completed % max(1, total // 10) == 0:
            logger.info(f"  Progress: {completed}/{total} ({completed*100//total}%)")
        return {
            "index": item["idx"],
            "answer": item["answer_letter"],
            "pred": pred if pred else "FAILED",
            "correct": pred == item["answer_letter"] if pred else False,
            "category": item["category"],
        }

    base_url = endpoint.rstrip("/")
    if "/v1/chat/completions" in base_url:
        base_url = base_url.replace("/v1/chat/completions", "")
    elif "/v1/completions" in base_url:
        base_url = base_url.replace("/v1/completions", "")
    elif base_url.endswith("/v1"):
        base_url = base_url[:-3]

    start_time = time.time()
    async with httpx.AsyncClient(
        base_url=base_url,
        verify=False,
        timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=30.0),
        limits=httpx.Limits(max_connections=concurrency + 5, max_keepalive_connections=concurrency),
    ) as client:
        tasks = [eval_one(client, item) for item in prepared]
        results = await asyncio.gather(*tasks)

    duration = time.time() - start_time
    results_list = list(results)

    # Compute metrics
    valid = [r for r in results_list if r["pred"] != "FAILED"]
    if not valid:
        logger.error(f"No valid results for {benchmark_id}")
        return {"benchmark_id": benchmark_id, "overall_accuracy": 0, "metrics": {}, "duration": duration}

    correct_count = sum(1 for r in valid if r["correct"])
    overall_acc = round(correct_count / len(valid) * 100, 2)

    # Category accuracy
    categories: dict[str, list[bool]] = {}
    for r in valid:
        cat = r.get("category") or "unknown"
        categories.setdefault(cat, []).append(r["correct"])

    metrics = {"overall_accuracy": overall_acc}
    for cat, correctness in sorted(categories.items()):
        metrics[f"category_accuracy.{cat}"] = round(sum(correctness) / len(correctness) * 100, 2)

    # Supercategory accuracy
    if benchmark_id == "click":
        culture_cats = {"Economy", "Geography", "History", "Law", "Politics", "Society", "Tradition", "Pop Culture"}
        language_cats = {"Functional", "Textual", "Grammar"}
        groups: dict[str, list[bool]] = {"Culture": [], "Language": []}
        for r in valid:
            cat = r.get("category", "")
            if cat in culture_cats:
                groups["Culture"].append(r["correct"])
            elif cat in language_cats:
                groups["Language"].append(r["correct"])
        for k, v in groups.items():
            if v:
                metrics[f"supercategory_accuracy.{k}"] = round(sum(v) / len(v) * 100, 2)

    elif benchmark_id in ("kmmlu", "kmmlu_hard"):
        mapping_file = MAPPING_DIR / "kmmlu_category.json"
        if mapping_file.exists():
            with open(mapping_file) as f:
                cat_to_super = json.load(f)
            groups: dict[str, list[bool]] = {}
            for r in valid:
                cat = r.get("category", "")
                sc = cat_to_super.get(cat, "Other")
                groups.setdefault(sc, []).append(r["correct"])
            for k, v in groups.items():
                if v:
                    metrics[f"supercategory_accuracy.{k}"] = round(sum(v) / len(v) * 100, 2)

    failed_count = len(results_list) - len(valid)
    logger.info(
        f"  {benchmark_id}: {overall_acc}% "
        f"(valid={len(valid)}, failed={failed_count}, duration={duration:.1f}s)"
    )

    return {
        "benchmark_id": benchmark_id,
        "overall_accuracy": overall_acc,
        "metrics": metrics,
        "num_samples": len(valid),
        "num_failed": failed_count,
        "duration": duration,
    }


def save_results(model_name: str, benchmark_id: str, result: dict):
    """Save results in the same format as EvalHub unified results."""
    output_dir = Path(__file__).parent / "results" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d-%H%M")
    job_id = f"{random.randint(0, 0xFFFFFFFF):08x}"
    filename = f"unified-{benchmark_id}-{timestamp}_{job_id}.json"

    data = {
        "job_id": job_id,
        "name": f"unified-{benchmark_id}-{timestamp}",
        "model": {"url": result.get("endpoint", ""), "name": model_name},
        "experiment": f"{model_name}-full-eval-{timestamp}",
        "benchmarks": [
            {
                "id": benchmark_id,
                "provider_id": "korean_mcq",
                "metrics": result["metrics"],
            }
        ],
    }

    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {filepath}")
    return filepath


async def main():
    parser = argparse.ArgumentParser(description="Run Korean MCQ benchmarks against a MaaS endpoint")
    parser.add_argument("--endpoint", required=True, help="Model API endpoint URL")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--model-name", required=True, help="Model name for results")
    parser.add_argument("--limit", type=int, default=10000, help="Max samples per benchmark (default: 10000)")
    parser.add_argument("--concurrency", type=int, default=20, help="Parallel requests (default: 20)")
    parser.add_argument("--benchmarks", nargs="+", default=["click", "haerae", "kmmlu", "kmmlu_hard", "kobest_boolq"],
                        help="Benchmarks to run")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    # Verify endpoint is reachable (with cold-start retry loop)
    logger.info(f"Testing endpoint: {args.endpoint}")
    base_url = args.endpoint.rstrip("/")
    if "/v1/chat/completions" in base_url:
        base_url = base_url.replace("/v1/chat/completions", "")
    elif base_url.endswith("/v1"):
        base_url = base_url[:-3]

    max_warmup_attempts = 20
    for attempt in range(max_warmup_attempts):
        try:
            async with httpx.AsyncClient(base_url=base_url, verify=False, timeout=180.0) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": args.model_name, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2},
                    headers={"Authorization": f"Bearer {args.api_key}"},
                )
                if resp.status_code == 200:
                    logger.info(f"Endpoint OK: {resp.json()['choices'][0]['message']['content'][:50]}")
                    break
                elif resp.status_code == 503:
                    wait = min(30 * (attempt + 1), 120)
                    logger.warning(f"Model not ready (503), attempt {attempt+1}/{max_warmup_attempts}, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Unexpected HTTP {resp.status_code}: {resp.text[:200]}")
                    sys.exit(1)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            wait = min(30 * (attempt + 1), 120)
            logger.warning(f"Connection issue ({type(e).__name__}), attempt {attempt+1}/{max_warmup_attempts}, waiting {wait}s...")
            await asyncio.sleep(wait)
        except Exception as e:
            logger.error(f"Cannot reach endpoint: {e}")
            sys.exit(1)
    else:
        logger.error(f"Model not ready after {max_warmup_attempts} attempts. Giving up.")
        sys.exit(1)

    # Run benchmarks
    all_results = {}
    for bm in args.benchmarks:
        if bm not in DATASET_CONFIGS:
            logger.error(f"Unknown benchmark: {bm}")
            continue
        result = await run_benchmark(
            benchmark_id=bm,
            endpoint=args.endpoint,
            api_key=args.api_key,
            model_name=args.model_name,
            limit=args.limit,
            concurrency=args.concurrency,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        result["endpoint"] = args.endpoint
        all_results[bm] = result
        save_results(args.model_name, bm, result)

    # Print summary
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY: {args.model_name}")
    print("=" * 70)
    print(f"{'Benchmark':<20} {'Accuracy':>10} {'Samples':>10} {'Duration':>10}")
    print("-" * 70)
    for bm, r in all_results.items():
        print(f"{bm:<20} {r['overall_accuracy']:>9.2f}% {r.get('num_samples', 0):>10} {r['duration']:>9.1f}s")
    print("=" * 70)

    # Save combined summary
    summary_path = Path(__file__).parent / "results" / args.model_name / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
