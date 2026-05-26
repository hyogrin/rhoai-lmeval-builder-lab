"""Evaluation helper utilities for Korean MCQ benchmarks.

Loads per-question CSV results, computes category/supercategory accuracy,
and generates Markdown comparison tables (similar to
https://github.com/hyogrin/evaluate-llm-on-korean-dataset).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MAPPING_DIR = Path(__file__).parent / "mapping"

CLICK_CULTURE_CATS = {
    "Economy", "Geography", "History", "Law", "Politics",
    "Society", "Tradition", "Pop Culture",
}
CLICK_LANGUAGE_CATS = {"Functional", "Textual", "Grammar"}

DATASET_DISPLAY_NAMES = {
    "click": "CLIcK",
    "haerae": "HAE-RAE Bench 1.1",
    "kmmlu": "KMMLU (0-shot)",
    "kmmlu_hard": "KMMLU-HARD (0-shot)",
    "kobest_boolq": "KoBEST BoolQ",
}

DATASETS_WITH_SUPERCATEGORY = {"click", "kmmlu", "kmmlu_hard"}


def _load_mapping(filename: str) -> dict:
    path = MAPPING_DIR / filename
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def evaluate(
    csv_path: str | Path,
    dataset: str,
) -> tuple[float, pd.DataFrame, pd.DataFrame | None, int]:
    """Evaluate a single CSV result file.

    Args:
        csv_path: Path to the detailed_results.csv
        dataset: Dataset key (click, haerae, kmmlu, kmmlu_hard, kobest_boolq)

    Returns:
        Tuple of (overall_accuracy, category_accuracy_df, supercategory_accuracy_df or None, num_samples)
    """
    df = pd.read_csv(csv_path)

    original_count = len(df)
    df = df[df["pred"] != "FAILED"]
    failed_count = original_count - len(df)
    if failed_count > 0:
        print(f"  Excluded {failed_count} FAILED responses ({original_count} total)")

    num_samples = len(df)

    if "correct" not in df.columns:
        df["correct"] = df["answer"] == df["pred"]
    else:
        df["correct"] = df["correct"].astype(bool)

    dataset_key = dataset.lower().replace("-", "_")

    if dataset_key == "click":
        if "category" not in df.columns and "id" in df.columns:
            id_to_cat = _load_mapping("id_to_category.json")
            df["category"] = df["id"].astype(str).map(id_to_cat)
            df = df.dropna(subset=["category"])

        df["supercategory"] = df["category"].apply(
            lambda x: "Culture" if x in CLICK_CULTURE_CATS
            else "Language" if x in CLICK_LANGUAGE_CATS
            else "Other"
        )

    elif dataset_key in ("kmmlu", "kmmlu_hard"):
        cat_to_super = _load_mapping("kmmlu_category.json")
        if "category" in df.columns:
            df["supercategory"] = df["category"].map(cat_to_super).fillna("Other")

    overall_acc = round(df["correct"].mean() * 100, 2)

    if "category" in df.columns:
        cat_acc = (
            df.groupby("category")["correct"]
            .mean()
            .multiply(100)
            .round(2)
            .reset_index()
            .rename(columns={"correct": "accuracy"})
            .sort_values("category")
        )
    else:
        cat_acc = pd.DataFrame(columns=["category", "accuracy"])

    supercategory_acc = None
    if dataset_key in DATASETS_WITH_SUPERCATEGORY and "supercategory" in df.columns:
        supercategory_acc = (
            df.groupby("supercategory")["correct"]
            .mean()
            .multiply(100)
            .round(2)
            .reset_index()
            .rename(columns={"correct": "accuracy"})
            .sort_values("supercategory")
        )

    return overall_acc, cat_acc, supercategory_acc, num_samples


def get_markdown_table(
    model_names: list[str],
    category_dfs: list[pd.DataFrame],
    overall_accs: list[float],
    supercategory_dfs: list[pd.DataFrame | None] | None = None,
) -> str:
    """Generate a Markdown comparison table across multiple models.

    Args:
        model_names: List of model display names (column headers)
        category_dfs: List of category_accuracy DataFrames from evaluate()
        overall_accs: List of overall accuracy values
        supercategory_dfs: Optional list of supercategory DataFrames

    Returns:
        Markdown string with comparison tables
    """
    parts: list[str] = []

    if supercategory_dfs and any(df is not None for df in supercategory_dfs):
        valid_sc_dfs = [df for df in supercategory_dfs if df is not None]
        if valid_sc_dfs:
            merged = _merge_accuracy_dfs(model_names, valid_sc_dfs, "supercategory")
            overall_row = {"supercategory": "**Overall**"}
            for i, name in enumerate(model_names):
                if i < len(overall_accs):
                    overall_row[name] = overall_accs[i]
            merged = pd.concat([merged, pd.DataFrame([overall_row])], ignore_index=True)
            parts.append("#### Accuracy by supercategory\n")
            parts.append(merged.to_markdown(index=False))
            parts.append("\n")

    if category_dfs and any(len(df) > 0 for df in category_dfs):
        merged = _merge_accuracy_dfs(model_names, category_dfs, "category")
        if "supercategory" not in merged.columns:
            overall_row = {"category": "**Overall**"}
            for i, name in enumerate(model_names):
                if i < len(overall_accs):
                    overall_row[name] = overall_accs[i]
            merged = pd.concat([merged, pd.DataFrame([overall_row])], ignore_index=True)
        parts.append("#### Accuracy by category\n")
        parts.append(merged.to_markdown(index=False))
        parts.append("\n")

    return "\n".join(parts)


def _merge_accuracy_dfs(
    model_names: list[str],
    dfs: list[pd.DataFrame],
    key_col: str,
) -> pd.DataFrame:
    """Merge multiple accuracy DataFrames on a key column."""
    all_keys = set()
    for df in dfs:
        if key_col in df.columns:
            all_keys.update(df[key_col].tolist())

    merged = pd.DataFrame({key_col: sorted(all_keys)})
    for i, df in enumerate(dfs):
        name = model_names[i] if i < len(model_names) else f"model_{i}"
        rename_df = df.rename(columns={"accuracy": name})
        merged = merged.merge(rename_df[[key_col, name]], on=key_col, how="left")

    merged = merged.fillna("-")
    return merged


def generate_results_md_from_api(
    job_metrics: dict[str, dict],
    output_path: str | Path | None = None,
) -> str:
    """Generate RESULTS.md from EvalHub API metrics (no CSV files needed).

    Args:
        job_metrics: Dict keyed by dataset name, each value is a dict with:
            - "model": model display name
            - "overall_accuracy": float
            - "num_samples": int
            - "metrics": dict of metric_name -> value from EvalHub API
        output_path: Optional path to write the markdown file

    Returns:
        The generated markdown string
    """
    parts: list[str] = []
    parts.append("# Korean LLM Evaluation Results\n")
    parts.append("Generated from EvalHub API metrics.\n")
    parts.append("")

    dataset_order = ["click", "haerae", "kmmlu", "kmmlu_hard", "kobest_boolq"]

    for dataset_key in dataset_order:
        if dataset_key not in job_metrics:
            continue

        info = job_metrics[dataset_key]
        display_name = DATASET_DISPLAY_NAMES.get(dataset_key, dataset_key)
        model_name = info["model"]
        overall = info["overall_accuracy"]
        n_samples = info["num_samples"]
        metrics = info["metrics"]

        parts.append(f"## {display_name}\n")
        parts.append(f"> Samples evaluated — {model_name}: {n_samples}\n")

        cat_rows = []
        sc_rows = []
        for k, v in sorted(metrics.items()):
            if k.startswith("category_accuracy."):
                cat_rows.append((k.replace("category_accuracy.", ""), v))
            elif k.startswith("supercategory_accuracy."):
                sc_rows.append((k.replace("supercategory_accuracy.", ""), v))

        if sc_rows:
            parts.append("#### Accuracy by supercategory\n")
            header = f"| supercategory | {model_name} |"
            sep = f"|:---|---:|"
            parts.append(header)
            parts.append(sep)
            for cat, acc in sc_rows:
                parts.append(f"| {cat} | {acc} |")
            parts.append(f"| **Overall** | {overall} |")
            parts.append("")

        if cat_rows:
            parts.append("#### Accuracy by category\n")
            header = f"| category | {model_name} |"
            sep = f"|:---|---:|"
            parts.append(header)
            parts.append(sep)
            for cat, acc in cat_rows:
                parts.append(f"| {cat} | {acc} |")
            if not sc_rows:
                parts.append(f"| **Overall** | {overall} |")
            parts.append("")

        parts.append("")

    md_content = "\n".join(parts)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_content, encoding="utf-8")
        print(f"Results written to {output_path}")

    return md_content


def generate_results_md(
    results_dir: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """Scan results/ directory and generate a full RESULTS.md.

    Expected structure:
        results_dir/
        ├── click/
        │   ├── model-a.csv
        │   └── model-b.csv
        ├── haerae/
        │   └── model-a.csv
        └── ...

    Args:
        results_dir: Path to the results directory
        output_path: Optional path to write the markdown file

    Returns:
        The generated markdown string
    """
    results_dir = Path(results_dir)
    parts: list[str] = []

    parts.append("# Korean LLM Evaluation Results\n")
    parts.append("Automatically generated from per-question evaluation CSVs.\n")
    parts.append("")

    dataset_order = ["click", "haerae", "kmmlu", "kmmlu_hard", "kobest_boolq"]

    for dataset_key in dataset_order:
        dataset_dir = results_dir / dataset_key
        if not dataset_dir.is_dir():
            continue

        csv_files = sorted(dataset_dir.glob("*.csv"))
        if not csv_files:
            continue

        display_name = DATASET_DISPLAY_NAMES.get(dataset_key, dataset_key)
        parts.append(f"## {display_name}\n")

        model_names: list[str] = []
        overall_accs: list[float] = []
        sample_counts: list[int] = []
        category_dfs: list[pd.DataFrame] = []
        supercategory_dfs: list[pd.DataFrame | None] = []

        for csv_file in csv_files:
            model_name = csv_file.stem
            try:
                overall, cat_df, sc_df, n_samples = evaluate(csv_file, dataset_key)
                model_names.append(model_name)
                overall_accs.append(overall)
                sample_counts.append(n_samples)
                category_dfs.append(cat_df)
                supercategory_dfs.append(sc_df)
            except Exception as e:
                print(f"  Warning: Failed to evaluate {csv_file}: {e}")

        if model_names:
            samples_info = ", ".join(
                f"{name}: {n}" for name, n in zip(model_names, sample_counts)
            )
            parts.append(f"> Samples evaluated — {samples_info}\n")
            table = get_markdown_table(
                model_names, category_dfs, overall_accs, supercategory_dfs
            )
            parts.append(table)
            parts.append("")

    md_content = "\n".join(parts)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_content, encoding="utf-8")
        print(f"Results written to {output_path}")

    return md_content
