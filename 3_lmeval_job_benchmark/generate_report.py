#!/usr/bin/env python3
"""
Generate an HTML benchmark report from LMEvalJob result JSON files.

Usage:
    python generate_report.py --results-dir ../results --output report.html
    python generate_report.py --results-dir ../results --output report.html --title "My Eval Report"

The script reads JSON result files from the results directory structure:
    results/<model-name>/<task-name>.json

And produces a standalone HTML file with:
    - Summary table with accuracy scores per model/task
    - Per-task breakdown with category-level scores
    - Bar chart visualizations (using inline Chart.js)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


KMMLU_SUPERCATEGORY = {
    "STEM": [
        "biology", "chemical_engineering", "chemistry", "civil_engineering",
        "computer_science", "ecology", "electrical_engineering",
        "information_technology", "materials_engineering", "math",
        "mechanical_engineering"
    ],
    "HUMSS": [
        "accounting", "criminal_law", "economics", "education",
        "korean_history", "law", "management",
        "political_science_and_sociology", "psychology",
        "social_welfare", "taxation"
    ],
    "Applied Science": [
        "aviation_engineering_and_maintenance", "electronics_engineering",
        "energy_management", "environmental_science",
        "gas_technology_and_engineering", "geomatics",
        "industrial_engineer", "machine_design_and_manufacturing",
        "maritime_engineering", "nondestructive_testing",
        "railway_and_automotive_engineering",
        "telecommunications_and_wireless_technology"
    ],
    "Other": [
        "agricultural_sciences", "construction", "fashion",
        "food_processing", "health", "interior_architecture_and_design",
        "marketing", "patent", "public_safety", "real_estate",
        "refrigerating_machinery"
    ]
}

CLICK_SUPERCATEGORY = {
    "Culture": [
        "cul_economy", "cul_geography", "cul_history", "cul_law",
        "cul_politics", "cul_kpop", "cul_society", "cul_tradition"
    ],
    "Language": [
        "lang_function", "lang_grammar", "lang_text"
    ]
}

MODEL_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac"
]


def load_results(results_dir: Path) -> dict:
    """Load all result JSON files organized by model name."""
    all_results = {}

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
        return all_results

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        all_results[model_name] = {}

        for json_file in sorted(model_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                task_name = json_file.stem
                all_results[model_name][task_name] = data
            except (json.JSONDecodeError, IOError) as e:
                print(f"  Warning: Skipped {json_file}: {e}", file=sys.stderr)

    return all_results


def extract_scores(results_data: dict, metric_filter: str = "acc,none") -> dict:
    """Extract accuracy scores from lm-evaluation-harness result format."""
    scores = {}
    results = results_data.get("results", {})

    for task_key, metrics in results.items():
        for metric_name, value in metrics.items():
            if metric_name == metric_filter and isinstance(value, (int, float)):
                scores[task_key] = round(value * 100, 2)

    return scores


def build_score_table(all_results: dict) -> tuple:
    """Build comparison data: (models list, {task: {model: score}})."""
    models = sorted(all_results.keys())
    table = {}

    for model_name, tasks in all_results.items():
        for task_file, data in tasks.items():
            task_scores = extract_scores(data)
            for task_key, score in task_scores.items():
                if task_key not in table:
                    table[task_key] = {}
                table[task_key][model_name] = score

    return models, table


def compute_overall(score_table: dict, models: list) -> dict:
    """Compute overall average across all tasks for each model."""
    model_totals = {m: [] for m in models}
    for task, model_scores in score_table.items():
        for model in models:
            if model in model_scores:
                model_totals[model].append(model_scores[model])

    return {
        model: round(sum(scores) / len(scores), 2) if scores else 0
        for model, scores in model_totals.items()
    }


def generate_html(all_results: dict, title: str, output_path: Path):
    """Generate standalone HTML report."""
    models, score_table = build_score_table(all_results)

    if not models:
        print("No results found. Creating empty report.", file=sys.stderr)
        models = []
        score_table = {}

    overall = compute_overall(score_table, models)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tasks_sorted = sorted(score_table.keys())
    color_map = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(models)}

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8f9fa;
            color: #333;
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            color: #1a1a2e;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 2rem;
            font-size: 0.9rem;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        .card h2 {{
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: #1a1a2e;
            border-bottom: 2px solid #4e79a7;
            padding-bottom: 0.5rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        th, td {{
            padding: 0.6rem 0.8rem;
            text-align: center;
            border: 1px solid #e9ecef;
        }}
        th {{
            background: #f1f3f5;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        th:first-child, td:first-child {{
            text-align: left;
            font-weight: 500;
        }}
        tr:hover {{ background: #f8f9fa; }}
        .best {{ background: #d4edda; font-weight: 700; }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 1rem 0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        .summary-item {{
            background: #f1f3f5;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}
        .summary-item .model-name {{
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 0.3rem;
        }}
        .summary-item .score {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #1a1a2e;
        }}
        .no-data {{
            text-align: center;
            color: #999;
            padding: 3rem;
            font-style: italic;
        }}
        footer {{
            text-align: center;
            margin-top: 2rem;
            color: #999;
            font-size: 0.8rem;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>{title}</h1>
    <p class="subtitle">Generated: {timestamp} | Models: {len(models)} | Tasks: {len(tasks_sorted)}</p>
"""

    # Overall summary cards
    if models:
        html += '    <div class="card">\n'
        html += '        <h2>Overall Accuracy (%)</h2>\n'
        html += '        <div class="summary-grid">\n'
        for model in sorted(overall, key=lambda m: -overall[m]):
            html += f'            <div class="summary-item">\n'
            html += f'                <div class="model-name">{model}</div>\n'
            html += f'                <div class="score">{overall[model]:.1f}%</div>\n'
            html += f'            </div>\n'
        html += '        </div>\n'
        html += '    </div>\n\n'

    # Chart
    if models and tasks_sorted:
        html += '    <div class="card">\n'
        html += '        <h2>Score Comparison</h2>\n'
        html += '        <div class="chart-container"><canvas id="mainChart"></canvas></div>\n'
        html += '    </div>\n\n'

    # Detail table
    if models and tasks_sorted:
        html += '    <div class="card">\n'
        html += '        <h2>Detailed Results</h2>\n'
        html += '        <div style="overflow-x: auto;">\n'
        html += '        <table>\n'
        html += '            <thead><tr><th>Task</th>'
        for m in models:
            html += f'<th>{m}</th>'
        html += '</tr></thead>\n'
        html += '            <tbody>\n'

        for task in tasks_sorted:
            html += '            <tr>'
            html += f'<td>{task}</td>'
            task_scores = score_table[task]
            max_score = max(task_scores.values()) if task_scores else 0
            for m in models:
                score = task_scores.get(m)
                if score is not None:
                    css = ' class="best"' if score == max_score and len(task_scores) > 1 else ''
                    html += f'<td{css}>{score:.2f}</td>'
                else:
                    html += '<td>-</td>'
            html += '</tr>\n'

        # Overall row
        html += '            <tr style="font-weight:700; background:#e9ecef;">'
        html += '<td>Overall</td>'
        max_overall = max(overall.values()) if overall else 0
        for m in models:
            css = ' class="best"' if overall[m] == max_overall and len(models) > 1 else ''
            html += f'<td{css}>{overall[m]:.2f}</td>'
        html += '</tr>\n'

        html += '            </tbody>\n'
        html += '        </table>\n'
        html += '        </div>\n'
        html += '    </div>\n\n'
    else:
        html += '    <div class="card"><p class="no-data">No evaluation results found. Run Phase 1 or Phase 2 evaluations first.</p></div>\n'

    # Chart.js script
    if models and tasks_sorted:
        datasets_js = []
        for i, model in enumerate(models):
            data_points = [score_table[t].get(model, "null") for t in tasks_sorted]
            data_str = ", ".join(str(d) if d != "null" else "null" for d in data_points)
            datasets_js.append(
                f'{{ label: "{model}", data: [{data_str}], '
                f'backgroundColor: "{color_map[model]}80", '
                f'borderColor: "{color_map[model]}", borderWidth: 1 }}'
            )

        labels_js = ", ".join(f'"{t}"' for t in tasks_sorted)

        html += f"""    <script>
    new Chart(document.getElementById('mainChart'), {{
        type: 'bar',
        data: {{
            labels: [{labels_js}],
            datasets: [{", ".join(datasets_js)}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ position: 'top' }},
                title: {{ display: false }}
            }},
            scales: {{
                y: {{
                    beginAtZero: true,
                    max: 100,
                    title: {{ display: true, text: 'Accuracy (%)' }}
                }},
                x: {{
                    ticks: {{ maxRotation: 45, minRotation: 45 }}
                }}
            }}
        }}
    }});
    </script>
"""

    html += f"""
    <footer>
        Generated by LM Eval Builder Lab | <a href="https://github.com/hyogrin/evaluate-llm-on-korean-dataset">evaluate-llm-on-korean-dataset</a>
    </footer>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report generated: {output_path}")
    print(f"  Models: {len(models)}")
    print(f"  Tasks: {len(tasks_sorted)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML benchmark report from LMEvalJob results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("../results"),
        help="Path to results directory (default: ../results)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report.html"),
        help="Output HTML file path (default: report.html)"
    )
    parser.add_argument(
        "--title",
        default="Korean LLM Benchmark Report",
        help="Report title"
    )

    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir.resolve()}")
    all_results = load_results(args.results_dir)

    if not all_results:
        print("Warning: No result files found. Generating empty report.")

    generate_html(all_results, args.title, args.output)


if __name__ == "__main__":
    main()
