"""Dataset loading and preprocessing configuration for Korean MCQ benchmarks."""

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Configuration for a single benchmark dataset."""

    hf_path: str
    hf_name: str | None = None
    split: str = "test"
    question_col: str = "question"
    choices_col: str = "choices"  # column name for list-type choices, or None if separate cols
    answer_col: str = "answer"
    context_col: str | None = None
    category_col: str | None = None
    id_col: str | None = None
    num_choices: int = 5
    answer_offset: int = 0  # 0-indexed vs 1-indexed answer mapping
    choice_labels: list[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E"])
    # If True, choices are in separate columns named A, B, C, D (like KMMLU)
    choices_in_separate_cols: bool = False
    # If True, load all configs from the dataset (for datasets with per-subject configs)
    load_all_configs: bool = False
    # Fixed choices (for boolean tasks like BoolQ where choices aren't in the dataset)
    fixed_choices: list[str] | None = None
    # If True, the question column already contains the full formatted prompt
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
        answer_offset=1,  # KMMLU answer is 1-indexed (1=A, 2=B, 3=C, 4=D)
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
        answer_offset=0,  # label: 0 -> A (거짓), 1 -> B (참)... but we want 1=A(참), 0=B(거짓)
        choice_labels=["A", "B"],
        fixed_choices=["참", "거짓"],  # A=참(True), B=거짓(False)
    ),
}


def normalize_answer(raw_answer, config: DatasetConfig, choices: list[str] | None = None) -> str:
    """Convert raw answer value to letter (A/B/C/D/E).

    Handles various formats:
      - Already a letter: "A", "B", etc.
      - Parenthesized letter: "(A)", "(B)", etc.
      - Integer index: 1 -> "A" (with offset), 0 -> "A" (without offset)
      - Numeric string: "1" -> index
      - Choice text: match against the choices list
      - KoBEST BoolQ special: label 1 (True) -> A, label 0 (False) -> B
    """
    # Special handling for KoBEST BoolQ: label 1=참(A), 0=거짓(B)
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

    if isinstance(raw_answer, str) and len(raw_answer) == 1 and raw_answer.upper() in "ABCDE":
        return raw_answer.upper()

    # Try integer-based mapping first (most common for KMMLU)
    try:
        idx = int(raw_answer) - config.answer_offset
        if 0 <= idx < config.num_choices:
            return chr(65 + idx)
    except (ValueError, TypeError):
        pass

    # Try matching against choices (answer is the choice text itself, e.g. CLIcK)
    if choices and isinstance(raw_answer, str):
        for i, choice in enumerate(choices):
            if choice.strip() == raw_answer.strip():
                return chr(65 + i)

    return str(raw_answer).upper()


def get_choices_from_example(example: dict, config: DatasetConfig) -> list[str]:
    """Extract choices list from a dataset example."""
    import ast

    if config.fixed_choices:
        return config.fixed_choices
    if config.choices_in_separate_cols:
        return [example[label] for label in config.choice_labels]

    raw = example[config.choices_col]

    # Handle stringified list (e.g. HAE-RAE: "['추석', '제사', ...]")
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return [c.strip() for c in raw.split(",")]

    return raw
