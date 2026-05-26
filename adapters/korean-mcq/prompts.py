"""MCQ prompt templates for Korean LLM benchmarks.

Based on: https://github.com/hyogrin/evaluate-llm-on-korean-dataset/blob/main/config/question_templates.py
"""


def _get_options_string(num_choices: int) -> str:
    letters = [chr(65 + i) for i in range(num_choices)]
    return ", ".join(letters[:-1]) + f", {letters[-1]}" if num_choices > 1 else letters[0]


def _get_options_format(num_choices: int) -> str:
    return ", ".join([f"{chr(65 + i)}: {{{chr(65 + i)}}}" for i in range(num_choices)])


def get_question_template(num_choices: int = 5, with_context: bool = False) -> str:
    options_str = _get_options_string(num_choices)
    options_format = _get_options_format(num_choices)

    if with_context:
        return (
            f"You are taking a multiple choice exam. "
            f"Read the context carefully, then select the single correct answer "
            f"from the choices provided and respond with ONLY the letter ({options_str}). "
            f"Do not include any explanation or additional text.\n\n"
            f"{{CONTEXT}}\n\n"
            f"{{QUESTION}}\n\n"
            f"{options_format}\n"
        )

    return (
        f"You are taking a multiple choice exam. "
        f"Select the single correct answer from the choices provided and "
        f"respond with ONLY the letter ({options_str}). "
        f"Do not include any explanation or additional text.\n\n"
        f"{{QUESTION}}\n\n"
        f"{options_format}\n"
    )


def format_prompt(
    question: str,
    choices: list[str],
    context: str | None = None,
) -> str:
    """Format a single MCQ item into a prompt string.

    Args:
        question: The question text.
        choices: List of answer choice strings (length determines num_choices).
        context: Optional context/paragraph for the question.

    Returns:
        Formatted prompt string ready for LLM input.
    """
    num_choices = len(choices)
    template = get_question_template(num_choices, with_context=context is not None)

    # Build substitution dict
    subs: dict[str, str] = {"QUESTION": question}
    if context is not None:
        subs["CONTEXT"] = context
    for i, choice in enumerate(choices):
        subs[chr(65 + i)] = choice

    prompt = template
    for key, value in subs.items():
        prompt = prompt.replace(f"{{{key}}}", value)

    return prompt
