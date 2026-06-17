"""Response parser for extracting answer letters (A/B/C/D/E) from LLM output."""

import re


_THINK_END_PATTERN = re.compile(r"</think>\s*", re.IGNORECASE)

_ANSWER_PATTERN = re.compile(
    r"""
    (?:
        ^[(\[]?\s*([A-E])\s*[)\]]?\s*$       |  # standalone letter (possibly wrapped)
        (?:answer|정답)[^A-E]*([A-E])          |  # "answer: X" or "정답: X"
        ^([A-E])\s*[.:)\]]                     |  # letter at start followed by separator
        (?:^|\s)([A-E])(?:\s*$)                   # isolated letter
    )
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def _strip_thinking(text: str) -> str:
    """Strip reasoning/thinking content from model output.

    Handles Qwen3/3.5/3.6 and DeepSeek-R1 style models that emit
    <think>...</think> blocks before the actual answer.
    """
    match = _THINK_END_PATTERN.search(text)
    if match:
        return text[match.end():].strip()
    return text


def parse_answer(response: str, num_choices: int = 5) -> str | None:
    """Extract a single answer letter from an LLM response.

    Tries multiple heuristics:
    1. Strip <think>...</think> reasoning blocks if present.
    2. If the response is a single character A-E, return it directly.
    3. Look for common patterns like "Answer: B" or standalone letter.
    4. Find the first valid letter in the response.

    Args:
        response: Raw LLM response string.
        num_choices: Number of valid choices (e.g. 4 means A-D, 5 means A-E).

    Returns:
        Uppercase letter (A-E) or None if parsing fails.
    """
    if not response or not response.strip():
        return None

    text = _strip_thinking(response.strip())
    if not text:
        return None

    valid_letters = {chr(65 + i) for i in range(num_choices)}

    # Fast path: single character response
    if len(text) == 1 and text.upper() in valid_letters:
        return text.upper()

    # Fast path: single character with punctuation like "A." or "(B)"
    cleaned = re.sub(r"[^A-Za-z]", "", text)
    if len(cleaned) == 1 and cleaned.upper() in valid_letters:
        return cleaned.upper()

    # Pattern matching
    for match in _ANSWER_PATTERN.finditer(text):
        for group in match.groups():
            if group and group.upper() in valid_letters:
                return group.upper()

    # Fallback: find the first valid letter in the text (prefer early occurrence)
    for char in text:
        if char.upper() in valid_letters:
            return char.upper()

    return None
