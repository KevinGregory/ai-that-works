"""Fleshes out the user's query using LLM."""

from src.baml_client import b


def expand_user_query(text: str) -> str:
    """Expand the user's query using LLM.

    Args:
        text: The user's query to expand.

    Returns:
        The expanded user's query.
    """
    expanded_text = b.ExpandUserQuery(text)
    return expanded_text
