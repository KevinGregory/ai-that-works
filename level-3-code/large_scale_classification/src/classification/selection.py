"""LLM-based final category selection using BAML."""

from src.baml_client import b
from src.baml_client.type_builder import TypeBuilder
from src.data.models import Category


class CategorySelector:
    """Handles final category selection using LLM/BAML."""

    def select_best_category(self, text: str, candidates: list[Category]) -> Category:
        """Select the single best category from candidates using LLM.

        Args:
            text: The text to classify.
            candidates: The candidates to select from.

        Returns:
            The selected category.
        """
        if not candidates:
            raise ValueError("No candidate categories provided")
        if len(candidates) == 1:
            return candidates[0]
        tb = self._build_dynamic_enum(candidates)
        selected_name = b.PickBestCategory(text, baml_options={"tb": tb})
        for category in candidates:
            if category.name == selected_name:
                return category
        # This should be impossible with BAML, but just in case
        raise ValueError(f"Selected category '{selected_name}' not found in candidates")

    def _build_dynamic_enum(self, categories: list[Category]) -> TypeBuilder:
        """Build BAML TypeBuilder for dynamic categories.

        Args:
            categories: The categories to build the TypeBuilder for.

        Returns:
            The TypeBuilder.
        """
        tb = TypeBuilder()

        for i, category in enumerate(categories):
            val = tb.Category.add_value(category.name)
            val.alias(f"k{i}")
            val.description(category.llm_description)

        return tb
