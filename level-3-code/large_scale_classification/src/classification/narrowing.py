"""Different strategies for narrowing down the category set."""

from abc import ABC, abstractmethod

from src.baml_client import b
from src.baml_client.type_builder import TypeBuilder
from src.classification.embeddings import EmbeddingService
from src.classification.vector_store import CategoryVectorStore
from src.config.settings import settings
from src.data.models import Category
from src.shared.enums import NarrowingStrategy
from src.shared.logger import get_logger

NARROWED_CATEGORIES_BUFFER = 2


class NarrowingStrategyBase(ABC):
    """Abstract base for category narrowing strategies."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        use_vector_store: bool = True,
    ) -> None:
        """Initialize base narrowing strategy.

        Args:
            embedding_service: The embedding service for similarity calculations.
            use_vector_store: Whether to use vector store for faster search.
        """
        self.logger = get_logger(__name__)
        self.embedding_service = embedding_service
        self._vector_store: CategoryVectorStore | None = None

        if embedding_service and use_vector_store and CategoryVectorStore.is_available():
            try:
                self._vector_store = CategoryVectorStore()
                self.logger.success(f"{self.__class__.__name__} using ChromaDB vector store")
            except Exception as e:
                self.logger.warning(
                    f"{self.__class__.__name__} failed to load vector store, falling back to in-memory: {e}"
                )
        elif embedding_service and use_vector_store:
            self.logger.warning(f"Vector store not available for {self.__class__.__name__}, using in-memory search")

    @abstractmethod
    def narrow(self, text: str, categories: list[Category]) -> list[Category]:
        """Narrow categories from all categories to a smaller set based on input text.

        Args:
            text: The text to narrow categories based on.
            categories: The categories to narrow.

        Returns:
            The narrowed categories.
        """
        pass

    def _narrow_with_embedding_similarity(
        self, text: str, categories: list[Category], max_results: int
    ) -> list[Category]:
        """Narrow categories with embedding similarity.

        Args:
            text: The text to narrow categories based on.
            categories: The categories to narrow.
            max_results: Maximum number of categories to return.

        Returns:
            The narrowed categories.
        """
        if not categories or not self.embedding_service:
            return categories[:max_results] if categories else []
        if self._vector_store is not None:
            return self._narrow_with_vector_store(text, max_results)
        return self._narrow_in_memory(text, categories, max_results)

    def _narrow_with_vector_store(self, text: str, max_results: int) -> list[Category]:
        """Use vector store for fast similarity search.

        Args:
            text: The text to narrow categories based on.
            max_results: Maximum number of categories to return.

        Returns:
            The narrowed categories.
        """
        if self._vector_store is None or self.embedding_service is None:
            raise RuntimeError("Vector store or embedding service is not available")

        text_embedding = self.embedding_service.embed_text(text)
        similar_categories = self._vector_store.find_similar_categories(
            query_embedding=text_embedding,
            n_results=max_results * NARROWED_CATEGORIES_BUFFER,
        )
        return similar_categories[:max_results]

    def _narrow_in_memory(self, text: str, categories: list[Category], max_results: int) -> list[Category]:
        """In-memory similarity search.

        Args:
            text: The text to narrow categories based on.
            categories: The categories to narrow.
            max_results: Maximum number of categories to return.

        Returns:
            The narrowed categories.
        """
        category_embeddings: list[tuple[Category, list[float]]] = []
        scored_categories: list[tuple[Category, float]] = []
        if not self.embedding_service:
            self.logger.warning("Embedding service is not available, returning all categories")
            return categories
        for category in categories:
            embedding = self.embedding_service.embed_category(category)
            category_embeddings.append((category, embedding))
        text_embedding = self.embedding_service.embed_text(text)
        for category, embedding in category_embeddings:
            similarity = self.embedding_service.compute_similarity(text_embedding, embedding)
            scored_categories.append((category, similarity))
        scored_categories.sort(key=lambda x: x[1], reverse=True)
        return [category for category, _ in scored_categories[:max_results]]

    def _narrow_with_llm(self, text: str, categories: list[Category], max_results: int) -> list[Category]:
        """Narrow categories with LLM.

        Args:
            text: The text to narrow categories based on.
            categories: The categories to narrow.
            max_results: Maximum number of categories to return.

        Returns:
            The narrowed categories.
        """
        if not categories:
            return []
        if len(categories) <= max_results:
            return categories
        tb = TypeBuilder()
        category_map: dict[str, Category] = {}
        alias_to_category: dict[str, Category] = {}
        for i, category in enumerate(categories):
            alias = f"k{i}"
            val = tb.Category.add_value(category.name)
            val.alias(alias)
            val.description(category.llm_description)
            category_map[category.name] = category
            alias_to_category[alias] = category

        try:
            selected_items = b.PickBestCategories(text, count=max_results, baml_options={"tb": tb})
            selected_categories = []
            for item in selected_items:
                if item in category_map:
                    selected_categories.append(category_map[item])
                elif item in alias_to_category:
                    selected_categories.append(alias_to_category[item])
            return selected_categories
        except Exception as e:
            self.logger.warning(f"LLM narrowing failed: {e}")
            return categories[:max_results]


class LLMBasedNarrowing(NarrowingStrategyBase):
    """Uses LLM for category narrowing."""

    def __init__(self) -> None:
        """Initialize the LLM-based narrowing strategy."""
        super().__init__()

    def narrow(self, text: str, categories: list[Category]) -> list[Category]:
        """Narrow using LLM understanding.

        Args:
            text: The text to narrow categories based on.
            categories: The categories to narrow.

        Returns:
            The narrowed categories.
        """
        try:
            return self._narrow_with_llm(text, categories, settings.max_narrowed_categories)
        except Exception as e:
            # Return all categories if LLM fails - let the user handle the failure
            self.logger.error(f"LLM narrowing failed: {e}, returning all categories")
            return categories[: settings.max_narrowed_categories]


class HybridNarrowing(NarrowingStrategyBase):
    """Combines embedding and LLM strategies with two-stage narrowing."""

    def __init__(self, embedding_service: EmbeddingService, use_vector_store: bool = True):
        """Initialize the hybrid narrowing strategy.

        Args:
            embedding_service: The module's embedding service.
            use_vector_store: Whether to use the ChromaDB vector store for faster search.
        """
        super().__init__(embedding_service, use_vector_store)
        self._use_hybrid = self._validate_hybrid_settings()

    def _validate_hybrid_settings(self) -> bool:
        """Validate that hybrid strategy settings are compatible.

        Returns:
            bool: True if settings are valid for hybrid strategy, False otherwise.
        """
        if settings.max_embedding_candidates < settings.max_final_categories:
            self.logger.warning(
                f"Invalid hybrid strategy settings: max_embedding_candidates ({settings.max_embedding_candidates}) "
                f"< max_final_categories ({settings.max_final_categories}). "
                "Falling back to embedding-only strategy."
            )
            return False
        return True

    def narrow(self, text: str, categories: list[Category]) -> list[Category]:
        """Use embedding first to get 10 candidates, then LLM to refine to 3.

        Args:
            text: The text to narrow categories based on.
            categories: The categories to narrow.

        Returns:
            The narrowed categories.
        """
        if not categories:
            return []
        # If hybrid settings are invalid, fall back to embedding-only strategy
        if not self._use_hybrid:
            return self._narrow_with_embedding_only(text, categories)
        embedding_candidates = self._narrow_with_embedding(text, categories)
        return self._narrow_with_llm_stage(text, embedding_candidates)

    def narrow_with_stages(self, text: str, categories: list[Category]) -> dict:
        """Use embedding first to get candidates, then LLM to refine, returning stage info.

        Args:
            text: The text to narrow categories based on.
            categories: The categories to narrow.

        Returns:
            Dictionary containing stage results and final candidates.
        """
        if not categories:
            return {"embedding_candidates": [], "llm_candidates": [], "final_candidates": []}

        # If hybrid settings are invalid, fall back to embedding-only strategy
        if not self._use_hybrid:
            embedding_candidates = self._narrow_with_embedding_only(text, categories)
            return {
                "embedding_candidates": embedding_candidates,
                "llm_candidates": [],  # No LLM stage in embedding-only
                "final_candidates": embedding_candidates,
            }

        # Get embedding stage results
        embedding_candidates = self._narrow_with_embedding(text, categories)

        # Get LLM stage results
        llm_candidates = self._narrow_with_llm_stage(text, embedding_candidates)

        return {
            "embedding_candidates": embedding_candidates,
            "llm_candidates": llm_candidates,
            "final_candidates": llm_candidates,
        }

    def _narrow_with_embedding_only(self, text: str, categories: list[Category]) -> list[Category]:
        """Use embedding-only strategy when hybrid settings are invalid.

        Args:
            text: The text to narrow categories based on.
            categories: The categories to narrow.

        Returns:
            The narrowed categories (up to max_final_categories).
        """
        return self._narrow_with_embedding_similarity(text, categories, settings.max_final_categories)

    def _narrow_with_embedding(self, text: str, categories: list[Category]) -> list[Category]:
        """Use embedding similarity to narrow to max_embedding_candidates.

        Args:
            text: The text to narrow categories based on.
            categories: The categories to narrow.

        Returns:
            The narrowed categories.
        """
        return self._narrow_with_embedding_similarity(text, categories, settings.max_embedding_candidates)

    def _narrow_with_llm_stage(self, text: str, categories: list[Category]) -> list[Category]:
        """Use LLM to narrow to final category count.

        Args:
            text: The text to narrow categories based on.
            categories: The categories to narrow.

        Returns:
            The narrowed categories.
        """
        try:
            return self._narrow_with_llm(text, categories, settings.max_final_categories)
        except Exception as e:
            self.logger.warning(f"LLM narrowing failed: {e}, returning top embedding candidates")
            return categories[: settings.max_final_categories]


class CategoryNarrower:
    """Main narrowing service that delegates to strategies."""

    def __init__(self, embedding_service: EmbeddingService, use_vector_store: bool = True) -> None:
        """Initialize the category narrowing service.

        Args:
            embedding_service: The module's embedding service.
            use_vector_store: Whether to use the ChromaDB vector store for faster search.
        """
        self.embedding_service = embedding_service
        self._strategy_map = {
            NarrowingStrategy.HYBRID: lambda: HybridNarrowing(embedding_service, use_vector_store),
        }

    def narrow_categories(self, text: str, categories: list[Category]) -> list[Category]:
        """Narrow categories using the configured strategy.

        Args:
            text: The text for which to narrow the categories.
            categories: The categories to narrow.

        Returns:
            The narrowed categories.
        """
        strategy_class = self._strategy_map[settings.narrowing_strategy]
        strategy = strategy_class()
        return strategy.narrow(text, categories)

    def narrow_categories_with_stages(self, text: str, categories: list[Category]) -> dict:
        """Narrow categories using the configured strategy, returning stage information.

        Args:
            text: The text for which to narrow the categories.
            categories: The categories to narrow.

        Returns:
            Dictionary containing stage results and final candidates.
        """
        strategy_class = self._strategy_map[settings.narrowing_strategy]
        strategy = strategy_class()

        # Check if strategy supports stage information
        if hasattr(strategy, "narrow_with_stages"):
            return strategy.narrow_with_stages(text, categories)
        else:
            # Fallback for strategies that don't support stages
            final_candidates = strategy.narrow(text, categories)
            return {
                "embedding_candidates": final_candidates,  # Assume all came from embedding
                "llm_candidates": [],  # No LLM stage
                "final_candidates": final_candidates,
            }
