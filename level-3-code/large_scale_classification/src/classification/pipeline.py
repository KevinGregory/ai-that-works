"""Orchestrates the full classification process."""

import time


from src.classification import expander
from src.classification.embeddings import EmbeddingService
from src.classification.narrowing import CategoryNarrower
from src.classification.selection import CategorySelector
from src.classification.vector_store import CategoryVectorStore
from src.config.settings import settings
from src.data.category_loader import CategoryLoader
from src.data.models import Category, ClassificationResult

from src.shared import constants as C
from src.shared.logger import get_logger

class ClassificationPipeline:
    """Orchestrates the full classification process."""

    def __init__(self, use_vector_store: bool = True) -> None:
        """Initialize the classification pipeline.

        Args:
            use_vector_store: Whether to use the vector store for caching embeddings.
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing Classification Pipeline...")

        self.category_loader = CategoryLoader()
        self.embedding_service = EmbeddingService(use_vector_store=use_vector_store)
        self.narrower = CategoryNarrower(self.embedding_service, use_vector_store=use_vector_store)
        self.selector = CategorySelector()
        self._categories_cache: list[Category] = []

        if use_vector_store and CategoryVectorStore.is_available():
            try:
                store = CategoryVectorStore()
                info = store.get_collection_info()
                self.logger.info(f"Vector store loaded: {info['count']} categories cached")
            except Exception as e:
                self.logger.warning(f"Vector store available but failed to load: {e}")
        elif use_vector_store:
            self.logger.info("Vector store will be created automatically as categories are processed")
        else:
            self.logger.info("Using in-memory embedding cache only")

        self.logger.success("Classification Pipeline initialized")

    def _get_categories(self) -> list[Category]:
        """Get categories.

        Returns:
            The categories.
        """
        if not self._categories_cache:
            self._categories_cache = self.category_loader.load_categories()
        return self._categories_cache

    def classify(self, text: str, max_candidates: int | None = None) -> ClassificationResult:
        """Full classification pipeline with detailed results.

        Args:
            text: The text to classify.
            max_candidates: The maximum number of candidates to return.

        Returns:
            The classification result.
        """
        start_time = time.time()
        categories = self._get_categories()
        self.logger.info(f"Classifying text with {len(categories)} total categories")
        narrowing_start = time.time()
        narrowing_results = self.narrower.narrow_categories_with_stages(text, categories)
        narrowed_categories = narrowing_results['final_candidates']
        narrowing_time_ms = (time.time() - narrowing_start) * 1000
        if max_candidates and len(narrowed_categories) > max_candidates:
            narrowed_categories = narrowed_categories[:max_candidates]
        self.logger.info(f"Narrowed to {len(narrowed_categories)} categories in {narrowing_time_ms:.1f}ms")
        if settings.expand_user_query:
            expanding_text_start = time.time()
            text = expander.expand_user_query(text)
            expanding_text_time_ms = (time.time() - expanding_text_start) * 1000
            self.logger.info(f"Expanded the user's query in {expanding_text_time_ms:.1f}ms")
        selection_start = time.time()
        selected_category = self.selector.select_best_category(text, narrowed_categories)
        selection_time_ms = (time.time() - selection_start) * 1000
        processing_time_ms = (time.time() - start_time) * 1000
        self.logger.success(f"Selected: {selected_category.path} (total: {processing_time_ms:.1f}ms)")

        return ClassificationResult(
            category=selected_category,
            candidates=narrowed_categories,
            processing_time_ms=processing_time_ms,
            metadata={
                C.TOTAL_CATEGORIES: len(categories),
                C.NARROWED_TO: len(narrowed_categories),
                C.NARROWING_TIME_MS: narrowing_time_ms,
                C.SELECTION_TIME_MS: selection_time_ms,
                C.NARROWING_STRATEGY: settings.narrowing_strategy.value,
                C.VECTOR_STORE_ENABLED: self.embedding_service.vector_store is not None,
            },
            embedding_candidates=narrowing_results.get('embedding_candidates', []),
            llm_candidates=narrowing_results.get('llm_candidates', []),
        )
