"""Data models for the large-scale classification system."""

from typing import Any

from pydantic import BaseModel, Field


class Category(BaseModel):
    """Category model supporting hierarchical structure."""

    name: str = Field(..., description="Category display name")
    path: str = Field(
        ...,
        description="Full hierarchical path like /Appliances/Refrigerators/French Door Refrigerators",
    )
    embedding_text: str = Field(..., description="Text optimized for embedding")
    llm_description: str = Field(..., description="Detailed description for LLM")

    @property
    def level(self) -> int:
        """Hierarchy level calculated from path (0=root)."""
        return self.path.count("/") - 1

    @property
    def parent_path(self) -> str:
        """Parent category path calculated from path."""
        return self.path.rsplit("/", 1)[0] if self.path.count("/") > 1 else self.path


class ClassificationRequest(BaseModel):
    """Classification request."""

    text: str = Field(..., description="Text to classify", min_length=1, max_length=10000)
    max_candidates: int | None = Field(5, description="Maximum number of candidates to return", ge=1, le=20)


class ClassificationResult(BaseModel):
    """Classification result."""

    category: Category = Field(..., description="Selected category")
    candidates: list[Category] = Field(default_factory=list, description="Candidate categories")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    # Stage information for pipeline analysis
    embedding_candidates: list[Category] = Field(default_factory=list, description="Categories from embedding stage")
    llm_candidates: list[Category] = Field(default_factory=list, description="Categories from LLM stage")
