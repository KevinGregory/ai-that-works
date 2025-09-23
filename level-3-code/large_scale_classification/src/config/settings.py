"""Application settings and configuration."""

import pathlib

from pydantic_settings import BaseSettings

from src.shared import constants as C
from src.shared.enums import NarrowingStrategy

CWD = pathlib.Path(__file__).parent


class Settings(BaseSettings):
    """Application configuration settings."""

    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    # Classification
    narrowing_strategy: NarrowingStrategy = NarrowingStrategy.HYBRID
    max_narrowed_categories: int = 50
    # Hybrid narrowing specific settings
    max_embedding_candidates: int = 100  # How many categories embedding stage returns
    max_final_categories: int = 15  # How many categories LLM stage returns3
    # Data
    categories_file_path: pathlib.Path = CWD.parents[1] / C.DATA / C.CATEGORIES_TXT
    # Expanded text
    expand_user_query: bool = True

    # Config
    class Config:
        """Configuration for the settings."""

        env_file = CWD.parents[1] / ".env"


settings = Settings()
