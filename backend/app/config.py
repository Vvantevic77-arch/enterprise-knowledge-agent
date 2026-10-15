from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "sqlite+aiosqlite:///./agent.db"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend_url: str = "redis://localhost:6379/1"

    # --- Authentication -----------------------------------------------------
    # HS256 signing key. MUST be overridden in any shared environment; the
    # default is only accepted when app_env is a local development value.
    jwt_secret: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 30
    refresh_token_ttl_days: int = 14
    # Escape hatch for offline demos and tests. When true, the X-User-ID /
    # X-Org-ID headers are accepted as an identity assertion again. Never
    # enable this outside a single-tenant local sandbox.
    allow_header_identity: bool = False

    # --- LLM ----------------------------------------------------------------
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_api_mode: str = "chat_completions"
    openai_reasoning_effort: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    openai_input_cost_per_1k: float = 0.00015
    openai_output_cost_per_1k: float = 0.00060

    # --- Embeddings ---------------------------------------------------------
    # ``deterministic`` keeps the stack offline and reproducible. Point
    # ``embedding_provider`` at ``openai`` (or an OpenAI-compatible gateway via
    # ``openai_base_url``) for real semantic retrieval.
    embedding_provider: str = "deterministic"
    embedding_dimensions: int = 1536
    embedding_model: str = "text-embedding-3-small"
    embedding_revision: str | None = None

    # Local models are optional: install the ``retrieval`` extra before enabling.
    retrieval_candidate_k: int = Field(default=50, ge=5, le=200)
    retrieval_device: str = "cpu"
    reranker_model: str | None = None
    reranker_revision: str | None = None

    # --- Risk policy --------------------------------------------------------
    # A ticket at or above this tier needs two distinct approvers.
    dual_approval_tiers: str = "high"

    logfire_token: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    @field_validator("embedding_revision", "reranker_model", "reranker_revision", mode="before")
    @classmethod
    def _empty_model_options(cls, value):
        return value.strip() or None if isinstance(value, str) else value

    @model_validator(mode="after")
    def _reject_default_secret_outside_dev(self) -> "Settings":
        if self.embedding_provider == "sentence_transformers" and not self.embedding_revision:
            raise ValueError("Set EMBEDDING_REVISION to a fixed model commit before local indexing")
        development = {"development", "dev", "local", "test", "testing", "ci"}
        if self.app_env.lower() not in development and self.jwt_secret.startswith(
            "dev-only-insecure"
        ):
            raise ValueError(
                "JWT_SECRET must be set to a strong random value when APP_ENV is not a development environment"
            )
        return self

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def llm_api_key(self) -> str | None:
        return self.deepseek_api_key if self.llm_provider == "deepseek" else self.openai_api_key

    @property
    def llm_base_url(self) -> str | None:
        return self.deepseek_base_url if self.llm_provider == "deepseek" else self.openai_base_url

    @property
    def llm_model(self) -> str:
        return self.deepseek_model if self.llm_provider == "deepseek" else self.openai_model

    @property
    def llm_api_mode(self) -> str:
        return "chat_completions" if self.llm_provider == "deepseek" else self.openai_api_mode

    @property
    def dual_approval_tier_set(self) -> set[str]:
        return {
            tier.strip().lower() for tier in self.dual_approval_tiers.split(",") if tier.strip()
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
