"""Configuration module for AI Commerce Intelligence Platform."""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Platform application settings."""

    model_config = SettingsConfigDict(
        env_prefix="COMMERCE_AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General Environment
    env: str = Field(default="development", description="Environment: development, test, production")
    debug: bool = Field(default=True, description="Debug mode flag")

    # Base Paths (default relative to workspace root)
    base_dir: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / "data")
    raw_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "raw")
    processed_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "processed")
    sample_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "sample")

    # Synthetic Generator Defaults
    gen_skus: int = Field(default=500, description="Default number of SKUs to generate")
    gen_warehouses: int = Field(default=5, description="Default number of warehouses")
    gen_channels: int = Field(default=4, description="Default number of sales channels")
    gen_suppliers: int = Field(default=20, description="Default number of suppliers")
    gen_historical_days: int = Field(default=730, description="Default historical days (2 years)")
    gen_seed: int = Field(default=42, description="Random seed for reproducibility")

    # API Server (Future)
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")


# Singleton instance
settings = Settings()
