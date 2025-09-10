"""Configuration management for the RAG application."""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    
    # ChromaDB Configuration
    chroma_persist_dir: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="rag_documents", env="CHROMA_COLLECTION_NAME")
    
    # Chunking Configuration (defaults, will be overridden based on document type)
    default_chunk_size: int = Field(default=512, env="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=50, env="DEFAULT_CHUNK_OVERLAP")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=True, env="RELOAD")
    
    # Security
    max_upload_size_mb: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")
    
    # Document Type Specific Chunking Strategies
    faq_chunk_size: int = Field(default=256)
    faq_chunk_overlap: int = Field(default=20)
    
    website_chunk_size: int = Field(default=512)
    website_chunk_overlap: int = Field(default=100)
    
    policy_chunk_size: int = Field(default=384)
    policy_chunk_overlap: int = Field(default=75)
    
    # Retrieval Configuration
    top_k_retrieval: int = Field(default=5)
    rerank_top_n: int = Field(default=3)
    
    @validator("chroma_persist_dir", pre=True)
    def create_persist_dir(cls, v):
        """Ensure the ChromaDB persist directory exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    @validator("llm_temperature")
    def validate_temperature(cls, v):
        """Ensure temperature is within valid range."""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Document type configurations
DOCUMENT_TYPE_CONFIG = {
    "faq": {
        "chunk_size": 256,
        "chunk_overlap": 20,
        "description": "FAQ documents with Q&A pairs",
        "metadata_extractor": "faq_metadata"
    },
    "website": {
        "chunk_size": 512,
        "chunk_overlap": 100,
        "description": "Website content with varied structure",
        "metadata_extractor": "website_metadata"
    },
    "policy": {
        "chunk_size": 384,
        "chunk_overlap": 75,
        "description": "Policy documents with formal structure",
        "metadata_extractor": "policy_metadata"
    }
}