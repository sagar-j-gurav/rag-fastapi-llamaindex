"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types."""
    FAQ = "faq"
    WEBSITE = "website"
    POLICY = "policy"


class UploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    message: str
    document_id: Optional[str] = None
    document_type: Optional[DocumentType] = None
    chunks_created: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    document_type: Optional[DocumentType] = Field(None, description="Filter by document type")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    include_sources: bool = Field(True, description="Include source chunks in response")
    
    @validator("query")
    def clean_query(cls, v):
        """Clean and validate query."""
        return v.strip()


class SourceChunk(BaseModel):
    """Model for a source chunk."""
    text: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    query: str
    answer: str
    sources: Optional[List[SourceChunk]] = None
    document_types_searched: List[str]
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "unhealthy"]
    chroma_connected: bool
    openai_configured: bool
    total_documents: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DocumentMetadata(BaseModel):
    """Metadata for indexed documents."""
    document_type: DocumentType
    source_file: str
    upload_timestamp: datetime
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    additional_metadata: Optional[Dict[str, Any]] = None