"""Main FastAPI application for RAG system."""

import logging
import time
from pathlib import Path
from typing import Optional
import tempfile
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import get_settings
from models import (
    DocumentType,
    UploadResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    ErrorResponse
)
from document_processor import DocumentProcessor
from rag_engine import RAGEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
rag_engine: Optional[RAGEngine] = None
doc_processor: Optional[DocumentProcessor] = None
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global rag_engine, doc_processor
    
    # Startup
    logger.info("Starting RAG FastAPI application...")
    try:
        rag_engine = RAGEngine()
        doc_processor = DocumentProcessor()
        logger.info("RAG engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG FastAPI application...")


# Initialize FastAPI app
app = FastAPI(
    title="RAG FastAPI Application",
    description="Production-ready RAG system with LlamaIndex and ChromaDB",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "RAG FastAPI Application",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "query": "/api/query",
            "health": "/api/health",
            "stats": "/api/stats"
        }
    }


@app.post(
    "/api/upload",
    response_model=UploadResponse,
    tags=["Documents"],
    summary="Upload and index a document"
)
async def upload_document(
    file: UploadFile = File(...),
    document_type: DocumentType = Form(...)
):
    """Upload and index a document.
    
    Accepts either:
    - Excel file (.xlsx) for FAQ documents
    - Text file (.txt) for website or policy documents
    """
    start_time = time.time()
    
    # Validate file size
    if file.size > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {settings.max_upload_size_mb}MB limit"
        )
    
    # Validate file type
    file_extension = Path(file.filename).suffix.lower()
    
    if document_type == DocumentType.FAQ and file_extension not in ['.xlsx', '.xls']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="FAQ documents must be Excel files (.xlsx or .xls)"
        )
    
    if document_type in [DocumentType.WEBSITE, DocumentType.POLICY] and file_extension != '.txt':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Website and policy documents must be text files (.txt)"
        )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        try:
            # Save uploaded file
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            tmp_path = Path(tmp_file.name)
            
            # Process document
            documents, metadata = doc_processor.process_document(
                file_path=tmp_path,
                document_type=document_type,
                additional_metadata={"original_filename": file.filename}
            )
            
            # Index documents
            stats = rag_engine.index_documents(documents, document_type)
            
            # Prepare response
            response = UploadResponse(
                success=True,
                message=f"Successfully indexed {file.filename}",
                document_id=metadata.get("document_id"),
                document_type=document_type,
                chunks_created=stats["chunks_created"],
                metadata={
                    **metadata,
                    **stats,
                    "processing_time_seconds": time.time() - start_time
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process document: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()


@app.post(
    "/api/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Query the RAG system"
)
async def query_rag(request: QueryRequest):
    """Query the RAG system with a user question."""
    start_time = time.time()
    
    try:
        # Execute query
        result = rag_engine.query(
            query_text=request.query,
            document_type=request.document_type,
            top_k=request.top_k,
            include_sources=request.include_sources
        )
        
        # Determine which document types were searched
        if request.document_type:
            doc_types_searched = [request.document_type.value]
        else:
            stats = rag_engine.get_stats()
            doc_types_searched = list(stats.get("document_types", {}).keys())
        
        # Prepare response
        response = QueryResponse(
            query=request.query,
            answer=result["answer"],
            sources=result.get("sources") if request.include_sources else None,
            document_types_searched=doc_types_searched,
            processing_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "top_k_used": request.top_k,
                "model_used": settings.llm_model
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint"
)
async def health_check():
    """Check system health."""
    try:
        health = rag_engine.health_check()
        stats = rag_engine.get_stats()
        
        return HealthResponse(
            status="healthy" if all(health.values()) else "unhealthy",
            chroma_connected=health["chroma_connected"],
            openai_configured=health["openai_configured"],
            total_documents=stats["total_vectors"]
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            chroma_connected=False,
            openai_configured=False
        )


@app.get(
    "/api/stats",
    tags=["System"],
    summary="Get system statistics"
)
async def get_stats():
    """Get RAG system statistics."""
    try:
        stats = rag_engine.get_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.delete(
    "/api/clear",
    tags=["System"],
    summary="Clear all indexed documents"
)
async def clear_index():
    """Clear all documents from the index."""
    try:
        rag_engine.clear_index()
        return {
            "success": True,
            "message": "Index cleared successfully"
        }
    except Exception as e:
        logger.error(f"Clear index error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear index: {str(e)}"
        )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.reload else None
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info"
    )