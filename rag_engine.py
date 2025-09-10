"""RAG engine for document indexing and retrieval."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings as LlamaSettings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from config import get_settings
from models import DocumentType, SourceChunk
from document_processor import create_text_splitter

logger = logging.getLogger(__name__)


class RAGEngine:
    """Main RAG engine for document indexing and retrieval."""
    
    def __init__(self):
        """Initialize the RAG engine."""
        self.settings = get_settings()
        self._setup_llm_settings()
        self._initialize_chroma()
        self._initialize_index()
    
    def _setup_llm_settings(self):
        """Configure LlamaIndex global settings."""
        # Set up embedding model
        LlamaSettings.embed_model = OpenAIEmbedding(
            model=self.settings.embedding_model,
            api_key=self.settings.openai_api_key
        )
        
        # Set up LLM
        LlamaSettings.llm = OpenAI(
            model=self.settings.llm_model,
            api_key=self.settings.openai_api_key,
            temperature=self.settings.llm_temperature
        )
        
        # Set default chunk settings
        LlamaSettings.chunk_size = self.settings.default_chunk_size
        LlamaSettings.chunk_overlap = self.settings.default_chunk_overlap
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.settings.chroma_persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.settings.chroma_collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _initialize_index(self):
        """Initialize or load the vector store index."""
        try:
            # Create vector store
            self.vector_store = ChromaVectorStore(
                chroma_collection=self.collection
            )
            
            # Create storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Create or load index
            if self.collection.count() > 0:
                # Load existing index
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=self.storage_context
                )
                logger.info(f"Loaded existing index with {self.collection.count()} vectors")
            else:
                # Create new index
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=self.storage_context
                )
                logger.info("Created new empty index")
                
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            raise
    
    def index_documents(
        self,
        documents: List[Document],
        document_type: DocumentType
    ) -> Dict[str, Any]:
        """Index documents into the vector store.
        
        Args:
            documents: List of documents to index
            document_type: Type of documents being indexed
            
        Returns:
            Indexing statistics
        """
        try:
            # Create appropriate text splitter
            text_splitter = create_text_splitter(document_type)
            
            # Process documents into nodes
            nodes = []
            for doc in documents:
                doc_nodes = text_splitter.get_nodes_from_documents([doc])
                nodes.extend(doc_nodes)
            
            # Add nodes to index
            self.index.insert_nodes(nodes)
            
            stats = {
                "documents_processed": len(documents),
                "chunks_created": len(nodes),
                "total_vectors": self.collection.count()
            }
            
            logger.info(f"Indexed {len(nodes)} chunks from {len(documents)} documents")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        document_type: Optional[DocumentType] = None,
        top_k: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Query the RAG system.
        
        Args:
            query_text: User query
            document_type: Optional filter by document type
            top_k: Number of chunks to retrieve
            include_sources: Whether to include source chunks
            
        Returns:
            Query response with answer and sources
        """
        try:
            # Create retriever with filters
            filters = None
            if document_type:
                filters = {"document_type": document_type.value}
            
            # Configure retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k,
                filters=filters
            )
            
            # Create response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                use_async=False
            )
            
            # Create query engine with postprocessor
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.5)
                ]
            )
            
            # Execute query
            response = query_engine.query(query_text)
            
            # Prepare result
            result = {
                "answer": str(response),
                "sources": []
            }
            
            # Add source chunks if requested
            if include_sources and hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source = SourceChunk(
                        text=node.node.text,
                        score=node.score if hasattr(node, 'score') else 0.0,
                        metadata=node.node.metadata,
                        chunk_id=node.node.id_
                    )
                    result["sources"].append(source.dict())
            
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics."""
        try:
            # Get document type counts
            doc_type_counts = {}
            if self.collection.count() > 0:
                # Query collection for unique document types
                results = self.collection.get(limit=1000)
                if results and 'metadatas' in results:
                    for metadata in results['metadatas']:
                        if metadata and 'document_type' in metadata:
                            doc_type = metadata['document_type']
                            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
            
            return {
                "total_vectors": self.collection.count(),
                "document_types": doc_type_counts,
                "index_ready": self.collection.count() > 0
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_vectors": 0,
                "document_types": {},
                "index_ready": False
            }
    
    def clear_index(self):
        """Clear all documents from the index."""
        try:
            self.chroma_client.delete_collection(self.settings.chroma_collection_name)
            self._initialize_chroma()
            self._initialize_index()
            logger.info("Index cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of RAG engine components."""
        health = {
            "chroma_connected": False,
            "openai_configured": False,
            "index_ready": False
        }
        
        try:
            # Check ChromaDB
            self.chroma_client.heartbeat()
            health["chroma_connected"] = True
        except:
            pass
        
        # Check OpenAI configuration
        health["openai_configured"] = bool(self.settings.openai_api_key)
        
        # Check index
        try:
            health["index_ready"] = self.collection.count() > 0
        except:
            pass
        
        return health