"""Document processing module for handling different file types."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import hashlib
from datetime import datetime
import re
import logging
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from models import DocumentType
from config import DOCUMENT_TYPE_CONFIG

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process different document types for RAG ingestion."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.processors = {
            DocumentType.FAQ: self._process_faq,
            DocumentType.WEBSITE: self._process_text,
            DocumentType.POLICY: self._process_text
        }
    
    def process_document(
        self,
        file_path: Path,
        document_type: DocumentType,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Process a document based on its type.
        
        Args:
            file_path: Path to the document file
            document_type: Type of document
            additional_metadata: Additional metadata to attach
            
        Returns:
            Tuple of (processed documents, metadata)
        """
        if document_type not in self.processors:
            raise ValueError(f"Unsupported document type: {document_type}")
        
        processor = self.processors[document_type]
        documents = processor(file_path)
        
        # Get chunking configuration for document type
        config = DOCUMENT_TYPE_CONFIG[document_type.value]
        
        # Add metadata to all documents
        metadata = {
            "document_type": document_type.value,
            "source_file": file_path.name,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "document_id": self._generate_document_id(file_path),
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"]
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        for doc in documents:
            doc.metadata.update(metadata)
        
        return documents, metadata
    
    def _process_faq(self, file_path: Path) -> List[Document]:
        """Process FAQ Excel file.
        
        Expected format:
        - Column 1: Question
        - Column 2: Answer
        - Optional Column 3: Category/Tag
        """
        documents = []
        
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Standardize column names
            df.columns = [col.strip().lower() for col in df.columns]
            
            # Identify Q&A columns
            question_col = None
            answer_col = None
            category_col = None
            
            for col in df.columns:
                if 'question' in col or 'q' == col:
                    question_col = col
                elif 'answer' in col or 'a' == col:
                    answer_col = col
                elif 'category' in col or 'tag' in col:
                    category_col = col
            
            if not question_col or not answer_col:
                # Fallback to first two columns
                question_col = df.columns[0]
                answer_col = df.columns[1]
                if len(df.columns) > 2:
                    category_col = df.columns[2]
            
            # Process each FAQ entry
            for idx, row in df.iterrows():
                question = str(row[question_col]).strip()
                answer = str(row[answer_col]).strip()
                
                # Skip empty rows
                if pd.isna(row[question_col]) or pd.isna(row[answer_col]):
                    continue
                
                # Create formatted FAQ text
                faq_text = f"Question: {question}\n\nAnswer: {answer}"
                
                # Create metadata
                faq_metadata = {
                    "faq_id": idx,
                    "question": question[:200],  # Truncate for metadata
                    "faq_type": "qa_pair"
                }
                
                if category_col and not pd.isna(row[category_col]):
                    faq_metadata["category"] = str(row[category_col]).strip()
                
                documents.append(
                    Document(
                        text=faq_text,
                        metadata=faq_metadata
                    )
                )
            
            logger.info(f"Processed {len(documents)} FAQ entries from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing FAQ file: {e}")
            raise ValueError(f"Failed to process FAQ file: {str(e)}")
        
        return documents
    
    def _process_text(self, file_path: Path) -> List[Document]:
        """Process text file (website content or policy document)."""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean the content
            content = self._clean_text(content)
            
            # Split into sections if possible
            sections = self._extract_sections(content)
            
            if sections:
                for section_title, section_content in sections:
                    doc = Document(
                        text=section_content,
                        metadata={
                            "section_title": section_title,
                            "content_type": "section"
                        }
                    )
                    documents.append(doc)
            else:
                # Process as single document
                doc = Document(
                    text=content,
                    metadata={"content_type": "full_document"}
                )
                documents.append(doc)
            
            logger.info(f"Processed text file into {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise ValueError(f"Failed to process text file: {str(e)}")
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\{\}\'\"\n]', '', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([\.,!?;:])', r'\1', text)
        
        return text.strip()
    
    def _extract_sections(self, text: str) -> List[Tuple[str, str]]:
        """Extract sections from text based on headers."""
        sections = []
        
        # Common header patterns
        header_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Z0-9\s]{2,})$',  # All caps headers
            r'^(\d+\.\s+.+)$',  # Numbered sections
            r'^([IVX]+\.\s+.+)$',  # Roman numeral sections
        ]
        
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a header
            is_header = False
            for pattern in header_patterns:
                if re.match(pattern, line, re.MULTILINE):
                    # Save previous section
                    if current_section and current_content:
                        sections.append((current_section, '\n'.join(current_content)))
                    
                    # Start new section
                    current_section = line
                    current_content = []
                    is_header = True
                    break
            
            if not is_header and current_section:
                current_content.append(line)
            elif not is_header and not current_section:
                # First content before any header
                if not sections:
                    current_section = "Introduction"
                current_content.append(line)
        
        # Add last section
        if current_section and current_content:
            sections.append((current_section, '\n'.join(current_content)))
        
        return sections
    
    def _generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID based on file content."""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f"{file_path.stem}_{file_hash[:8]}"
    
    def get_chunk_config(self, document_type: DocumentType) -> Dict[str, int]:
        """Get chunking configuration for document type."""
        config = DOCUMENT_TYPE_CONFIG[document_type.value]
        return {
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"]
        }


def create_text_splitter(document_type: DocumentType) -> SentenceSplitter:
    """Create a text splitter with optimal settings for document type."""
    config = DOCUMENT_TYPE_CONFIG[document_type.value]
    
    return SentenceSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separator=" ",
        paragraph_separator="\n\n",
        secondary_chunking_regex=r"[.!?]\s+"
    )