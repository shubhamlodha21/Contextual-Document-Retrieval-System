# document_processor.py
import re
import logging
from typing import List, Optional
import PyPDF2
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    """
    A comprehensive document processing class for handling various document types
    and preparing text for embedding and retrieval.
    
    Key Features:
    - Support for PDF and TXT file types
    - Text normalization and cleaning
    - Intelligent text chunking
    - Embedding generation
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2", 
                 chunk_size: int = 50, 
                 chunk_overlap: int = 10):
        """
        Initialize the DocumentProcessor with configurable parameters.
        
        Args:
            embedding_model (str): Name of the sentence transformer model
            chunk_size (int): Number of words per chunk
            chunk_overlap (int): Number of words to overlap between chunks
        """
        self.file_path: Optional[str] = None
        self.content: Optional[str] = None
        self.chunks: Optional[List[str]] = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: str) -> str:
        """
        Load and process document content with robust error handling.
        
        Args:
            file_path (str): Path to the document file
        
        Returns:
            str: Processed document content
        
        Raises:
            ValueError: For unsupported file types or empty documents
        """
        self.file_path = file_path
        
        try:
            content = ""
            if file_path.endswith(".pdf"):
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    content = " ".join(page.extract_text() for page in reader.pages)
            elif file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            # Normalize text
            content = re.sub(r'\s+', ' ', content).strip()
            
            if not content:
                self.logger.warning(f"No content extracted from {file_path}")
                raise ValueError("Document is empty")
            
            self.content = content
            self.logger.info(f"Successfully loaded document: {file_path}")
            return content
        
        except Exception as e:
            self.logger.error(f"Error loading document {file_path}: {e}")
            raise

    def chunk_text(self) -> List[str]:
        """
        Intelligently chunk text with configurable size and overlap.
        
        Returns:
            List[str]: List of text chunks
        """
        if not self.content:
            raise ValueError("Document not loaded. Call load_document() first.")
        
        # Split content into words
        words = self.content.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = words[i:i + self.chunk_size]
            chunks.append(' '.join(chunk))
        
        # Remove duplicates while preserving order
        unique_chunks = []
        seen = set()
        for chunk in chunks:
            if chunk not in seen:
                unique_chunks.append(chunk)
                seen.add(chunk)
        
        self.chunks = unique_chunks
        self.logger.info(f"Created {len(self.chunks)} unique chunks")
        return self.chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for given texts.
        
        Args:
            texts (List[str]): List of text chunks
        
        Returns:
            List[List[float]]: List of embeddings
        """
        try:
            embeddings = self.embedding_model.encode(texts).tolist()
            return embeddings
        except Exception as e:
            self.logger.error(f"Embedding generation error: {e}")
            raise

    def process_query(self, query: str) -> List[float]:
        """
        Process and embed a query.
        
        Args:
            query (str): Input query
        
        Returns:
            List[float]: Query embedding
        """
        processed_query = query.lower().strip()
        query_embedding = self.embedding_model.encode([processed_query])[0].tolist()
        return query_embedding
