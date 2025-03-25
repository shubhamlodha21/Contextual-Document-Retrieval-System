# vector_store.py
import time
import logging
from typing import List, Dict, Optional, Any
from pinecone import Pinecone, ServerlessSpec

class VectorStore:
    """
    A robust vector database implementation using Pinecone.
    
    Key Features:
    - Dynamic index creation
    - Flexible document storage
    - Advanced search capabilities
    - Comprehensive error handling
    """
    
    def __init__(
        self, 
        api_key: str, 
        index_name: str = "document-retrieval", 
        dimension: int = 384,
        cloud: str = 'aws', 
        region: str = 'us-east-1'
    ):
        """
        Initialize Pinecone Vector Database with advanced configuration.
        
        Args:
            api_key (str): Pinecone API key
            index_name (str): Name of the vector index
            dimension (int): Embedding vector dimension
            cloud (str): Cloud provider
            region (str): Cloud region
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=api_key)
            self.index_name = index_name
            self.dimension = dimension
            
            # Create index if not exists with robust retry mechanism
            self._create_index(cloud, region)
            
            # Get the index
            self.index = self.pc.Index(self.index_name)
            self.logger.info("Vector database initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Error initializing vector database: {e}")
            raise

    def _create_index(self, cloud: str, region: str):
        """
        Create Pinecone index with retry and readiness check.
        
        Args:
            cloud (str): Cloud provider
            region (str): Cloud region
        """
        if self.index_name not in self.pc.list_indexes().names():
            self.logger.info(f"Creating new index: {self.index_name}")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            
            # Wait for index readiness
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    index_stats = self.pc.describe_index(self.index_name)
                    if index_stats.status.ready:
                        break
                    time.sleep(5)
                except Exception as wait_error:
                    self.logger.warning(f"Waiting for index (Attempt {attempt+1}): {wait_error}")
                    time.sleep(5)

    def add_documents(
        self, 
        embeddings: List[List[float]], 
        texts: List[str], 
        document_name: str
    ) -> int:
        """
        Add documents to the vector store with comprehensive metadata.
        
        Args:
            embeddings (List[List[float]]): List of document embeddings
            texts (List[str]): Corresponding text chunks
            document_name (str): Name of the source document
        
        Returns:
            int: Number of vectors added
        """
        try:
            # Prepare vectors for upsert
            vectors = [
                (
                    f"{document_name}_chunk_{i}", 
                    embedding, 
                    {"text": chunk, "document": document_name}
                ) 
                for i, (embedding, chunk) in enumerate(zip(embeddings, texts))
            ]
            
            # Upsert in batches
            batch_size = 100
            total_added = 0
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.upsert(batch)
                total_added += len(batch)
            
            self.logger.info(f"Successfully added {total_added} chunks from {document_name}")
            return total_added
        
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise

    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search and return ranked results.
        
        Args:
            query_embedding (List[float]): Query vector
            top_k (int): Number of top results to return
        
        Returns:
            List[Dict[str, Any]]: Ranked search results
        """
        try:
            results = self.index.query(
                vector=query_embedding, 
                top_k=top_k, 
                include_metadata=True
            )
            
            # Transform results into a clean format
            ranked_results = [
                {
                    "text": match.get('metadata', {}).get('text', 'No text'),
                    "document": match.get('metadata', {}).get('document', 'Unknown'),
                    "score": match.get('score', 0.0)
                }
                for match in results.get('matches', [])
            ]
            
            if not ranked_results:
                self.logger.warning("No relevant document chunks found")
            
            return ranked_results
        
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []

    def delete_document(self, document_name: str) -> bool:
        """
        Delete all vectors associated with a specific document.
        
        Args:
            document_name (str): Name of the document to delete
        
        Returns:
            bool: Whether deletion was successful
        """
        try:
            # Fetch document IDs
            results = self.index.query(
                vector=[0] * self.dimension,  # Dummy vector
                filter={"document": document_name},
                top_k=1000  # Large number to fetch all document vectors
            )
            
            # Extract and delete vector IDs
            vector_ids = [match['id'] for match in results.get('matches', [])]
            
            if vector_ids:
                self.index.delete(ids=vector_ids)
                self.logger.info(f"Deleted {len(vector_ids)} vectors for {document_name}")
                return True
            
            self.logger.warning(f"No vectors found for document: {document_name}")
            return False
        
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return False
