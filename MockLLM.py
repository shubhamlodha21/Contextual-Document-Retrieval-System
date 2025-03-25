# mock_llm.py
import logging
from typing import Dict, Any

class MockLLM:
    """
    A flexible mock Large Language Model (LLM) for generating context-aware responses.
    
    Key Features:
    - Configurable response generation
    - Context-based text generation
    - Comprehensive logging
    """
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize the MockLLM with optional model specification.
        
        Args:
            model_name (str): Name or identifier of the mock model
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name

    def generate_response(
        self, 
        query: str, 
        context: Dict[str, Any], 
        max_tokens: int = 300
    ) -> Dict[str, str]:
        """
        Generate a context-aware response using a simplified LLM strategy.
        
        Args:
            query (str): User's input query
            context (Dict[str, Any]): Contextual information from vector search
            max_tokens (int): Maximum token limit for response
        
        Returns:
            Dict[str, str]: Generated response with metadata
        """
        try:
            # Validate inputs
            if not query or not context:
                return {"response": "Insufficient context or query."}
            
            # Context-based response strategy
            context_text = context.get('text', 'No context available')
            relevance_score = context.get('score', 0.0)
            
            # Simple response generation based on context
            if relevance_score > 0.5:
                response = (
                    f"Based on the context from '{context.get('document', 'Unknown Document')}', "
                    f"here's a focused response to: {query}\n\n"
                    f"Context: {context_text[:300]}...\n\n"
                    f"Response: [Simulated context-aware response]"
                )
            else:
                response = (
                    "The provided context doesn't seem sufficiently relevant. "
                    "Consider refining your query or providing more specific information."
                )
            
            # Log generation details
            self.logger.info(
                f"Response generated | "
                f"Query Length: {len(query)} | "
                f"Context Relevance: {relevance_score:.2f}"
            )
            
            return {
                "response": response,
                "model": self.model_name,
                "context_relevance": relevance_score
            }
        
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            return {"response": f"Error generating response: {str(e)}"}
