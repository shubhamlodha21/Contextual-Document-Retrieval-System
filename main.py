# main.py
import os
from DocumentProcessor import DocumentProcessor
from VectorStore import VectorStore
from MockLLM import MockLLM

def main():
    # Configuration
    PINECONE_API_KEY = "pcsk_5teY1M_22uUQnc4qsChRrprWoir238K1hpdKPprcV5eAPVe8qbMTsLr5Zc9NkctZ6aifCv"
    INDEX_NAME = "document-retrieval2"
    DOCUMENT_PATH = "data/sample.pdf"  # Adjust as needed

    try:
        # Initialize components
        doc_processor = DocumentProcessor()
        vector_store = VectorStore(PINECONE_API_KEY, INDEX_NAME)
        mock_llm = MockLLM()

        # Load and process document
        doc_processor.load_document(DOCUMENT_PATH)
        chunks = doc_processor.chunk_text()
        embeddings = doc_processor.generate_embeddings(chunks)

        # Add to vector store
        vector_store.add_documents(embeddings, chunks, os.path.basename(DOCUMENT_PATH))

        # Interactive query loop
        while True:
            query = input("\nEnter your query (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit']:
                break
            
            # Process query
            query_embedding = doc_processor.process_query(query)
            
            # Search relevant context
            search_results = vector_store.search(query_embedding, top_k=1)
            
            if search_results:
                # Generate response
                response = mock_llm.generate_response(query, search_results[0])
                
                print("\n--- Search Context ---")
                print(f"Document: {search_results[0].get('document', 'N/A')}")
                print(f"Relevance Score: {search_results[0].get('score', 0.0):.2f}")
                
                print("\n--- Generated Response ---")
                print(response['response'])
            else:
                print("No relevant context found.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
