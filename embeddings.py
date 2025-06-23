from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from preprocess import process_documents
import pickle
import os

def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for text chunks."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

def build_vectorstore_index(embeddings, chunks, index_path="faiss_index"):
    """Build and save FAISS index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, f"{index_path}/index.faiss")
    # Save chunks for retrieval
    with open(f"{index_path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    return index

def test_retrieval(query, index_path="faiss_index", model_name="all-MiniLM-L6-v2", k=3):
    """Test retrieving top-k chunks for a query."""
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    index = faiss.read_index(f"{index_path}/index.faiss")
    with open(f"{index_path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

def main():
    # Create directory if it doesn't exist
    os.makedirs("faiss_index", exist_ok=True)
    
    # Load chunks
    chunks = process_documents()
    if not chunks:
        print("No chunks to process.")
        return
    
    # Create embeddings
    embeddings = create_embeddings(chunks)
    
    # Build FAISS index
    build_vectorstore_index(embeddings, chunks)
    
    # Test retrieval
    test_query = "What was Pakistan's trade balance in 2023?"
    retrieved_chunks = test_retrieval(test_query)
    print(f"\nTest Query: {test_query}")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"Retrieved Chunk {i+1}: {chunk[:100]}...")

if __name__ == "__main__":
    main()