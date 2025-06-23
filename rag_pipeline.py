from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from transformers import pipeline

def load_vector_index(index_path="faiss_index"):
    """Load FAISS index and chunks."""
    index = faiss.read_index(f"{index_path}/index.faiss")
    with open(f"{index_path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve_chunks(query, index, chunks, model, k=3):
    """Retrieve top-k relevant chunks for the query."""
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query, context, model_name="distilgpt2"):
    """Generate answer using a text generation model."""
    prompt = f"Context: {' '.join(context)}\n\nQuestion: {query}\n\nAnswer:"
    
    generator = pipeline("text-generation", model=model_name, max_length=200, do_sample=False)
    result = generator(prompt)[0]["generated_text"]
    
    return result.split("Answer:")[-1].strip()

def rag_query(query, index_path="faiss_index"):
    """Run the full RAG pipeline."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_vector_index(index_path)
    context = retrieve_chunks(query, index, chunks, model)
    answer = generate_answer(query, context)
    return answer

if __name__ == "__main__":
    query = "What was Pakistan's trade balance in 2023?"
    answer = rag_query(query)
    print(f"Query: {query}\nAnswer: {answer}")