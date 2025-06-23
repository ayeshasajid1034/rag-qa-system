import streamlit as st
import os
from preprocess import process_documents
from embeddings import create_embeddings, build_vectorstore_index
from rag_pipeline import rag_query
import shutil

st.title("RAG-Based Q&A System")
st.write("Upload a PDF document and ask questions about its content.")

# Create temporary directory for uploaded files
if not os.path.exists("temp_documents"):
    os.makedirs("temp_documents")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Save uploaded file
    file_path = os.path.join("temp_documents", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process document and build index
    with st.spinner("Processing document..."):
        chunks = process_documents("temp_documents")
        if chunks:
            embeddings = create_embeddings(chunks)
            # Ensure temp_faiss_index directory exists
            os.makedirs("temp_faiss_index", exist_ok=True)
            build_vectorstore_index(embeddings, chunks, index_path="temp_faiss_index")
            st.success("Document processed successfully!")
        else:
            st.error("Failed to process document.")
    
    # Clean up temp documents
    shutil.rmtree("temp_documents")
    os.makedirs("temp_documents")

# Query input
query = st.text_input("Ask a question about the document:")
if query and os.path.exists("temp_faiss_index/index.faiss"):
    with st.spinner("Generating answer..."):
        answer = rag_query(query, index_path="temp_faiss_index")
        st.write("**Answer**:")
        st.write(answer)

# Clean up index on app restart
if st.button("Reset"):
    if os.path.exists("temp_faiss_index"):
        shutil.rmtree("temp_faiss_index")
    st.write("Index reset. Upload a new document.")