import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")

import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def process_documents(directory="documents"):
    """Process all PDFs in the directory and return text chunks."""
    all_chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":
    # Test the preprocessing
    chunks = process_documents()
    print(f"Extracted {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks
        print(f"Chunk {i+1}: {chunk[:100]}...")