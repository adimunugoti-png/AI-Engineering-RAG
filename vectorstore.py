from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from loader import load_pdf
from Chunker import chunk_text
import os

if __name__ == "__main__":
    pdf_path = r"C:\Users\vikra\OneDrive\Desktop\AI Engineering.pdf"

    print("Loading PDF...")
    text = load_pdf(pdf_path)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    os.makedirs("faiss_index", exist_ok=True)
    vectorstore.save_local("faiss_index")

    print("✅ Vector store created successfully!")