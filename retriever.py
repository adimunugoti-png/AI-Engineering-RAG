from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


if __name__ == "__main__":
    vectorstore = load_vector_store()

    query = "What is AI Engineering?"
    docs = vectorstore.similarity_search(query, k=3)

    print("\n🔎 Retrieved Chunks:\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Result {i} ---")
        print(doc.page_content)
        print()