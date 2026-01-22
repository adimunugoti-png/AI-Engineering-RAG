from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(model="gemma:2b")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    while True:
        query = input("\nAsk a question (type 'exit' to stop): ")

        if query.lower() == "exit":
            break

        result = qa_chain.invoke(query)

        print("\nAnswer:")
        print(result["result"])

        print("\nSource documents:")
        for doc in result["source_documents"]:
            print(doc.metadata.get("source", "unknown"))

if __name__ == "__main__":
    main()