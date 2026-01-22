import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

st.set_page_config(page_title="AI Engineering RAG", layout="centered")

st.title("AI Engineering Document Chat")

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

query = st.text_input("Ask a question about the document")

if query:
    result = qa_chain.invoke(query)

    st.subheader("Answer")
    st.write(result["result"])

    with st.expander("Source Documents"):
        for doc in result["source_documents"]:
            st.write(doc.metadata.get("source", "unknown"))