from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_pdf

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    pdf_path = r"C:\Users\vikra\OneDrive\Desktop\AI Engineering.pdf"

    text = load_pdf(pdf_path)
    chunks = chunk_text(text)

    print(f"Total chunks created: {len(chunks)}")
    print("\n--- First Chunk ---\n")
    print(chunks[0])