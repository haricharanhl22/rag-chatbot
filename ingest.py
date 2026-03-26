from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import sys

def ingest(pdf_path):
    print(f"Loading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"  Loaded {len(pages)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)
    print(f"  Created {len(chunks)} chunks")

    print("  Creating embeddings (free, runs locally)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("faiss_index")
    print("  Saved vector store to faiss_index/")
    print("Done! Now run: streamlit run app.py")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py yourfile.pdf")
        sys.exit(1)
    ingest(sys.argv[1])