import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Q&A Chatbot")

PROMPT = PromptTemplate.from_template("""Answer the question using ONLY the context below.
If not in context, say "I couldn't find that in the document."

Context: {context}
Question: {question}
Answer:""")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    if "ready" not in st.session_state:
        st.write("Indexing... please wait")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded.read())
            tmp = f.name

        loader = PyPDFLoader(tmp)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        st.write(f"Created {len(chunks)} chunks, embedding now...")

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.unlink(tmp)

        st.session_state.vectorstore = vectorstore
        st.session_state.ready = True
        st.write("Done! Ask a question below.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": question})

        llm = ChatOllama(model="llama3.2", temperature=0)

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(question)
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})