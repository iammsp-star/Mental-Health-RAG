# -*- coding: utf-8 -*-
"""Mental Health Resource Finder — RAG App"""

import os
import tempfile
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Mental Health Resource Finder",
    page_icon="🧠",
    layout="wide",
)

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── App Header ──────────────────────────────────────────────
st.title("🧠 Mental Health Resource Finder")
st.markdown("*AI-powered answers from your mental health PDF resources*")
st.divider()

# ── Sidebar ─────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

# API Key: try Streamlit secrets first, then sidebar input
default_key = ""
if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
    default_key = st.secrets["OPENAI_API_KEY"]

api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    value=default_key,
    type="password",
    help="Get a free key at https://openrouter.ai/",
)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# PDF Upload
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a mental health resource PDF",
    type=["pdf"],
    help="Upload a PDF document to search through",
)


# ── Load & Index PDF ────────────────────────────────────────
@st.cache_resource
def load_and_index_pdf(_file_bytes, file_name):
    """Save uploaded PDF to a temp file, load, chunk, and index it."""
    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_file_bytes)
        tmp_path = tmp.name

    # Load the PDF
    loader = PyPDFLoader(file_path=tmp_path)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # In-memory Chroma (no persist needed for cloud)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name="mental_health",
    )

    # Clean up temp file
    os.unlink(tmp_path)

    return vectorstore, len(docs), len(chunks)


# ── Main App Logic ──────────────────────────────────────────
if not api_key:
    st.warning("👈 Please enter your **OpenRouter API Key** in the sidebar to get started.")
    st.stop()

if not uploaded_file:
    st.info("👈 Upload a **mental health PDF** in the sidebar to begin.")
    st.stop()

# Index the PDF
with st.spinner("📚 Indexing your PDF..."):
    file_bytes = uploaded_file.getvalue()
    vectorstore, num_pages, num_chunks = load_and_index_pdf(file_bytes, uploaded_file.name)

st.success(f"✅ Indexed **{num_pages} pages** into **{num_chunks} chunks** — ready to answer!")

# Setup LLM
llm = ChatOpenAI(
    model="arcee-ai/trinity-large-preview:free",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

# Prompt
prompt_template = """
You are a helpful assistant for mental health resources. Use the following context to answer the question structured as:
- **Summary**: Brief overview.
- **Key Resources**: Bullet list.
- **Disclaimer**: Always seek professional help.

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(prompt_template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": vectorstore.as_retriever(search_kwargs={"k": 3}) | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ── Chat Interface ──────────────────────────────────────────
st.subheader("💬 Ask a Question")

query = st.text_input(
    "What would you like to know about mental health?",
    placeholder="e.g. What are some coping strategies for stress?",
)

if st.button("🔍 Get Answer", type="primary", use_container_width=True):
    if query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)

        st.subheader("📝 Answer")
        st.markdown(response)

        with st.expander("📄 Source Passages"):
            search_results = vectorstore.similarity_search(query, k=3)
            for i, res in enumerate(search_results, 1):
                st.markdown(f"**Passage {i}:**")
                st.caption(res.page_content)
                st.divider()
    else:
        st.warning("Please enter a question.")

# ── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This app is an educational tool and does not replace "
    "professional mental health support. If you or someone you know is in crisis, "
    "please contact a licensed professional or your local emergency services."
)