# -*- coding: utf-8 -*-
"""AbdulRAG_Fixed_with_Streamlit.ipynb"""

import os
import streamlit as st

# Set your OpenAI API key if not already set
os.environ["OPENAI_API_KEY"] = ""  # Replace with your actual key

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Streamlit app title
st.title("Mental Health Resource Finder RAG App")

# Sidebar for configuration
st.sidebar.header("Configuration")
pdf_path = st.sidebar.text_input("PDF File Path", value="D:\RAG.pdf")
api_key = st.sidebar.text_input("OpenAI API Key (for OpenRouter)", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Function to load and index the PDF
@st.cache_resource
def load_and_index_pdf(file_path):
    if not os.path.exists(file_path):
        st.error(f"PDF file not found at {file_path}")
        return None

    # Load the PDF
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()
    st.write(f"Loaded {len(docs)} pages from PDF.")

    # Split into smaller chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased from 100 for more context; adjust as needed
        chunk_overlap=200  # Overlap for continuity
    )
    result = splitter.split_documents(docs)
    st.write(f"Split into {len(result)} chunks.")

    # Initialize embeddings
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Create or load Chroma vector store
    persist_dir = 'sampledb'
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory=persist_dir,
        collection_name='Manas'
    )

    # Check if collection is empty; if so, add documents
    if vectorstore._collection.count() == 0:
        vectorstore.add_documents(result)
        st.write("Vector store indexed successfully.")
    else:
        st.write("Loaded existing vector store.")

    return vectorstore

# Load vectorstore
vectorstore = load_and_index_pdf(pdf_path)

if vectorstore:
    # Setup LLM
    llm = ChatOpenAI(
        model="arcee-ai/trinity-large-preview:free",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )  # Use your preferred model

    # Define prompt for structured answers
    prompt_template = """
    You are a helpful assistant for mental health resources. Use the following context to answer the question structured as:
    - **Summary**: Brief overview.
    - **Key Resources**: Bullet list.
    - **Disclaimer**: Always seek professional help.

    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # RAG chain: Retrieve -> Format context -> Prompt -> LLM -> Parse output
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": vectorstore.as_retriever(search_kwargs={"k": 3}) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # User input for query
    query = st.text_input("Enter your question about mental health resources:", value="What are some coping strategies for stress?")

    if st.button("Get Response"):
        if query:
            with st.spinner("Generating response..."):
                response = rag_chain.invoke(query)
                st.subheader("RAG Response:")
                st.write(response)

                # Optional: Similarity search example
                st.subheader("Similarity Search Results:")
                search_results = vectorstore.similarity_search(query, k=1)
                for res in search_results:
                    st.write(res.page_content)
        else:
            st.warning("Please enter a query.")
else:
    st.warning("Please provide a valid PDF path and API key to proceed.")