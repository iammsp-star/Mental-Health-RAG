# 🧠 Mental Health Resource Finder — RAG App

An AI-powered **Retrieval-Augmented Generation (RAG)** application that helps users discover mental health resources, coping strategies, and professional guidance from curated PDF documents.

Built with **Streamlit**, **LangChain**, and **ChromaDB** for fast, context-aware answers.

---

## ✨ Features

- 📄 **PDF Ingestion** — Upload and index any mental health resource PDF
- 🔍 **Semantic Search** — Find the most relevant passages using vector similarity
- 🤖 **AI-Powered Answers** — Get structured responses with summaries, key resources, and disclaimers
- 💾 **Persistent Vector Store** — ChromaDB stores embeddings locally for fast subsequent queries
- 🔐 **Configurable API Key** — Bring your own OpenAI / OpenRouter API key

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Streamlit** | Interactive web UI |
| **LangChain** | RAG pipeline orchestration |
| **ChromaDB** | Vector store for document embeddings |
| **HuggingFace** | Sentence embeddings (`all-MiniLM-L6-v2`) |
| **OpenRouter** | LLM API gateway |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- An [OpenRouter](https://openrouter.ai/) API key (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/iammsp-star/Mental-Health-RAG.git
cd Mental-Health-RAG

# Install dependencies
pip install streamlit langchain langchain-openai langchain-community langchain-huggingface chromadb pypdf

# Run the app
streamlit run main.py
```

### Usage

1. Launch the app with `streamlit run main.py`
2. Enter your **OpenRouter API key** in the sidebar
3. Provide the path to your **mental health PDF resource**
4. Ask questions like:
   - *"What are some coping strategies for stress?"*
   - *"How can I support someone with anxiety?"*
   - *"What are signs of burnout?"*

---

## 📁 Project Structure

```
Mental-Health-RAG/
├── main.py          # Streamlit app + RAG pipeline
├── sampledb/        # ChromaDB persistent storage (auto-generated)
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚠️ Disclaimer

> This app is an **educational tool** and does not replace professional mental health support. If you or someone you know is in crisis, please contact a licensed mental health professional or your local emergency services.

---

## 📬 Connect

<a href="https://github.com/iammsp-star" target="_blank">
  <img src="https://img.shields.io/badge/GitHub-iammsp--star-181717?style=for-the-badge&logo=github" alt="GitHub"/>
</a>
<a href="https://www.linkedin.com/in/manas-puthanpura-5b06b0377/" target="_blank">
  <img src="https://img.shields.io/badge/LinkedIn-Manas%20Puthanpura-0A66C2?style=for-the-badge&logo=linkedin" alt="LinkedIn"/>
</a>

---

<p align="center">Made with ❤️ by <a href="https://github.com/iammsp-star">Manas</a></p>