# 🧠 Research Paper Q&A Agent

Your personal AI-powered research assistant that finds, reads, and explains research papers for you!

---

## 📌 What is this?

The **Research Paper Q&A Agent** is a full-stack AI tool that:

🔎 Searches and downloads research papers based on user queries using **Arxiv API**  
📄 Extracts and parses PDF content  
📦 Chunks and embeds documents using **LangChain** + **FAISS**  
🤖 Uses **Ollama-powered LLMs** for local, private RAG (Retrieval-Augmented Generation)  
💬 Allows users to ask any question related to the downloaded paper and get factual, citation-grounded answers  
🌐 Has a beautifully designed **Streamlit frontend** with a split layout — download and Q&A are handled separately!

---

## 🚀 Features

- 🔍 **Smart ArXiv Search**: Enter your topic and fetch the top relevant research paper (PDF + metadata).
- 📥 **PDF Download**: Downloaded only if not already present.
- 🧩 **Text Splitting & Embedding**: Powered by `PyMuPDF` and `LangChain`'s `RecursiveCharacterTextSplitter`.
- 🧠 **Local LLM Q&A**: Uses **Ollama** models like `llama3`, `mistral`, `nomic-embed-text`.
- 🧠 **Multi-question Insight**: Ask predefined research-related questions (summary, contributions, methodology, etc.).
- ✍️ **Freeform Questions**: Ask any question based on the paper, with grounded answers.
- 🌈 **Interactive UI**: Modern two-column Streamlit layout separating download and Q&A logic.

---

## 🗂️ Project Structure

research_agent/
│
├── data/

│ └── raw_papers/ # Downloaded PDFs
│
├── embeddings/

│ └── faiss_index/ # FAISS index from chunk embeddings
│
├── utils/

│ ├── fetch_papers.py # Arxiv paper search + metadata fetcher

│ ├── parse_pdf.py # PDF loader and chunker

│ ├── chunk_embed.py # FAISS embedding creation

│ └── rag_agent.py # LangChain RAG pipeline + Q&A logic
│
├── app.py # Streamlit frontend (UI logic)

└── README.md # You're here!


---

## ⚙️ Tech Stack

| Layer         | Tools/Frameworks                         |
|---------------|------------------------------------------|
| 🔗 Backend     | Python, LangChain, FAISS, Ollama          |
| 🧠 LLM         | Ollama (`llama3`, `mistral`, `nomic-embed`) |
| 📄 PDF Parsing | PyMuPDF, LangChain loaders               |
| 🖥️ Frontend     | Streamlit (Dark themed, 2-column layout) |
| 📡 APIs        | ArXiv API (no key required)              |

---

## 💡 How It Works (Workflow)

1. **User enters a query** like `"Graph Transformers with PyTorch"`  
2. App fetches the top matching paper from Arxiv  
3. If PDF already exists → skip download ✅  
4. Else download it to `data/raw_papers/`  
5. If FAISS index exists → reuse it ✅  
6. Else parse → chunk → embed the PDF  
7. Now ask any question — system answers only from that paper's content  
8. Done! 🎯

---


🧪 Supported Ollama Models
llama3

mistral

nomic-embed-text

👉 Make sure to pull them first:
ollama pull llama3
ollama pull mistral
ollama pull nomic-embed-text


🙌 Acknowledgements
Thanks to:

LangChain

Ollama

Streamlit

ArXiv for open-access research


🧔‍♂️ Author
Made with 💻 by Karan Shelar

🌟 Show Your Support
If you like this project, don’t forget to:

⭐ Star it on GitHub
🍴 Fork it
🚀 Share it with others!
