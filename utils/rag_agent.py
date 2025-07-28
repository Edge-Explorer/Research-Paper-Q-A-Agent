import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

# -------------------------------
# Load FAISS vectorstore with Ollama embeddings
# -------------------------------
def load_vectorstore(index_dir="embeddings/faiss_index", embedding_model_name="nomic-embed-text"):
    embedding_model = OllamaEmbeddings(model=embedding_model_name)
    return FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)

# -------------------------------
# Initialize QA chain with LLM + retriever
# -------------------------------
def init_qa_chain(vectorstore, model_name="llama3"):
    llm = OllamaLLM(model=model_name)
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False  # Set to True if you want citations
    )

# -------------------------------
# Ask a single question
# -------------------------------
def answer_query(query, qa_chain):
    return qa_chain.invoke(query)

# -------------------------------
# Ask multiple predefined questions about a paper
# -------------------------------
def ask_multiple_questions(qa_chain):
    questions = [
        "Summarize the abstract of this paper.",
        "What is the main contribution of this paper?",
        "What methodology is proposed in this research?",
        "What are the key findings or results?",
        "What are the conclusions or future work suggestions?"
    ]

    print("📄 Answering multiple research questions:")
    for q in questions:
        print(f"\n➡️ {q}")
        answer = answer_query(q, qa_chain)
        print(f"🧠 {answer}")

        # Optional: save to file (uncomment below if needed)
        # with open("reports/summaries/answers.txt", "a", encoding="utf-8") as f:
        #     f.write(f"\n\nQuestion: {q}\nAnswer: {answer}\n{'-'*40}")





