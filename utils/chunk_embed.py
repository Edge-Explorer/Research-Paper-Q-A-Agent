from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaEmbeddings

import os

def embed_chunks(chunks, index_dir="embeddings/faiss_index", embedding_model_name="nomic-embed-text"):

    embedding_model = OllamaEmbeddings(model=embedding_model_name)

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    if not os.path.exists(index_dir):
        
        os.makedirs(index_dir)

    vectorstore.save_local(index_dir)
    
    print(f"FAISS index saved to: {index_dir}")
    
    return vectorstore.as_retriever()
