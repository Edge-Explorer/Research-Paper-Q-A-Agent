import os

import requests

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores import FAISS

import arxiv

def fetch_paper_from_arxiv(query, max_results=3):
    
    search = arxiv.Search(
        
        query=query,
        
        max_results=max_results,
        
        sort_by=arxiv.SortCriterion.Relevance
        
    )

    results = []
    
    for result in search.results():
        
        paper_info = {
            
            "title": result.title,
            
            "authors": [a.name for a in result.authors],
            
            "summary": result.summary,
            
            "url": result.entry_id,
            
            "pdf_url": result.pdf_url
            
        }
        
        results.append(paper_info)
    
    return results


def download_arxiv_pdf(pdf_url: str, save_dir="data/raw_papers/"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    filename = pdf_url.split("/")[-1] + ".pdf"
    
    save_path = os.path.join(save_dir, filename)

    response = requests.get(pdf_url)
    
    with open(save_path, "wb") as f:
        
        f.write(response.content)
    
    print(f"PDF downloaded: {save_path}")
    
    return save_path


def process_pdf_to_embeddings(pdf_path: str, embed_save_path="embeddings/faiss_index"):
    
    loader = PyPDFLoader(pdf_path)
    
    pages = loader.load()
    
    print(f"Loaded {len(pages)} pages.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    chunks = splitter.split_documents(pages)
    
    print(f"Split into {len(chunks)} chunks.")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(embed_save_path, exist_ok=True)
    
    vectorstore.save_local(embed_save_path)
    
    print(f"Embeddings saved to: {embed_save_path}")
