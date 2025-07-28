from langchain_community.document_loaders import PyMuPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):

    loader = PyMuPDFLoader(pdf_path)
    
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        
        chunk_size=chunk_size,
        
        chunk_overlap=chunk_overlap
        
    )
    
    chunks = splitter.split_documents(documents)
    
    print(f"Loaded {len(documents)} pages and split into {len(chunks)} chunks.")
    
    return chunks
