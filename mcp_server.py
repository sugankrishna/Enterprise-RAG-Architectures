import os
import sys
import warnings
import logging
import pandas as pd
import faiss

# --- AGGRESSIVE MCP STDIO & THREADING PATCHES ---
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
faiss.omp_set_num_threads(1)
# --------------------------------------

from mcp.server.fastmcp import FastMCP
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader, 
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- SERVER INITIALIZATION ---
mcp = FastMCP("UniversalDocumentStorage")
DB_PATH = "./faiss_document_index"
_embeddings_cache = None

def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        print("[Server] Loading Embedding Model on CPU...")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    return _embeddings_cache

# --- TOOL 1: UNIVERSAL INDEXER ---
# Add this near your DB_PATH variable at the top
TRACKER_PATH = "./indexed_files_registry.txt"

@mcp.tool()
def index_documents(folder_path: str) -> str:
    """Universal Loader with Incremental Indexing."""
    print(f"\n[Server] Incremental Indexing triggered for: {folder_path}")
    if not os.path.exists(folder_path):
        return f"Error: Path {folder_path} does not exist."
    
    # 1. Load the registry of already processed files
    processed_files = set()
    if os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH, "r") as f:
            processed_files = set(f.read().splitlines())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        length_function=len
    )
    
    raw_docs = []
    newly_processed = [] # Keep track of what we successfully read this time
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 2. SKIP files we have already indexed!
            if file in processed_files:
                print(f"[Server] Skipping already indexed file: {file}")
                continue
                
            file_path = os.path.join(root, file)
            ext = file.lower()
            
            try:
                if ext.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    raw_docs.extend(loader.load())
                elif ext.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                    raw_docs.extend(loader.load())
                elif ext.endswith('.csv'):
                    loader = CSVLoader(file_path)
                    raw_docs.extend(loader.load())
                elif ext.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                    text_data = df.to_csv(index=False) 
                    raw_docs.append(Document(page_content=text_data, metadata={"source": file}))
                elif ext.endswith('.pptx'):
                    loader = UnstructuredPowerPointLoader(file_path)
                    raw_docs.extend(loader.load())
                elif ext.endswith(('.txt', '.md')):
                    loader = TextLoader(file_path, encoding='utf-8')
                    raw_docs.extend(loader.load())
                
                print(f"[Server] Successfully parsed NEW file: {file}")
                newly_processed.append(file) # Add to our success list
                
            except Exception as e:
                print(f"[Server] Failed to parse {file}: {e}")
    
    # 3. Append to Database (Only if we found new documents)
    if raw_docs:
        final_chunks = text_splitter.split_documents(raw_docs)
        
        if os.path.exists(DB_PATH):
            # If DB exists, APPEND to it
            print("[Server] Appending new chunks to existing FAISS database...")
            db = FAISS.load_local(DB_PATH, get_embeddings(), allow_dangerous_deserialization=True)
            db.add_documents(final_chunks)
        else:
            # If no DB exists, CREATE it
            print("[Server] Creating new FAISS database...")
            db = FAISS.from_documents(final_chunks, get_embeddings())
            
        db.save_local(DB_PATH)
        
        # 4. Save the new files to our registry so they are skipped next time
        with open(TRACKER_PATH, "a") as f:
            for nf in newly_processed:
                f.write(nf + "\n")
                
        return f"Incremental Indexing Complete: Added {len(newly_processed)} NEW files into {len(final_chunks)} chunks."
    
    return "No new documents to index. Everything is up to date."

# --- TOOL 2: SEARCH ENGINE ---
@mcp.tool()
def search_knowledge_base(search_query: str, top_k: int = 3) -> str:
    """Queries the local FAISS database for information from uploaded documents."""
    print(f"\n[Server] Search Query Received: '{search_query}'")
    if not os.path.exists(DB_PATH):
        return "The knowledge base index is empty. Run the indexer first."
    
    db = FAISS.load_local(DB_PATH, get_embeddings(), allow_dangerous_deserialization=True)
    results = db.similarity_search(search_query, k=top_k)
    
    formatted_results = []
    for doc in results:
        formatted_results.append(f"Source Document: {doc.metadata['source']}\nExtract:\n{doc.page_content}\n---")
    
    print(f"[Server] Search complete! Returning {len(results)} matches.")
    return "\n".join(formatted_results)

if __name__ == "__main__":
    print("\n=== UNIVERSAL MCP SERVER STARTED ===")
    print("Listening for HTTP network requests on port 8000...")
    mcp.run(transport="sse")