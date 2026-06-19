import os
import glob
from mcp.server.fastmcp import FastMCP
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the FastMCP Server
mcp = FastMCP("UniversalOfflineRAG")

INDEX_DIR = "faiss_document_index"
REGISTRY_FILE = "indexed_files_registry.txt"

# Lightweight embeddings that run easily on CPU/RAM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@mcp.tool()
def index_documents(folder_path: str) -> str:
    """Reads all documents in a folder and indexes them into the local FAISS database."""
    if not os.path.exists(folder_path):
        return f"Error: Directory {folder_path} does not exist."

    # THE HARDWARE-SAFE CHUNKING STRATEGY
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # Strict diet to prevent VRAM overflow
        chunk_overlap=150,    # Preserves sentence context
        length_function=len
    )

    documents = []
    supported_files = []
    for ext in ["*.pdf", "*.txt", "*.csv", "*.docx"]:
        supported_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not supported_files:
        return "No supported documents found to index."

    for file_path in supported_files:
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_path.endswith(".csv"):
                loader = CSVLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            
            docs = loader.load()
            split_docs = text_splitter.split_documents(docs)
            documents.extend(split_docs)
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

    if not documents:
        return "Failed to extract text from documents."

    # Save to local FAISS index
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local(INDEX_DIR)
    
    with open(REGISTRY_FILE, "w") as f:
        for file_path in supported_files:
            f.write(os.path.basename(file_path) + "\n")

    return f"Successfully indexed {len(documents)} safe-sized chunks from {len(supported_files)} files."

@mcp.tool()
def search_knowledge_base(search_query: str, top_k: int = 3) -> str:
    """Searches the offline FAISS index for relevant context."""
    if not os.path.exists(INDEX_DIR):
        return "Error: No indexed documents found. Please index documents first."

    try:
        vector_db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        results = vector_db.similarity_search(search_query, k=top_k)
        
        if not results:
            return "No relevant context found."
            
        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        return f"Search error: {str(e)}"

if __name__ == "__main__":
    print("=== OFFLINE MCP SERVER STARTED ===")
    print("Listening for SSE connections on port 8000...")
    # Run the server on port 8000
    mcp.run(transport="sse")