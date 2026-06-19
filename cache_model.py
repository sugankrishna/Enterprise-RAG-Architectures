import os
# This prevents FAISS and PyTorch from freezing on Windows CPU threads
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.embeddings import HuggingFaceEmbeddings

print("Starting safe download of the embedding model...")
# This forces the download to happen safely
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Download complete! The model is safely cached on your hard drive.")