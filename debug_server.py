import sys
print("--- STARTING DIRECT SERVER DIAGNOSTIC ---")
print("Bypassing MCP Client to force logs to terminal...\n")

# Import your server file directly (this will trigger the os.environ patches)
try:
    import mcp_server
    print("[Debug] Successfully imported mcp_server.py")
except Exception as e:
    print(f"[Debug] CRASH DURING IMPORT: {e}")
    sys.exit(1)

try:
    print("\n>> Diagnostic Step 1: Forcing Embedding Model Load...")
    mcp_server.get_embeddings()
    print(">> Step 1 Passed! Hardware allocation successful.")
    
    print("\n>> Diagnostic Step 2: Forcing FAISS Indexing...")
    # Make sure your temenos_docs folder exists!
    result = mcp_server.index_banking_repository("./temenos_docs")
    print(f">> Step 2 Passed! Output: {result}")
    
    print("\n--- DIAGNOSTIC COMPLETE: NO FREEZES DETECTED ---")

except Exception as e:
    print(f"\n!!! FATAL CRASH CAUGHT: {e}")