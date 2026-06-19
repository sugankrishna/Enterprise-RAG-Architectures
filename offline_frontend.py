import os
import sys
import asyncio
import warnings

# ==========================================
# 1. WINDOWS ASYNCIO BUG PATCH (WinError 10054)
# ==========================================
if os.name == "nt":
    try:
        import asyncio.proactor_events as _pe
        _orig_call_connection_lost = _pe._ProactorBasePipeTransport._call_connection_lost
        
        def _patched_call_connection_lost(self, exc):
            try:
                return _orig_call_connection_lost(self, exc)
            except ConnectionResetError as e:
                if "forcibly closed by the remote host" in str(e):
                    return
                raise
                
        _pe._ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost
    except ImportError:
        pass

# ==========================================
# 2. CORE IMPORTS & CONFIGURATION
# ==========================================
import streamlit as st
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_ollama import ChatOllama

warnings.filterwarnings("ignore")

DOCS_DIR = "./user_docs"
os.makedirs(DOCS_DIR, exist_ok=True)

st.set_page_config(page_title="100% Offline AI", page_icon="🔒", layout="wide")

# Initialize Local AI optimized for 4GB VRAM
model = ChatOllama(
    model="gemma2:2b", 
    temperature=0.1
)

# ==========================================
# 3. NETWORK BRIDGE (FastMCP SSE Protocol)
# ==========================================
async def call_mcp_tool(tool_name, arguments):
    url = "http://localhost:8000/sse"
    try:
        async with sse_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return result.content[0].text
    except Exception as e:
        return f"ERROR_NETWORK: Ensure offline_mcp_server.py is running on port 8000. Details: {e}"

def run_tool_sync(tool_name, arguments):
    return asyncio.run(call_mcp_tool(tool_name, arguments))

# ==========================================
# 4. STATE MANAGEMENT & SIDEBAR
# ==========================================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am running 100% offline via Phi-3. Ask me anything about your documents."}]

with st.sidebar:
    st.header("🔐 Access Control")
    
    if not st.session_state.is_admin:
        st.info("Guest Mode: Chat only.")
        admin_password = st.text_input("Admin Password", type="password")
        
        if st.button("Log In"):
            if admin_password == "admin123": 
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.error("Access Denied")
    else:
        st.success("Admin Access Granted")
        if st.button("Log Out"):
            st.session_state.is_admin = False
            st.rerun()
            
        st.divider()
        st.header("📂 Local Knowledge Management")
        
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            type=["txt", "pdf", "docx", "csv"], 
            accept_multiple_files=True
        )
        
        if st.button("Save & Index Documents"):
            if uploaded_files:
                with st.spinner("Processing offline..."):
                    for f in uploaded_files:
                        file_path = os.path.join(DOCS_DIR, f.name)
                        with open(file_path, "wb") as out:
                            out.write(f.getbuffer())
                    
                    index_result = run_tool_sync("index_documents", {"folder_path": DOCS_DIR})
                    st.success(index_result)
            else:
                st.warning("Please upload a file first.")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Offline conversation cleared."}]
        st.rerun()

# ==========================================
# 5. MAIN CHAT INTERFACE & LOGIC
# ==========================================
st.title("🔒 100% Offline Document Assistant")
st.caption("Powered by Phi-3, FAISS, and MCP. Zero internet connection required.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your files..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("🔍 *Searching local database...*")
        
        # The Goldilocks Search: top_k=3
        context_segments = run_tool_sync("search_knowledge_base", {"search_query": prompt, "top_k": 3})
        
        if "ERROR_NETWORK" in context_segments:
            status_placeholder.empty()
            error_msg = "⚠️ Error communicating with local MCP server. Ensure offline_mcp_server.py is running on port 8000."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            status_placeholder.markdown("🔍 *Found relevant context. Reading documents...*")
            
            final_prompt = f"""You are an offline data analyst. 
            Your goal is to answer the User Request based on the Document Context below.
            If the data is completely missing, say "I cannot find this exact data in the offline documents." But try your best to find relevant numbers matching the request.
            
            Document Context:
            {context_segments}
            
            User Request: {prompt}"""
            
            status_placeholder.empty()
            
            final_answer = ""
            try:
                response_stream = model.stream(final_prompt)
                final_answer = st.write_stream(chunk.content for chunk in response_stream)
                
                if not final_answer or final_answer.strip() == "":
                    final_answer = "⚠️ Hardware Limit Reached: Ensure Phi-3 is running via Ollama."
                    st.error(final_answer)
                    
            except Exception as e:
                final_answer = f"⚠️ Generation Error: {str(e)}"
                st.error(final_answer)
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})