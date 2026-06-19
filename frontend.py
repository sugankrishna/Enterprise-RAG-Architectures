import streamlit as st
import asyncio
import os
import warnings
from mcp import ClientSession
from mcp.client.sse import sse_client
import google.generativeai as genai

warnings.filterwarnings("ignore")

DOCS_DIR = "./user_docs"
os.makedirs(DOCS_DIR, exist_ok=True)

st.set_page_config(page_title="Universal Document AI", page_icon="📄", layout="wide")

# --- INITIALIZE GEMINI CLIENT ---
# REPLACE WITH YOUR ACTUAL KEY
GEMINI_API_KEY = "YOUR API KEY" 
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# --- NETWORK BRIDGE ---
async def call_mcp_tool(tool_name, arguments):
    url = "http://localhost:8000/sse"
    try:
        async with sse_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return result.content[0].text
    except Exception as e:
        return f"ERROR_NETWORK: Ensure mcp_server.py is running on port 8000. Details: {e}"

def run_tool_sync(tool_name, arguments):
    return asyncio.run(call_mcp_tool(tool_name, arguments))

# --- STATE MANAGEMENT ---
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Universal Document Assistant. Ask me anything about the knowledge base."}]

# --- UI: SIDEBAR (RBAC) ---
with st.sidebar:
    st.header("🔐 Access Control")
    
    if not st.session_state.is_admin:
        st.info("Guest Mode: You can chat with the existing knowledge base. Admin login required to upload new documents.")
        admin_password = st.text_input("Admin Password", type="password")
        
        if st.button("Log In"):
            if admin_password == "admin123": 
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.error("Access Denied: Incorrect Password")
    else:
        st.success("Admin Access Granted")
        if st.button("Log Out"):
            st.session_state.is_admin = False
            st.rerun()
            
        st.divider()
        st.header("📂 Knowledge Base Management")
        st.write("Upload Universal Documents.")
        
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            type=["txt", "pdf", "docx", "csv", "xlsx", "pptx"], 
            accept_multiple_files=True
        )
        
        if st.button("Save & Index Documents"):
            if uploaded_files:
                with st.spinner("Processing and vectorizing documents locally..."):
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
        st.session_state.messages = [{"role": "assistant", "content": "Conversation cleared. How can I help?"}]
        st.rerun()

# --- UI: MAIN CHAT INTERFACE ---
st.title("📄 High-Capacity Document Assistant")
st.caption("Powered by Gemini 2.5 Flash, Local FAISS, and MCP.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your files..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent..."):
            
            # Context window for memory
            history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages[-5:]])
            
            # Intent Router
            router_prompt = f"""Analyze the Chat History and the User's Latest Request.
            Does the Latest Request require searching the documents for NEW facts, or is it just a conversational follow-up (like "make it short", "summarize", "explain more", "thanks")?
            
            Chat History:
            {history_text}
            
            Latest Request: {prompt}
            
            If it requires a document search, reply ONLY with the 3 to 5 best search keywords.
            If it is just a conversational follow-up, reply with exactly the word: CHAT"""
            
            router_decision = model.generate_content(router_prompt).text.strip()
            
            if router_decision == "CHAT":
                st.caption("🗨️ Conversational follow-up detected.")
                context_segments = "No new documents needed. Rely on the Chat History provided below."
            else:
                st.caption(f"🔍 Searching database for: '{router_decision}'")
                context_segments = run_tool_sync("search_knowledge_base", {"search_query": router_decision, "top_k": 20})
            
            if "ERROR_NETWORK" in context_segments:
                final_answer = "Could not connect to your local mcp_server.py backend."
                st.error(final_answer)
            else:
                final_prompt = f"""You are an expert Data Analyst and Document Intelligence Assistant. 
                CRITICAL RULE: Answer the User's Latest Request based on the Document Context AND the Chat History.
                Do not make up facts. If the answer is not in the context or history, say so.
                
                Chat History:
                {history_text}
                
                Document Context:
                {context_segments}
                
                Latest Request: {prompt}"""
                
                final_answer = model.generate_content(final_prompt).text
                st.markdown(final_answer)
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
