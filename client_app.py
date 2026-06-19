import asyncio
import re
import os
import torch
import warnings
from mcp import ClientSession
from mcp.client.sse import sse_client
from transformers import AutoTokenizer, pipeline, TextIteratorStreamer
from threading import Thread

# Suppress client-side warnings to keep the terminal clean
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

SYSTEM_PROMPT = """You are a core banking assistant specialized in Temenos systems.
You have access to the following tools to look up banking data:

1. To index a folder containing text or document assets:
   Use the format: [TOOL: IndexBankingDocs, path="./folder_name"]

2. To search the indexed banking knowledge base:
   Use the format: [TOOL: SearchBankingKnowledgeBase, query="your search keywords"]

If you need to look up data to answer a question, you MUST output the tool call format above first.
When you have the final information, provide a direct answer to the user.
"""

async def run_agent():
    # Connect via Localhost HTTP instead of a background process
    url = "http://localhost:8000/sse"
    print(f"\n[Network] Attempting to connect to MCP Server at {url}...")
    
    try:
        async with sse_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                print("[Network] Connection Established Successfully!\n")
                
                print("[System] Loading Qwen 1.5B model into memory...")
                model_id = "Qwen/Qwen2.5-1.5B-Instruct"
                
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                hf_pipeline = pipeline(
                    "text-generation",
                    model=model_id,
                    tokenizer=tokenizer,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    model_kwargs={
                        "device_map": "auto",
                        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
                    }
                )
                
                def ask_qwen(user_message, history=[]):
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    for h in history:
                        messages.append(h)
                    messages.append({"role": "user", "content": user_message})
                    
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    generation_kwargs = dict(text_inputs=prompt, streamer=streamer, max_new_tokens=256, temperature=0.1, do_sample=True)
                    
                    thread = Thread(target=hf_pipeline, kwargs=generation_kwargs)
                    thread.start()
                    
                    generated_text = ""
                    for new_text in streamer:
                        print(new_text, end="", flush=True)
                        generated_text += new_text
                    print() 
                    
                    return generated_text.strip()

                # --- PHASE 1: INGESTION ---
                print("\n=== PHASE 1: INGESTING REPOSITORY ===")
                ingest_query = "Please index the documentation located in ./temenos_docs"
                print(f"User: {ingest_query}\nAgent: ", end="")
                
                response1 = ask_qwen(ingest_query)
                
                if "IndexBankingDocs" in response1:
                    print("\n[Network Action] Sending Indexing request to Server...")
                    tool_res = await session.call_tool("index_banking_repository", {"folder_path": "./temenos_docs"})
                    print(f"[Server Responded]: {tool_res.content[0].text}")
                    
                    print("\nAgent Final: ", end="")
                    ask_qwen(
                        user_message=f"Tool output: {tool_res.content[0].text}. Please wrap up your response.",
                        history=[{"role": "user", "content": ingest_query}, {"role": "assistant", "content": response1}]
                    )
                
                # --- PHASE 2: RETRIEVAL ---
                print("\n=== PHASE 2: BANKING RETRIEVAL QUERY ===")
                user_query = """What is the onboarding policy if a customer risk score is 45? Also check what the minimum opening balance is for a retail checking account.

                SYSTEM REMINDER: You do not know the answer. You MUST look this up using the exact format: [TOOL: SearchBankingKnowledgeBase, query="your keywords"]"""
                print(f"User: {user_query}\nAgent: ", end="")
                
                response2 = ask_qwen(user_query)
                
                if "SearchBankingKnowledgeBase" in response2:
                    match = re.search(r'query=["\'](.*?)["\']', response2)
                    search_keywords = match.group(1) if match else user_query
                    
                    print(f"\n[Network Action] Sending Search query for: '{search_keywords}'...")
                    tool_res = await session.call_tool("query_banking_knowledge", {"search_query": search_keywords})
                    context = tool_res.content[0].text
                    
                    print("\nAgent Final Answer: ", end="")
                    ask_qwen(
                        user_message=f"Based on the following retrieved documentation context, answer the original user query:\n\n{context}",
                        history=[{"role": "user", "content": user_query}, {"role": "assistant", "content": response2}]
                    )
                    
    except Exception as e:
        print(f"\n[Fatal Error] Could not connect to the network server. Did you start mcp_server.py in a separate terminal?")
        print(f"Error Details: {e}")

if __name__ == "__main__":
    asyncio.run(run_agent())