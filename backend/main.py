import json
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

# SETUP
load_dotenv()

# Groq client using GROQ_API_KEY from environment
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# FastAPI app instance
app = FastAPI()

# CORS middleware (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD KB
current_dir = os.path.dirname(__file__)
kb_path = os.path.join(current_dir, "knowledge_base.json")

try:
    with open(kb_path, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
except FileNotFoundError:
    print("Error: knowledge_base.json not found")
    sys.exit(1)

# RAG INDEXING (runs at startup)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("nova_kb")

skip_sections = {"company", "tone_guidelines", "escalation_rules"}
indexed_count = 0

for section_name, text in knowledge_base.items():
    if section_name in skip_sections:
        continue
    
    embedding = embedding_model.encode(text).tolist()
    
    collection.add(
        ids=[section_name],
        embeddings=[embedding],
        documents=[text]
    )
    indexed_count += 1

print(f"RAG index built: {indexed_count} sections indexed")

# MODELS
class Message(BaseModel):
    role: str
    content: str
    
class TicketRequest(BaseModel):
    message: str
    customer_name: str
    history: list = []

SYSTEM_PROMPT = """You are Nova, NovaTech AI support agent. 
Answer ONLY from the provided knowledge base. 
NEVER make up pricing, policies, or features, strong emphasis on NEVER.
Always address the customer by their first name. 
End every response with a clear next step. 
During legal issues or amounts over £500 -> pass to specialist (human)."""

def get_relevant_context(message: str, kb: dict) -> str:
    base_context = kb["company"] + "\n" + kb["tone_guidelines"] + "\n" + kb["escalation_rules"] + "\n\n"
    
    query_embedding = embedding_model.encode(message).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2,
        include=["documents", "distances"]
    )
    
    valid_docs = []
    retrieved_ids = []
    
    if "distances" in results and results["distances"]:
        for i, dist in enumerate(results["distances"][0]):
            if dist < 1.5:  # Distance threshold
                valid_docs.append(results["documents"][0][i])
                if "ids" in results:
                    retrieved_ids.append(results["ids"][0][i])
    else:
        valid_docs = results["documents"][0]
        if "ids" in results:
            retrieved_ids = results["ids"][0]
            
    print(f"Retrieved section ids passing threshold: {retrieved_ids}")
    
    if not valid_docs:
        retrieved_docs = "[SYSTEM NOTE: No relevant articles found in KB for this query.]"
    else:
        retrieved_docs = "\n".join(valid_docs)
        
    return base_context + retrieved_docs

@app.get("/")
def health_check():
    return {"status": "active", "message": "NovaTech AI Support Agent is running!"}

@app.post("/ask")
def ask_nova(request: TicketRequest):
    try:
        # 1. Edge Case: Empty or whitespace-only messages
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
            
        # 2. Edge Case: Excessively long messages (Token Exhaustion prevention)
        if len(request.message) > 1000:
            raise HTTPException(status_code=400, detail="Message exceeds the 1000 character limit.")

        context = get_relevant_context(request.message, knowledge_base)
        
        # 3. Edge Case: History Truncation (Keep only last 10 turns to save tokens)
        recent_history = request.history[-10:] if request.history else []
        
        formatted_history_parts = []
        for h in recent_history:
            if isinstance(h, dict):
                if "role" in h and "content" in h:
                    role_str = "Customer" if h["role"] in ("user", "customer") else "Nova"
                    formatted_history_parts.append(f"{role_str}: {h['content']}")
                else:
                    msg = h.get("msg", h.get("message", ""))
                    resp = h.get("response", "")
                    formatted_history_parts.append(f"Customer: {msg}  /  Nova: {resp}")
            else:
                formatted_history_parts.append(str(h))
                
        formatted_history = "\n".join(formatted_history_parts)
        
        prompt = f"""{SYSTEM_PROMPT}

Knowledge Base:
{context}

Customer name: {request.customer_name}

Conversation history:
{formatted_history}

Customer: {request.message}
Nova:"""

        result = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "response": result.choices[0].message.content,
            "customer": request.customer_name,
            "status": "resolved"
        }
        
    except Exception as e:
        import traceback
        print("\n--- ERROR DETAILS ---")
        traceback.print_exc()
        print("---------------------\n")
        
        error_msg = str(e).lower()
        if "429" in error_msg:
            raise HTTPException(status_code=429, detail="Too Many Requests")
        elif "403" in error_msg:
            raise HTTPException(status_code=401, detail="Unauthorized")
        elif "404" in error_msg:
            raise HTTPException(status_code=404, detail="Not Found")
        else:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def get_health():
    return {
        "status": "ok",
        "agent": "Nova",
        "rag_enabled": True,
        "kb_sections": list(knowledge_base.keys()),
        "indexed_sections": collection.get()["ids"]
    }
