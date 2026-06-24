from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from src.vector_store import SemanticSearchIndex

app = FastAPI(title="Semantic Search Engine API", version="1.0")
search_engine = SemanticSearchIndex()

# Bootstrap data index on startup
@app.on_event("startup")
def load_and_index_data():
    data_path = "data/knowledge_base.json"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            docs = json.load(f)
        search_engine.initialize_index(docs)
    else:
        # Fallback dummy initialization
        fallback = [{"id": 0, "title": "System", "text": "Default internal server configuration details."}]
        search_engine.initialize_index(fallback)

class QueryPayload(BaseModel):
    query: str
    limit: int = 1

@app.post("/search")
def run_semantic_search(payload: QueryPayload):
    try:
        matches = search_engine.search(query=payload.query, top_k=payload.limit)
        return {"query": payload.query, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))