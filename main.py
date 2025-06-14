from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from rag.vectorstore import build_chroma_vectorstore
from rag.retriever import get_retriever
from rag.generator import get_rag_chain

DATA_FILES = [
    "./data/processed_jobs_third_column_fixed.json",
    "./data/wuzzuf_jobs_since_time.json"
]
VSTORE_DIR = "./chromadb_jobs"

vectordb = build_chroma_vectorstore(DATA_FILES, VSTORE_DIR)
retriever = get_retriever(vectordb)
rag_chain = get_rag_chain(retriever)

app = FastAPI(title="Job Finder RAG API with Llama 3.2")

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@app.post("/find_jobs")
def find_jobs(request: QueryRequest):
    retriever.search_kwargs["k"] = request.top_k
    response = rag_chain(request.query)
    return {
        "query": request.query,
        "llm_response": response["result"],
        "jobs_used": [d.metadata for d in response["source_documents"]]
    }
