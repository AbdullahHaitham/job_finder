import os

# Directory structure
folders = [
    "job_finder",
    "job_finder/data",
    "job_finder/rag",
    "job_finder/api",
]

# Files with initial content (where appropriate)
files_content = {
    "job_finder/requirements.txt": """fastapi
uvicorn
langchain
chromadb
sentence-transformers
ollama
""",

    "job_finder/README.md": """# Job Finder RAG API

See api/main.py and rag/ for code structure and instructions.
""",

    "job_finder/rag/__init__.py": "",
    "job_finder/rag/embedding.py": """from langchain.embeddings import HuggingFaceEmbeddings

def get_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
""",

    "job_finder/rag/vectorstore.py": """import json
from langchain.vectorstores import Chroma
from .embedding import get_embedder

def load_jobs(filepaths):
    jobs = []
    for path in filepaths:
        with open(path, "r", encoding="utf-8") as f:
            jobs += json.load(f)
    return jobs

def job_to_text(job):
    return " | ".join(str(job.get(k, "")) for k in job)

def build_chroma_vectorstore(filepaths, persist_dir="./chromadb_jobs"):
    jobs = load_jobs(filepaths)
    texts = [job_to_text(job) for job in jobs]
    embedder = get_embedder()
    metadatas = jobs
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas,
        persist_directory=persist_dir
    )
    return vectordb
""",

    "job_finder/rag/retriever.py": """from langchain.vectorstores import Chroma

def get_retriever(vectordb, k=5):
    return vectordb.as_retriever(search_kwargs={"k": k})
""",

    "job_finder/rag/generator.py": """from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def get_llm():
    # Make sure Ollama server is running and llama3 is pulled
    return Ollama(model="llama3", base_url="http://localhost:11434")  # Update if needed

def get_rag_chain(retriever):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a professional job assistant.\\n"
            "User query: {question}\\n"
            "Relevant jobs:\\n{context}\\n"
            "Based on the context, recommend the most relevant jobs, showing title, company (if any), location, salary, and a short reason."
        )
    )
    llm = get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
""",

    "job_finder/api/__init__.py": "",
    "job_finder/api/main.py": """from fastapi import FastAPI
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
"""
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files with content
for filepath, content in files_content.items():
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

print("Project structure created successfully!")
