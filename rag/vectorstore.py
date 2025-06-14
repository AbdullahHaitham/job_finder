import json
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings

def load_jobs(filepaths):
    jobs = []
    for path in filepaths:
        with open(path, "r", encoding="utf-8") as f:
            jobs += json.load(f)
    return jobs

def job_to_text(job):
    return " | ".join(str(job.get(k, "")) for k in job)
def get_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_chroma_vectorstore(filepaths, persist_dir="data/chromadb_jobs"):
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
