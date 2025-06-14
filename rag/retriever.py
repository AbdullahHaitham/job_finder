from langchain.vectorstores import Chroma

def get_retriever(vectordb, k=5):
    return vectordb.as_retriever(search_kwargs={"k": k})
