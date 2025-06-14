from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def get_llm():
    return Ollama(model="llama3.2:latest", base_url="http://localhost:11434")
 # Update if needed

def get_rag_chain(retriever):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a professional job assistant.\n"
            "User query: {question}\n"
            "Relevant jobs:\n{context}\n"
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
