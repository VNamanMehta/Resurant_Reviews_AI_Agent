from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from typing import Any

from vectorSearch import get_vector_store, RETRIEVER_K

LLM_MODEL_NAME = "gemma3:1b"

def get_llm_model() -> OllamaLLM:
    """Initializes and returns the Ollama LLM model."""
    print(f"Initializing LLM: {LLM_MODEL_NAME}")
    return OllamaLLM(model=LLM_MODEL_NAME)

def setup_rag_chain() -> Any:
    """
    Sets up the RAG chain using the LLM and retriever.
    This function handles the core logic of querying the vector store
    and passing relevant context to the LLM.
    """
    llm = get_llm_model()
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

    rag_template = """
    You are an expert in answering questions about a pizza restaurant.
    Use the following pieces of retrieved restaurant reviews to answer the question.
    If you don't know the answer based on the provided reviews, just say that you don't know,
    don't try to make up an answer.

    Context: {context}

    Question: {input}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    document_combiner_chain = create_stuff_documents_chain(llm, rag_prompt)
    '''
    above line:
    All retrieved documents are concatenated together into one long string of context,
    which is then fed into the prompt for the LLM and returns the LLM's answer.
    '''

    rag_chain = create_retrieval_chain(retriever, document_combiner_chain)
    '''
    above line:
    This wraps the entire RAG pipeline:
    it retrieves relevant docs from a vector store
    passes them to the document combiner (like the "stuff" chain)
    gets a final response from the LLM
    '''
    
    print("RAG chain setup complete.")
    return rag_chain

def get_updated_rag_chain() -> Any:
    """
    Re-initializes the RAG chain.
    Useful when the underlying vector store might have been updated (e.g., new reviews added).
    """
    print("Re-initializing RAG chain to pick up latest vector store data...")
    return setup_rag_chain()