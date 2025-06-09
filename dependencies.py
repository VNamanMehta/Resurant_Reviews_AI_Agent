# dependencies.py
from typing import Any, Optional
from fastapi import HTTPException, status

_rag_chain_instance: Optional[Any] = None

def set_rag_chain_instance(chain: Any):
    """Setter function for the global RAG chain instance."""
    global _rag_chain_instance
    _rag_chain_instance = chain

def get_rag_chain():
    """Dependency that provides the initialized RAG chain."""
    if _rag_chain_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized. Please try again later."
        )
    return _rag_chain_instance