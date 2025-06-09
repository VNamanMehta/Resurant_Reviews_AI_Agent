from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
from routes import router
from main import setup_rag_chain
from dependencies import set_rag_chain_instance, get_rag_chain

from models import MessageResponse 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application startup and shutdown events.
    Initializes the RAG chain when the application starts.
    content before yeild is executed on app startup and content after yeild is executed upon shutdown.
    """
    print("--- Initializing RAG components on startup ---")
    try:
        initialized_chain = setup_rag_chain()
        set_rag_chain_instance(initialized_chain) 
        print("--- RAG components initialized successfully ---")
    except Exception as e:
        print(f"Error during RAG component initialization: {e}")
        raise RuntimeError(f"Failed to initialize RAG components: {e}")

    yield

    print("--- Shutting down RAG API ---")

app = FastAPI(
    title="Pizza Restaurant RAG API",
    description="API for querying and adding reviews to a pizza restaurant RAG system.",
    lifespan=lifespan
)

app.include_router(router, prefix="/api") 

@app.get("/health", response_model=MessageResponse)
async def health_check():
    """Simple health check endpoint."""
    try:
        rag_chain = get_rag_chain()
        return {"message": "RAG API is healthy and ready."}
    except HTTPException as e:
        raise e
