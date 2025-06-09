# routes.py
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Any
from vectorSearch import add_review_to_csv_and_db
from main import get_updated_rag_chain
from models import ReviewInput, QuestionInput, QuestionResponse, MessageResponse
from dependencies import get_rag_chain, set_rag_chain_instance 

router = APIRouter()

@router.post("/add_review", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def add_review_endpoint(
    review: ReviewInput 
):
    """
    Adds a new review to the restaurant's review dataset (CSV) and updates the vector database.
    """
    try:
        add_review_to_csv_and_db(
            review.title,
            review.review_content,
            review.rating,
            review.date
        )

        updated_chain = get_updated_rag_chain()
        
        set_rag_chain_instance(updated_chain)
        
        return {"message": "Review added successfully and database updated."}
    except Exception as e:
        print(f"Error adding review: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to add review: {str(e)}")

@router.post("/ask_question", response_model=QuestionResponse)
async def ask_question_endpoint(
    query: QuestionInput,
    rag_chain: Any = Depends(get_rag_chain) # Depends() - makes get_rag_chain() run before the endpoint is called.
):
    """
    Answers a question about the restaurant using retrieved reviews.
    """
    try:
        response = rag_chain.invoke({"input": query.question})
        
        retrieved_texts = [doc.page_content for doc in response.get("context", [])]

        return QuestionResponse(
            answer=response.get("answer", "No answer generated."),
            retrieved_context=retrieved_texts
        )
    except Exception as e:
        print(f"Error asking question: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process question: {str(e)}")