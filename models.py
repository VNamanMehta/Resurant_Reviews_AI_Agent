from pydantic import BaseModel, Field
from typing import List, Optional

class ReviewInput(BaseModel):
    title: str = Field(...,example="Amazing Pizza!")
    review_content: str = Field(...,example="THe pizza was delicious, especially with extra cheese.")
    rating: float = Field(...,ge=0.0, le=5.0, example=4.5)
    date: Optional[str] = Field(None, example="2025-06-08", description="Date in YYYY-MM-DD format.Default is today")

class QuestionInput(BaseModel):
    question: str = Field(...,example="What do people say about the ambiance?")

class QuestionResponse(BaseModel):
    answer: str
    retrieved_context: List[str]

class MessageResponse(BaseModel):
    message: str