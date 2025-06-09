# Pizza Restaurant RAG API

A FastAPI-based Retrieval-Augmented Generation (RAG) system for querying and adding reviews to a pizza restaurant dataset. This project leverages vector search, embeddings, and large language models to provide intelligent, context-aware answers to user questions about a restaurant, based on real customer reviews.

---

## Features

- **Semantic Search:** Uses vector embeddings to retrieve the most relevant restaurant reviews for a given question.
- **RAG Pipeline:** Combines retrieved context with an LLM (Large Language Model) to generate accurate, grounded answers.
- **Review Management:** Allows users to add new reviews, which are immediately indexed and available for future queries.

---

## How the Agent Works

1. **Data Ingestion:** Loads restaurant reviews from a CSV file, splits them into manageable chunks, and generates embeddings using the `nomic-embed-text:v1.5` model.
2. **Vector Store:** Stores embeddings and metadata in a ChromaDB vector database for efficient similarity search.
3. **Question Answering:**
   - User submits a question.
   - The question is embedded and used to retrieve the most relevant review chunks from the vector store.
   - Retrieved context is combined with the user’s question and passed to the `gemma3:1b` LLM.
   - The LLM generates an answer grounded in the retrieved reviews.
4. **Review Addition:** New reviews are appended to the CSV and immediately embedded and indexed in the vector database, ensuring up-to-date responses.

---

## Tools & Technologies Used

- **FastAPI:** Web framework for building the API.
- **LangChain:** Framework for chaining LLMs and retrieval components.
- **ChromaDB:** Vector database for storing and searching embeddings.
- **Ollama:** For running local LLMs and embedding models.
- **Pandas:** For CSV data manipulation.
- **Python 3.11+**

---

## API Endpoints

| Endpoint                | Method | Description                                      |
|-------------------------|--------|--------------------------------------------------|
| `/api/add_review`       | POST   | Add a new restaurant review                      |
| `/api/ask_question`     | POST   | Ask a question about the restaurant              |
| `/health`               | GET    | Health check for the API                         |

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with `nomic-embed-text:v1.5` and `gemma3:1b` models available

### Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd AI_Agent_RAG_talk_with_csv_pdf
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the API:**
   ```sh
   uvicorn app:app --reload
   ```

4. **Access the API docs:**
   - Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger UI.

---

## Project Structure

- [`app.py`](app.py): FastAPI app and startup logic
- [`main.py`](main.py): RAG chain setup and LLM integration
- [`vectorSearch.py`](vectorSearch.py): Embedding, vector store, and review management
- [`routes.py`](routes.py): API endpoints
- [`models.py`](models.py): Pydantic models for request/response validation
- [`dependencies.py`](dependencies.py): Dependency injection for RAG chain
- [`realistic_restaurant_reviews.csv`](realistic_restaurant_reviews.csv): Review dataset
- [`chroma_lanchain_db/`](chroma_lanchain_db/): Vector database files

---

## Example Usage

### Add a Review

```json
POST /api/add_review
{
  "title": "Amazing Pizza!",
  "review_content": "The pizza was delicious, especially with extra cheese.",
  "rating": 4.5,
  "date": "2025-06-08"
}
```

### Ask a Question

```json
POST /api/ask_question
{
  "question": "What do people say about the crust?"
}
```

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://www.trychroma.com/)
- [Ollama](https://ollama.com/)
- [FastAPI](https://fastapi.tiangolo.com/)

---
