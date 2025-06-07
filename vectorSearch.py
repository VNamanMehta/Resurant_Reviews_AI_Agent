"""
vector search (semantic search or similarity search) - finds items (documents, images, audio etc)",
that are semantically similar to a given query, rather than just matching keywords.

embedding - text and other data is converted into a numerical list (embeddings/vectors),
Items with similar meanings will have similar vectors.

vector database(vector store) - a specialized database designed to store and search these vectors efficiently.
they allow for fast similarity searches, often using techniques like approximate nearest neighbors (ANN).

querying - when a user inputs a query, it is also converted into a vector using the same embedding model.

similarity search - the vector database compares the query vector to the stored vectors (usually using distance metrics like cosine similarity) and retrieves items
"""

"""
steps when data(reviews) is not already embedded or when it changes: 
1. Load your data: Get your restaurant reviews (or other documents) from a source (e.g., a text file, a database, a list of strings).
Example: Let's say you have a list of strings, each being a review.

2. Chunk your data: If reviews are long, break them into smaller, manageable chunks.
This is important because LLMs have token limits, and you only want to retrieve the most relevant parts of a review.

3. Create Embeddings: Use your nomic-embed-text:v1.5 model to convert each chunk of text into an embedding (vector).

4. Store in Vector Database: Store these embeddings and their corresponding original text chunks in ChromaDB.
"""

"""
steps when question is asked:
1. User Question: The user asks a question about the restaurant.

2. Embed User Question: Use the same nomic-embed-text:v1.5 model to convert the user's question into an embedding.

3. Vector Search (Retrieve): Query ChromaDB with the user's question embedding to find the top k (e.g., top 3 or 5) most semantically similar review chunks.

4. Augment Prompt: Take these retrieved review chunks and inject them into your template as the {reviews} context.

5. Generate Answer: Pass the augmented prompt to gemma3:1b (via your prompt | model chain),
   which will then generate an answer using both its own knowledge and the provided relevant reviews.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")

embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

db_location = "./chroma_lanchain_db"

add_documents = not os.path.exists(db_location) # If the db does not exist (not False) - add documents to db, if it exists (not True) - do nothing

if add_documents:
    documents = []
    ids = [] # List to store document IDs, which is required by the vectorstore

    for i, row in df.iterrows():
        document = Document(page_content=row["Title"] + row["Review"], metadata={"rating": row["Rating"],"date": row["Date"]},
                            id = str(i))
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)