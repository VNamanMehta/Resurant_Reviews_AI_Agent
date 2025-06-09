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

from csv import QUOTE_ALL
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas as pd
import datetime

DB_LOCATION = "./chroma_lanchain_db"
COLLECTION_NAME = "restaurant_reviews"
EMBEDDING_MODEL_NAME = "nomic-embed-text:v1.5"
RETRIEVER_K = 5 # Number of documents to retrieve for each query
CSV_FILE_PATH = "realistic_restaurant_reviews.csv"

CSV_COLUMNS = ["Title", "Date", "Rating", "Review"]

def ensure_csv_exists(file_path=CSV_FILE_PATH):
    """Ensures the CSV file exists and has the correct columns, if not, creates it an empty one."""
    if not os.path.exists(file_path):
        print(f"CSV file {file_path} does not exist, creating an empty one.")
        df_empty = pd.DataFrame(columns=CSV_COLUMNS)
        df_empty.to_csv(file_path, index=False)

def get_embedding_model():
    """Initializes and returns the Ollama embeddings model."""
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    return OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

def load_reviews_from_csv(file_path=CSV_FILE_PATH):
    """Loads reviews from a CSV file and converts them into LangChain Document objects."""
    ensure_csv_exists(file_path)
    print(f"Loading reviews from CSV file: {file_path}")

    df = pd.read_csv(file_path)
    documents = []
    for i, row in df.iterrows():
        full_content = f"Title: {row['Title']}\nReview: {row['Review']}"
        document = Document(
            page_content=full_content,
            metadata={
                "rating": row["Rating"],
                "date": row["Date"],
                "original_csv_row_id": str(i)
            },
            id=f"review_doc_{i}"
        )
        documents.append(document)
    print(f"Loaded {len(documents)} raw documents from CSV.")
    return documents # it is a list of Document objects
    

def chunk_documents(documents: list[Document]):
    """Splits documents into smaller chunks for better processing, due to LLM token limits"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    print(f"Chunking {len(documents)} documents.")
    chunked_documents = text_splitter.split_documents(documents)

    '''
    Below we assign ids to each of the chunk.
    doc is the chunk, metadata is propogated from the parent (full document) to the chunk (doc).
    "unknown is the default value in case the original id is not found"
    id is an essential requirement by the chromaDB
    '''
    for i, doc in enumerate(chunked_documents):
        original_doc_id = doc.metadata.get("original_csv_row_id", "unknown")
        doc.id = f"review_chunk_{original_doc_id}_{i}"

    print(f"Created {len(chunked_documents)} chunks.")
    return chunked_documents

def get_vector_store():
    """
    Initializes or loads the ChromaDB vector store.
    Indexes documents if the database doesn't exist or is empty.
    """
    embeddings = get_embedding_model()
    
    db_exists_and_populated = os.path.exists(DB_LOCATION) and bool(os.listdir(DB_LOCATION))

    if not db_exists_and_populated:
        print(f"ChromaDB not found or empty at {DB_LOCATION}. Creating and indexing documents...")
        raw_documents = load_reviews_from_csv()
        chunked_docs = chunk_documents(raw_documents)
        """
        Chroma.from_documents()
        This is a static factory method used to create a *new* Chroma vector store
        and immediately populate it with a list of documents.
                Working:
        1. It internally initializes a fresh ChromaDB instance (or collection).
        2. For each 'Document' object in the 'documents' list provided:
           a. It uses the 'embedding' function to generate a numerical vector (embedding)
              for the 'page_content' of that document.
           b. It stores this generated embedding, along with the original 'page_content'
              and its 'metadata', into the ChromaDB.
        3. It returns this newly created and populated in-memory ChromaDB object.
                When to use:
        - Initial setup: When you're creating your vector database for the first time
          from a collection of raw text documents.
        - Rebuilding: If you need to completely re-index all your data from scratch.
        """
        vector_store = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=DB_LOCATION
        )
        """
        vector_store.persist()
        This method is called on an *in-memory* ChromaDB object to save its current
        state (embeddings, documents, and database structure) to the specified
        'persist_directory' on disk. Without calling this, any data added to
        the ChromaDB in memory would be lost when the Python script/application terminates.
        """
        vector_store.persist()
        print("ChromaDB created and documents indexed.")
    else:
        print(f"Loading existing ChromaDB from {DB_LOCATION}")
        """
        Chroma() - Constructor
        This is the standard constructor for the Chroma class. When used with
        'persist_directory', its primary function is to *load* an existing
        ChromaDB instance that was previously saved to disk.
    
        Working:
        1. It does NOT generate new embeddings or add new documents.
        2. It reads the database files from the specified 'persist_directory'.
        3. It reconstructs the in-memory representation of the ChromaDB,
           including all its stored embeddings, documents, and metadata.
        4. The 'embedding_function' *must* be provided here. This tells Chroma
           which embedding model was originally used to create the stored embeddings.
           It's crucial for consistency when performing similarity searches later,
           as your query will be embedded using this same function.
    
        When to use:
        - Loading: When you want to retrieve a pre-existing vector database
          from disk to perform operations like retrieval or add new documents.
        """
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_LOCATION,
            embedding_function=embeddings
        )
        print("ChromaDB loaded")

    return vector_store

def add_review_to_csv_and_db(title: str, review_content: str, rating: float, date: str = None):
    ensure_csv_exists(CSV_FILE_PATH)

    if date is None:
        date = datetime.date.today().strftime("%Y-%m-%d")

    # Open the CSV file in read and append mode with UTF-8 encoding.
    # 'a+' allows us to read and write (append) at the same time.
    with open(CSV_FILE_PATH, 'a+', encoding='utf-8') as f:
        # Move the file pointer to the end of the file to check if it's empty
        f.seek(0, os.SEEK_END)

        # Check if the file is not empty
        if f.tell() > 0:
            # Move the file pointer one byte back to read the last character
            f.seek(f.tell() - 1, os.SEEK_SET)

            # If the last character is not a newline, write a newline character
            # This ensures the new row doesn't get appended on the same line
            if f.read(1) != '\n':
                f.write('\n')


    new_review_df = pd.DataFrame([{
        "Title": title,
        "Date": date,
        "Rating": rating,
        "Review": review_content
    }])

    #check if file exists at the path and if it is empty or not, if it exists only then check if empty or not,
    # else the os.stat() will through file not found error
    write_header = not os.path.exists(CSV_FILE_PATH) or os.stat(CSV_FILE_PATH).st_size == 0
    new_review_df.to_csv(CSV_FILE_PATH, mode="a", header=write_header, index=False,
                         lineterminator="\n", quoting=QUOTE_ALL)
    print(f"✅ New review appended to {CSV_FILE_PATH}")

    current_df = pd.read_csv(CSV_FILE_PATH)

    new_review_csv_index = len(current_df) - 1

    new_doc_content = f"Title: {title}\nReview: {review_content}"
    new_doc = Document(
        page_content=new_doc_content,
        metadata={
            "rating": rating,
            "date": date,
            "original_csv_row_id": str(new_review_csv_index)
        }
    )

    chunked_new_docs = chunk_documents([new_doc])
    current_vector_store = get_vector_store()
    current_vector_store.add_documents(chunked_new_docs)
    if hasattr(current_vector_store, "persist"):
        current_vector_store.persist()
    print("✅ New review indexed into ChromaDB")