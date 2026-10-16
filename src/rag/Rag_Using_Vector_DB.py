#########################################################
# IMPORTS
#########################################################
import os
from groq import Groq
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Load API keys and settings from the .env file
load_dotenv()

#########################################################
# API KEYS
#########################################################
QDRANT_API = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GROQ_API = os.getenv("GROQ_API_KEY")

#########################################################
# MODELS
#########################################################
# Embedding model: transforms sentences into 384-dimensional dense vectors
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# LLM model on Groq used to generate the final human-readable answer
llm_model = "openai/gpt-oss-120b"

#########################################################
# VALIDATIONS
#########################################################
if not QDRANT_API:
    print("Please provide QDRANT_API_KEY in the .env file.")
    exit(1)
if not QDRANT_URL:
    print("Please provide QDRANT_URL in the .env file.")
    exit(1)
if not GROQ_API:
    print("Please provide GROQ_API_KEY in the .env file.")
    exit(1)


#########################################################
# FUNCTION 1: QDRANT CONNECTION
#########################################################
# ----------------------------------------------------------------------
# Purpose : Connects to the remote Qdrant Cloud Vector Database.
# Returns : An active QdrantClient instance.
# Details :
#   - prefer_grpc=True : Uses gRPC protocol for faster vector data transfer.
#   - timeout=60       : Prevents "operation timed out" on slow internet connections.
# ----------------------------------------------------------------------
def qdrant_connection():
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API, 
        prefer_grpc=True, 
        timeout=60
    )
    print("Connected to Qdrant Vector Database Successfully!")
    return client


#########################################################
# FUNCTION 2: KNOWLEDGE BASE LOADING & EMBEDDING
#########################################################
# ----------------------------------------------------------------------
# Purpose : Reads the text file line-by-line and generates vector embeddings.
# Input   : knowledge_file (str) -> Path to the raw text knowledge base file.
# Returns : 
#   - documents          : List of clean text strings (one per line).
#   - document_embedding : NumPy array of 384-dim vector embeddings.
# ----------------------------------------------------------------------
def read_knowledge_file(knowledge_file):
    with open(knowledge_file, "r", encoding="utf-8") as file:
        # Strip trailing newlines and ignore empty lines
        documents = [line.strip() for line in file if line.strip()]
        filename = os.path.basename(knowledge_file)
        print(f"Loaded {len(documents)} documents from {filename}")

        # Convert all sentences into numerical vectors
        document_embedding = embedding_model.encode(documents)
        print("Documents Embedding Created Successfully!")
        return documents, document_embedding


#########################################################
# FUNCTION 3: INSERTING DATA INTO VECTOR DATABASE (INDEXING)
#########################################################
# ----------------------------------------------------------------------
# Purpose : Converts raw text + embeddings into Qdrant "PointStruct" objects
#           and stores (upserts) them inside the vector collection.
# Inputs  : 
#   - documents          : List of original text strings.
#   - document_embedding : Vector representations of each document.
#   - client             : Active QdrantClient instance.
# Details :
#   - .tolist()          : Converts NumPy vector to Python list for JSON compatibility.
#   - payload            : Stores the original text alongside its vector.
# ----------------------------------------------------------------------
def create_qdrant_points(documents, document_embedding, client):
    points = []
    for index, (each_embedding, each_document) in enumerate(zip(document_embedding, documents)):
        points.append(
            PointStruct(
                id=index + 1,                       # Unique integer ID for each point
                vector=each_embedding.tolist(),      # Numerical vector coordinates
                payload={
                    "text": each_document           # Original readable text metadata
                } 
            )
        )
    # Upsert (insert or update) points into the collection
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print("Data successfully added to Qdrant")


#########################################################
# FUNCTION 4: SEMANTIC VECTOR SEARCH (RETRIEVAL)
#########################################################
# ----------------------------------------------------------------------
# Purpose : Finds the most semantically relevant documents for a user's question.
# Inputs  :
#   - user_query : The question asked by the user (string).
#   - top_k      : Number of nearest matching documents to retrieve (default: 3).
# Returns :
#   - context    : A combined string of the top matching facts separated by newlines.
# Details :
#   - Converts the question into a vector using the same embedding model.
#   - Uses cosine distance to find vectors closest to the question vector.
# ----------------------------------------------------------------------
def qdrant_search(user_query, top_k=3):
    # Convert query text into a vector
    query_vector = embedding_model.encode(user_query)

    # Search the vector database for nearest points
    result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    print(f"\nTop {top_k} most relevant documents are:")
    for point in result:
        print(f"  • ID {point.id}: \"{point.payload['text']}\" (Score: {point.score:.4f})")

    # Join the retrieved texts to form the reference context for the LLM
    context = "\n".join([point.payload['text'] for point in result])
    return context


#########################################################
# FUNCTION 5: ASK LLM (GENERATION)
#########################################################
# ----------------------------------------------------------------------
# Purpose : Prompts Groq LLM to generate an answer based ONLY on the retrieved context.
# Inputs  :
#   - user_query : The user's question.
#   - context    : The retrieved relevant facts from Qdrant.
# Returns :
#   - answer     : The LLM's final generated answer.
# Details :
#   - Uses a strict system prompt to avoid hallucinations and external guessing.
# ----------------------------------------------------------------------
def ask_llm(user_query, context):
    system_message = {
        "role": "system",
        "content": f"""You are a helpful assistant. Answer only based on this context.
do not hallucinate. Context: {context}."""
    }

    message = {
        "role": "user",
        "content": user_query
    }
    
    messages = [system_message, message]
    client = Groq(api_key=GROQ_API)
    response = client.chat.completions.create(
        model=llm_model,
        messages=messages
    )

    answer = response.choices[0].message.content
    return answer


#########################################################
# MAIN EXECUTION PIPELINE
#########################################################
COLLECTION_NAME = "knowledge_base"
EMBEDDING_SIZE = 384  # Must match output dimension of all-MiniLM-L6-v2

# 1. Connect to Qdrant
qdrant_client = qdrant_connection()

# 2. Create collection if it doesn't already exist
if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE
        )
    )

# 3. Read knowledge base text and create embeddings
knowledge_file = "../../knowledge_base.txt"
documents, document_embedding = read_knowledge_file(knowledge_file)

# 4. Upload vectors + text into Qdrant
create_qdrant_points(documents, document_embedding, qdrant_client)

# 5. Search for documents relevant to user query
user_query = "What do you know about the Venus?"
top_k = 3
context = qdrant_search(user_query, top_k)

# 6. Ask LLM to answer the question using the retrieved context
answer = ask_llm(user_query, context)
print(f"\nAnswer:\n{answer}\n")