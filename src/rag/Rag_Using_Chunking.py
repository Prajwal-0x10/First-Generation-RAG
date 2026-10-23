#########################################################
# IMPORTS
#########################################################
import os
from groq import Groq
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchAny, MatchValue, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

load_dotenv()


#########################################################
# API KEYS
#########################################################
QDRANT_API = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GROQ_API = os.getenv("GROQ_API_KEY")

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

def qdrant_connection():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API)
    print(f"Connected to Qdrant Vector Database {QDRANT_URL} Successfully!")
    return client

#########################################################
# MODELS
#########################################################
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm_model = "openai/gpt-oss-120b"

def read_knowledge_file(knowledge_file):
    with open(knowledge_file,"r") as file:
        text = file.read()

        ##### FIXED CHUNKING PROCESS ######
        splitter = CharacterTextSplitter(
            separator = "",
            chunk_size = 50,
            chunk_overlap = 0
        )

        #### RECURSIVE CHUNKING PROCESS #####
        recursive_splitter = RecursiveCharacterTextSplitter(
            separators = ["\n\n", "\n", ".", " "],
            chunk_size = 50,
            chunk_overlap = 10
        )

        document = recursive_splitter.split_text(text)
        print(document)
        document_embedding = embedding_model.encode(document)
        return document, document_embedding


def create_qdrant_points(document, document_embedding, client):
    points = []
    for index, (text, each_embedding) in enumerate(zip(document, document_embedding)):
        points.append(
            PointStruct(
                id = index + 1,
                vector = each_embedding.tolist(),
                payload = {
                    "text":text
                }
            )
        )
    client.upsert(
        collection_name = COLLECTION_NAME,
        points = points
    )        
    print("Data successfully added to Qdrant")

def qdrant_search(user_query, top_k):
    query_vector = embedding_model.encode(user_query)
    result = client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_vector,
        limit = top_k,
        with_payload = True).points

    print(f"Top {top_k} most relevant documents are")

    for point in result:
        print(f"{point.id}. {point.payload['text']} Score {point.score}")

    context = "\n".join([point.payload['text'] for point in result])
    return context

def ask_llm(user_query, context):
    llm = Groq(api_key = GROQ_API)
    response = llm.chat.completions.create(
        model = llm_model,
        messages = [
            {"role":"system", "content":"Answer the following question based on the context provided"},
            {"role":"user", "content":f"Context: {context}\n\nQuestion: {user_query}"}
        ]
    )
    return response.choices[0].message.content

COLLECTION_NAME = "chunk_documents"
EMBEDDING_SIZE = 384

client = qdrant_connection()
if not client.collection_exists(collection_name = COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config = VectorParams(
            size = EMBEDDING_SIZE,
            distance = Distance.COSINE
        )
    )

knowledge_file = "knowledge_base_chunk.txt"
document, document_embedding = read_knowledge_file(knowledge_file)
create_qdrant_points(document, document_embedding, client)
user_query = "What is AI?"
top_k = 3
context = qdrant_search(user_query,top_k)       
answer = ask_llm(user_query,context)
print(answer)