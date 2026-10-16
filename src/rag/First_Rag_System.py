import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from glob import glob 
import sys
import os


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
MY_API_KEY = os.getenv("GROQ_API_KEY")

if not MY_API_KEY:
    print("Please provide the GROQ_API_KEY in the .env file.")
    exit(1)

client = Groq(api_key=MY_API_KEY)
llm_model = "openai/gpt-oss-120b"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


knowledge_base = [
    "Honey never spoils. Archaeologists have found edible honey in ancient Egyptian tombs.",
    "Octopuses have three hearts and blue blood.",
    "A day on Venus is longer than its year.",
    "Bananas are botanically classified as berries, but strawberries are not true berries.",
    "Sharks existed before trees appeared on Earth.",
    "The Eiffel Tower can grow taller during hot weather due to thermal expansion.",
    "Wombats produce cube-shaped poop.",
    "An adult human body has 206 bones.",
    "The Moon is slowly moving away from Earth by about 3.8 centimeters per year.",
    "A group of flamingos is called a flamboyance.",
    "Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid of Giza.",
    "Some turtles can absorb oxygen through specialized tissues near their tails while underwater.",
    "The shortest recorded war in history was the Anglo-Zanzibar War of 1896, lasting less than an hour.",
    "Your brain uses roughly 20% of your body's energy at rest.",
    "Dolphins use unique signature whistles that function somewhat like individual names."
]

knowledge_base_embedding = embedding_model.encode(knowledge_base)

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def retrive_context(query_embedding):
    context_list = []
    for index, each_knowledge in enumerate(knowledge_base_embedding):
        similarity = cosine_similarity(query_embedding, each_knowledge)
        context_list.append((similarity, knowledge_base[index]))
    context_list.sort(reverse=True, key= lambda x: x[0])
    return context_list[0]

def ask_llm(query, context):

    system_message = {
        "role":"system",
        "content":f"""You are a helpful assistant. Answer only based on this context.
            do no hallucinate . Context : {context}.
        """
    }

    message = {
        "role":"user",
        "content":query
    }
    
    messages = [system_message, message]

    response = client.chat.completions.create(
        model = llm_model,
        messages = messages
    )

    answer = response.choices[0].message.content
    return answer

query = "How long does honey last?"
query_embedding = embedding_model.encode([query])
context = retrive_context(query_embedding)
answer = ask_llm(query, context)
print(answer)

