from sys import platform
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from glob import glob 

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
MY_API_KEY = os.getenv("GROQ_API_KEY")

if not MY_API_KEY:
    print("Please provide the GROQ_API_KEY in the .env file.")
    exit(1)

client = Groq(api_key=MY_API_KEY)
model = "openai/gpt-oss-120b"

# Step 1 Knowledge Base
knowledge_base = {
    "age" : "Prajwal is 26 years Old",
    "name" : "My Name is Prajwal",
    "qualification" : "He is a Software Engineer"
}


# Step 2 Retrieval
def retrieve_context(question):
    question = question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "name" in question:
        return knowledge_base["name"]
    elif "qualification" in question:
        return knowledge_base["qualification"]
    else:
        return None

def ask_llm(question):
    context = retrieve_context(question)

    system_message = {
        "role":"system",
        "content":f"""You are a helpful assistant. Answer only based on this context.
            do no hallucinate . Context : {context}.
        """
    }

    message = {
        "role":"user",
        "content":question
    }
    
    messages = [system_message, message]

    response = client.chat.completions.create(
        model = model,
        messages = messages
    )

    answer = response.choices[0].message.content
    return answer

# Passed Prompt
print("Passed Prompt")
print(ask_llm("User Name"))
print(ask_llm("User Age"))
print(ask_llm("User Qualification"))

# Failed Prompt
print("Failed Prompt")
print(ask_llm("What you think about user ?"))
print(ask_llm("How old is user"))
