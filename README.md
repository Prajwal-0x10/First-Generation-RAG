# Retrieval-Augmented Generation (RAG) 🚀

This repository demonstrates the evolution of **Retrieval-Augmented Generation (RAG)** systems, starting from basic **First-Generation (Naive) RAG** to advanced and agentic RAG architectures.

---

## 📌 Project Overview

**Retrieval-Augmented Generation (RAG)** bridges the gap between static LLMs and dynamic/private datasets. Instead of relying solely on pre-trained weights, RAG retrieves relevant context from a knowledge base before generating responses.

### 🧩 First-Generation (Naive) RAG

First-Generation RAG follows a simple 3-step linear pipeline:
1. **Indexing / Knowledge Base**: Storing raw data chunks or key-value entries.
2. **Retrieval**: Performing direct lookup or vector similarity search based on the user's query.
3. **Generation**: Injecting retrieved context into the LLM system prompt to generate an answer.

#### ⚖️ Pros & Cons

| Pros | Cons |
| :--- | :--- |
| **Fast Setup**: Easy to prototype and build in an afternoon. | **Phrasing Sensitivity**: Simple lookup fails when query phrasing differs from stored keywords. |
| **Grounding**: Reduces hallucinations by providing exact context. | **No Multi-Hop Reasoning**: Struggles to synthesize facts across multiple files or entries. |
| **Cost-Effective**: Single-pass retrieval without complex loops. | **Context Noise**: Vulnerable to "Lost in the Middle" syndrome with large text chunks. |

---

## 📁 Repository Structure

```text
RAG/
├── README.md                  # Project Documentation
├── pyproject.toml             # Python dependencies and build config
└── src/
    └── rag/
        ├── __init__.py
        └── First_Generation_Rag.py  # First-Generation RAG implementation
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites

- Python 3.10+ (or Python 3.14 via `uv`)
- A **Groq API Key** (or LLM provider key)

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Installation

Using `uv` (recommended):
```bash
uv sync
```

Or using standard `pip`:
```bash
pip install groq python-dotenv pydantic
```

---

## 🚀 Running First-Generation RAG

Execute the python script directly:

```bash
python src/rag/First_Generation_Rag.py
```

Or with `uv`:
```bash
uv run python src/rag/First_Generation_Rag.py
```

### 🔍 Code Example & Output

```python
# Knowledge Base Lookup
Passed Prompt:
Query: "User Name"          -> Answer: "The user's name is Prajwal." ✅
Query: "User Age"           -> Answer: "Prajwal is 26 years old." ✅
Query: "User Qualification" -> Answer: "Prajwal is a Software Engineer." ✅

Failed Prompt:
Query: "What do you think about user?" -> Fails (No matching keyword context) ❌
Query: "How old is user"               -> Fails (Phrasing mismatch) ❌
```

---

## 👤 Author

**Prajwal Gawande**
- GitHub: [@Prajwal-0x10](https://github.com/Prajwal-0x10)
- Email: prajwalgawande84@gmail.com
