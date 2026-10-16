# Retrieval-Augmented Generation (RAG) 🚀

This repository demonstrates the step-by-step architectural evolution of **Retrieval-Augmented Generation (RAG)** systems, progressing from basic keyword lookup to in-memory vector embeddings, and finally to a scalable vector database using **Qdrant** and **Groq**.

---

## 📌 RAG Architectural Evolution

| Level | File | Retrieval Method | Storage | Semantic Understanding | Scalability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Level 1** | `First_Generation_Rag.py` | Keyword matching (`if "keyword" in query`) | Python `dict` in code | ❌ None (exact matches only) | ❌ Very Low |
| **Level 2** | `First_Rag_System.py` | Dense vector similarity (`all-MiniLM-L6-v2`) | NumPy array in memory | ✅ High (understands meaning) | ⚠️ Medium (RAM-bound) |
| **Level 3** | `Rag_Using_Vector_DB.py` | Approximate Nearest Neighbor (ANN) search | **Qdrant Cloud** Vector DB | ✅ High (HNSW vector indexing) | 🚀 High (Production-grade) |

---

## 📁 Repository Structure

```text
RAG/
├── README.md                         # Project documentation
├── pyproject.toml                    # UV / Python dependencies configuration
├── knowledge_base.txt                # Raw text knowledge base facts
└── src/
    └── rag/
        ├── __init__.py
        ├── First_Generation_Rag.py   # Level 1: Keyword-based Naive RAG
        ├── First_Rag_System.py       # Level 2: In-Memory Semantic RAG (NumPy + Embeddings)
        ├── Rag_Using_Vector_DB.py    # Level 3: Production RAG with Qdrant Vector DB
        └── knowledge_base.txt        # Knowledge base copy for module access
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.10+ (or Python 3.14 via `uv`)
- A **Groq API Key** ([Groq Console](https://console.groq.com/))
- A **Qdrant Cloud API Key & Cluster URL** ([Qdrant Cloud](https://cloud.qdrant.io/))

### 2. Environment Variables
Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_URL=https://your-cluster-id.us-east-4-0.aws.cloud.qdrant.io:6333
```

### 3. Installation

Using `uv` (recommended):
```bash
uv sync
```

Or using standard `pip`:
```bash
pip install groq python-dotenv pydantic qdrant-client sentence-transformers numpy
```

---

## 🔬 Detailed Implementation Breakdown

### 1. `First_Generation_Rag.py` (Keyword / Naive RAG)

The simplest form of RAG. It stores facts in a static Python dictionary and checks whether specific keywords appear in the user's question.

- **How it works:**
  1. Defines a dictionary `knowledge_base` with keys like `"name"`, `"age"`, `"qualification"`.
  2. Scans the user query string using `if "keyword" in question:`.
  3. Sends the matched string as context to Groq LLM (`openai/gpt-oss-120b`).

- **Strengths:** Instant, zero embedding computation, no database needed.
- **Limitations:** Extremely brittle. If the user asks *"How old is the user?"* instead of using the word *"age"*, the retrieval returns `None` and the system fails.

**Run it:**
```bash
python src/rag/First_Generation_Rag.py
```

---

### 2. `First_Rag_System.py` (In-Memory Semantic Search with NumPy)

Solves the phrasing sensitivity issue by translating text into dense vector embeddings and calculating mathematical similarity.

- **How it works:**
  1. Stores knowledge statements in a Python list.
  2. Encodes each statement into a 384-dimensional dense vector using `SentenceTransformer("all-MiniLM-L6-v2")`.
  3. Converts the user query into a vector and computes **Cosine Similarity** with each knowledge vector using NumPy:
     $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
  4. Selects the highest-scoring fact and feeds it to Groq LLM as context.

- **Strengths:** True semantic understanding. Asking *"How long does honey last?"* correctly retrieves *"Honey never spoils..."* even though the words do not overlap.
- **Limitations:** Every query requires an exhaustive linear scan ($O(N)$) across all vectors stored in RAM. Not feasible for large datasets.

**Run it:**
```bash
python src/rag/First_Rag_System.py
```

---

### 3. `Rag_Using_Vector_DB.py` (Production-Ready RAG with Qdrant)

A complete, production-grade RAG pipeline integrating an external knowledge base file, an enterprise vector database (**Qdrant**), and Groq LLM.

- **Key Highlights:**
  - **Modular Pipeline:** Structured into clear helper functions (`qdrant_connection`, `read_knowledge_file`, `create_qdrant_points`, `qdrant_search`, `ask_llm`).
  - **External Ingestion:** Dynamically parses text line-by-line from `knowledge_base.txt`.
  - **Robust Cloud Connection:** Configured with `timeout=60` and `prefer_grpc=True` for high-throughput, low-latency vector operations over remote connections.
  - **Qdrant Indexing:** Checks collection existence and upserts `PointStruct` objects containing vectors and metadata payloads.
  - **Top-K Vector Retrieval:** Uses `query_points()` to retrieve the $K$ most relevant passages based on cosine distance.
  - **Anti-Hallucination Prompting:** Injects retrieved context into Groq LLM with strict guardrails to prevent hallucination.

**Run it:**
```bash
python src/rag/Rag_Using_Vector_DB.py
```

**Sample Output:**
```text
Connected to Qdrant Vector Database Successfully!
Loaded 15 documents from knowledge_base.txt
Documents Embedding Created Successfully!
Data successfully added to Qdrant

Top 3 most relevant documents are:
  • ID 3: "A day on Venus is longer than its year." (Score: 0.6653)
  • ID 5: "Sharks existed before trees appeared on Earth." (Score: 0.2278)
  • ID 11: "Cleopatra lived closer in time to the Moon landing..." (Score: 0.2058)

Answer:
Based on the provided context, a day on Venus (one full rotation on its axis) is longer than its year (one orbit around the Sun).
```

---

## 👤 Author

**Prajwal Gawande**
- GitHub: [@Prajwal-0x10](https://github.com/Prajwal-0x10)
- Email: prajwalgawande84@gmail.com
