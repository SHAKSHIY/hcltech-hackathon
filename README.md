# HCL Tech- Hackathon
# Mini RAG-Powered Assistant (AI Python)

## 1. Problem Overview

Large Language Models (LLMs) often generate hallucinated or generic responses when answering questions outside their training data. This project aims to build a **Retrieval-Augmented Generation (RAG) assistant** that answers user queries by grounding responses in a **custom document corpus**, ensuring higher accuracy and reliability.

The solution demonstrates practical understanding of **GenAI concepts, system design, and integration logic**, aligned with enterprise use cases.

---

## 2. High-Level System Design

The system follows a modular RAG architecture with clear separation of responsibilities.

**Architecture Flow:**

```
User (Streamlit UI)
        ↓
Query Processing Layer
        ↓
Embedding Model
        ↓
Vector Database (FAISS)
        ↓
Relevant Context (Top-K Chunks)
        ↓
LLM (GPT / Azure OpenAI)
        ↓
Final Answer
```
<img width="760" height="372" alt="image" src="https://github.com/user-attachments/assets/f4be02bc-d490-4856-ab4c-bbbadff93062" />


---

## 3. Design Approach

### 3.1 Corpus Preparation

* Selected **3–5 documents** (PDFs / articles) as the knowledge base.
* Extracted raw text from documents.
* Performed **text chunking** (500–800 tokens) to preserve semantic meaning.
* Stored metadata for traceability.

### 3.2 Embedding & Vector Storage

* Converted document chunks into **vector embeddings** using a pretrained embedding model.
* Stored embeddings locally using **FAISS** for fast similarity search.
* FAISS enables low-latency retrieval and is suitable for hackathon-scale deployments.

### 3.3 Query Handling

* User submits a query via **Streamlit UI**.
* The query is converted into an embedding using the same embedding model.
* FAISS performs **cosine similarity search** to retrieve the top-K relevant chunks.

### 3.4 Response Generation (RAG)

* Retrieved chunks are injected into a structured **prompt template**.
* The LLM generates a **context-aware and grounded response**.
* This approach significantly reduces hallucinations compared to vanilla LLM queries.

---

## 4. Implementation Approach

### Tech Stack

* **Programming Language:** Python
* **Frontend:** Streamlit
* **Backend Logic:** Python modules
* **Vector Store:** FAISS
* **Embeddings:** HuggingFace
* **LLM:** OpenAI GPT / Azure OpenAI
* **Version Control:** GitHub

### Execution Flow

1. One-time document ingestion and FAISS index creation.
2. Real-time user query processing.
3. Retrieval of relevant document chunks.
4. Context-aware answer generation using LLM.
5. Display of final response on Streamlit UI.

---

## 5. Why This Architecture?

| Component      | Justification                                       |
| -------------- | --------------------------------------------------- |
| RAG            | Reduces hallucination and improves factual accuracy |
| FAISS          | Lightweight, fast, hackathon-friendly               |
| Streamlit      | Rapid UI development with minimal overhead          |
| Modular Design | Scalable, maintainable, and testable                |
| Azure Ready    | Aligns with enterprise deployment standards         |

---

## 6. Challenges & Mitigation

* **Hallucination Risk:** Mitigated using retrieval-grounded prompts.
* **Latency Constraints:** Controlled using top-K retrieval and efficient FAISS search.
* **Time Limitations:** Focused on MVP features over unnecessary complexity.

---

## 7. Future Enhancements

* Document upload support via UI
* Hybrid search (keyword + vector)
* Query history and caching
* Role-based access control

---

## 8. Conclusion

This project demonstrates a real-world application of **GenAI + system design**, showcasing how Retrieval-Augmented Generation improves reliability, scalability, and trustworthiness of LLM-based systems in enterprise scenarios.
