# 🤖 Zyro Dynamics HR Assistant

An AI-powered HR assistant built using **Retrieval-Augmented Generation (RAG)** to answer employee questions based only on the company's HR knowledge base.

## 📌 Project Overview

The Zyro Dynamics HR Assistant allows employees to ask questions about company HR policies such as:

- Leave policies
- Reimbursement policies
- Code of conduct
- Other internal HR information

Instead of asking an LLM to answer from its general knowledge, this project first retrieves relevant information from the company's HR documents and then uses an LLM to generate the answer.

This helps make the responses more relevant to the provided knowledge base.

## 🧠 How It Works

The application follows a **Retrieval-Augmented Generation (RAG)** pipeline:

```text
User Question
      ↓
FastAPI Backend
      ↓
Scope / Guardrail Check
      ↓
Question Embedding
      ↓
FAISS Similarity Search
      ↓
Relevant HR Document Chunks
      ↓
Groq LLM
      ↓
Generated Answer
      ↓
Frontend
1. Document Loading

HR PDF documents are loaded from the local knowledge base.

2. Text Chunking

The documents are divided into smaller chunks so that relevant information can be retrieved efficiently.

3. Embeddings

Each document chunk is converted into a numerical vector using:

Hugging Face sentence-transformers/all-MiniLM-L6-v2

4. Vector Database

The embeddings are stored in FAISS, which is used to find document chunks that are semantically similar to the user's question.

5. Retrieval

For each question, the system retrieves the most relevant document chunks.

6. LLM Generation

The retrieved context is provided to a Groq-hosted LLM, which generates the final answer.

7. Guardrails

A scope classifier checks whether the question is related to HR.

Questions outside the HR domain are refused instead of being answered using general knowledge.

🛠️ Technologies Used
Python
LangChain
Hugging Face Embeddings
FAISS
Groq LLM
FastAPI
Jinja2
HTML
CSS
JavaScript
python-dotenv