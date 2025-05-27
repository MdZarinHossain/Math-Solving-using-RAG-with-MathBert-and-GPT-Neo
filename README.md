# Math Problem Solving using RAG with MathBERT and GPT-Neo

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to solve math problems using contextually similar examples. The solution retrieves relevant problems and solutions from a dataset using **MathBERT** embeddings and generates answers using **GPT-Neo**.

## 📁 Dataset

The dataset consists of structured math problems stored in JSON format with fields such as:
- `problem`: The problem statement
- `solution`: The detailed solution

The JSON files are located inside:

```
/content/drive/MyDrive/Capstone/MATH/train/
```

## 📊 Data Processing & Embedding

- Problems and solutions are extracted and cleaned.
- Text is split into manageable chunks for embedding.
- **MathBERT (`tbs17/MathBERT`)** is used to generate dense vector embeddings for both problems and solutions.
- Embeddings are indexed using **FAISS** (Facebook AI Similarity Search) for efficient retrieval.

## 🧠 Retrieval-Augmented Generation (RAG) Pipeline

### Embedding
- `generate_embedding_mathbert(content)` uses MathBERT to encode text.
- Both problem and solution texts are embedded and stored in FAISS.

### Retrieval
- Given a query, the top-k most similar problem/solution embeddings are retrieved using FAISS.
- Retrieved contexts are combined to form the prompt for the language model.

### Generation
- A prompt template is used:
  ```
  You are a math tutor. Use the following retrieved context to answer the question.
  Context: {context}
  Question: {question}
  Answer:
  ```
- **GPT-Neo (`EleutherAI/gpt-neo-1.3B`)** generates the answer based on the prompt.

## 🧪 Requirements

Install required libraries:

```bash
pip install langchain openai faiss-cpu sentence-transformers transformers tiktoken langchain-community torch accelerate
```

## 💾 Saving and Loading Index

- Embeddings and FAISS index are serialized using `pickle` and `faiss.write_index`.
- Files saved:
  - `faiss_index.idx`
  - `embedding_map.pkl`
- Stored in Google Drive for persistence.

## ⚙️ Example Query

```python
query = "What is the sum of all possible whole number values of n using Triangle Inequality?"
response = rag_pipeline(query)
print(response)
```

The pipeline retrieves related problems and generates a solution using the retrieved context.

## 📈 Output

- Logs token lengths and retrieved context.
- Displays retrieved examples and the generated solution.
- Verifies embedding count for quality assurance.

## 💡 Technologies Used

- **MathBERT**: For domain-specific embedding of math problems.
- **FAISS**: For efficient vector similarity search.
- **GPT-Neo**: For autoregressive answer generation.
- **LangChain**: For prompt templates and modular RAG architecture.
