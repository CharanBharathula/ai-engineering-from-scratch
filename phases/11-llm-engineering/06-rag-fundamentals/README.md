# Lesson 06: RAG (Retrieval-Augmented Generation) Fundamentals

LLMs have a knowledge cutoff date and hallucinate when asked about private data. **RAG** solves this by injecting your private documents directly into the prompt *before* the LLM answers.

## The RAG Architecture Loop

1. **Ingestion Phase (Done Once)**
   - Extract text from documents (PDFs, Notion, Confluence).
   - Chunk the text into smaller pieces (e.g., 500 words per chunk).
   - Convert chunks into Embeddings.
   - Store the vectors and the original text in a Vector Database.

2. **Retrieval & Generation Phase (Done on every user query)**
   - User asks: *"What is our refund policy?"*
   - Convert the user's query into an embedding.
   - Search the Vector Database for the top 3 closest chunks.
   - Inject those 3 chunks into the LLM Prompt as context.
   - The LLM reads the context and answers the user.

## Conceptual Code Implementation

```python
from openai import OpenAI

client = OpenAI()

# SIMULATED DATABASE (In production, use Pinecone, Qdrant, or pgvector)
vector_db = [
    {"text": "The company was founded in 2012 by Sarah Connor.", "vector": [0.1, 0.2, ...]},
    {"text": "Refunds are processed within 14 business days.", "vector": [0.8, 0.1, ...]}
]

def retrieve_context(user_query):
    # 1. Convert query to vector (Skipping math for simplicity)
    # 2. Search vector_db for highest cosine similarity
    # 3. Return the text of the best match
    return "Refunds are processed within 14 business days."

def generate_rag_response(user_query):
    # STEP 1: RETRIEVAL
    context = retrieve_context(user_query)
    
    # STEP 2: AUGMENTATION & GENERATION
    prompt = f"""
    Answer the user's question using ONLY the context provided below.
    If the answer is not in the context, say "I do not know."
    
    Context: {context}
    
    Question: {user_query}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Test the RAG loop
print(generate_rag_response("How long does it take to get my money back?"))
```