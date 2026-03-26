# Lesson 04: Embeddings and Vector Search

Embeddings are the backbone of modern AI memory. LLMs do not read words like humans do; they calculate the distance between mathematical vectors.

## What is an Embedding?
An embedding is a list of floating-point numbers (a vector) that represents the semantic meaning of a piece of text. For example, OpenAI's `text-embedding-3-small` model converts any text into an array of 1536 numbers.

Because they represent *meaning*, sentences with similar meanings will have vectors that are mathematically close to each other, even if they share zero identical words (e.g., "The cat chased the mouse" and "A feline hunted a rodent").

## Calculating Cosine Similarity
To find related text, we calculate the Cosine Similarity between two vectors. A score of 1.0 means identical meaning, 0.0 means orthogonal (unrelated), and -1.0 means opposite meaning.

## Code Example: Generating & Comparing Embeddings

```python
import numpy as np
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def cosine_similarity(v1, v2):
    # Dot product of the vectors divided by their magnitudes
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 1. Get vectors for different phrases
vec_dog = get_embedding("I love my golden retriever")
vec_cat = get_embedding("My feline friend is cute")
vec_car = get_embedding("The engine is making a weird noise")

# 2. Compare the vectors
print("Dog vs Cat:", cosine_similarity(vec_dog, vec_cat))  # High similarity (Both pets)
print("Dog vs Car:", cosine_similarity(vec_dog, vec_car))  # Low similarity
```