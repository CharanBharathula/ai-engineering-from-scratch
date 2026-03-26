# Phase 05: NLP Foundations to Advanced

Natural Language Processing (NLP) is how AI understands text. This phase covers the evolution of NLP from simple word counts to the powerful Transformer architecture.

## Roadmap

| Lesson | Description | Status |
|--------|-------------|--------|
| 01. Text Preprocessing | Tokenization, Stemming, and Lemmatization (NLTK/SpaCy). | ✅ |
| 02. Word Embeddings | Word2Vec, GloVe, and TF-IDF. | ⬚ |
| 03. RNNs & LSTMs | Handling sequential data and memory. | ⬚ |
| 04. Seq2Seq Models | Encoder-Decoder architecture for translation. | ⬚ |
| 05. The Attention Mechanism | The math behind "Attention is All You Need". | ⬚ |

## Code Example: Using Pre-Trained Transformers (HuggingFace)

Before building a Transformer from scratch, it is crucial to know how to use existing ones via the HuggingFace `transformers` library.

```python
# pip install transformers torch
from transformers import pipeline

# 1. Initialize a pre-trained Sentiment Analysis pipeline
# This automatically downloads the model weights and tokenizer
sentiment_analyzer = pipeline("sentiment-analysis")

# 2. Analyze text
texts = [
    "I absolutely loved the new AI Engineering course!",
    "The code was broken and the documentation was terrible.",
    "The weather today is cloudy with a chance of rain."
]

results = sentiment_analyzer(texts)

# 3. Print Results
for text, result in zip(texts, results):
    print(f"Text: '{text}'")
    print(f"Sentiment: {result['label']} (Confidence: {result['score']:.4f})\n")
```