# Lesson 01: Prompt Engineering & System Design

Prompt engineering for production systems is vastly different from chatting with ChatGPT. In code, you must design prompts that are robust, predictable, and resilient against injection attacks.

## Core Concepts

1. **System Prompts vs. User Prompts**
   - **System Prompt:** Sets the persona, boundaries, and rigid rules. The LLM prioritizes this.
   - **User Prompt:** The variable data provided by the user.

2. **Few-Shot Prompting**
   - Providing 2-3 examples of desired input/output pairs in the prompt before asking the actual question. This drastically reduces formatting hallucinations.

3. **Chain of Thought (CoT)**
   - Forcing the model to output its reasoning step-by-step before giving the final answer. E.g., adding "Let's think step by step" to the prompt.

## Code Example: OpenAI Python SDK

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_entities(user_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": "You are a data extraction bot. Extract the name and age from the text. Respond ONLY in the format: Name | Age. Do not add conversational filler."
            },
            {
                "role": "user", 
                "content": "My name is John and I just turned 30 last week."
            },
            {
                "role": "assistant", 
                "content": "John | 30" # Few-shot example
            },
            {
                "role": "user", 
                "content": user_text
            }
        ],
        temperature=0.0 # Strict, predictable output
    )
    return response.choices[0].message.content

print(extract_entities("Hi, I'm Sarah. I am 25 years old."))
# Expected Output: Sarah | 25
```