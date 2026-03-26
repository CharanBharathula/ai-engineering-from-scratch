# Phase 07: Transformers Deep Dive
This phase breaks down the "Attention Is All You Need" paper and the architecture that revolutionized AI.

## Roadmap
| Lesson | Description | Status |
|--------|-------------|--------|
| 01. Self-Attention | The math behind $Softmax(QK^T/\sqrt{d})V$. | ✅ |
| 02. Multi-Head Attention | Parallel attention mechanisms. | ⬚ |
| 03. Positional Encoding | Giving sequence order to the model. | ⬚ |
| 04. Encoder vs Decoder | BERT vs GPT architectures. | ⬚ |

## Code Example: Self-Attention in PyTorch
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.values = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        K = self.keys(x)
        Q = self.queries(x)
        V = self.values(x)
        
        # Q * K^T / sqrt(d)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.embed_size)
        attention = F.softmax(attention_scores, dim=-1)
        
        out = torch.matmul(attention, V)
        return out
```
