# Phase 12: Multimodal AI
Models that can see, hear, and read simultaneously.

## Roadmap
| Lesson | Description | Status |
|--------|-------------|--------|
| 01. Contrastive Learning | CLIP architecture. | ✅ |
| 02. Vision-Language Models | BLIP, LLaVA. | ⬚ |
| 03. Audio-Language Models | SeamlessM4T. | ⬚ |

## Code Example: HuggingFace CLIP
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Compare an image against text labels
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image 
probs = logits_per_image.softmax(dim=1)
print(probs)
```
