# Phase 08: Generative AI
Moving beyond classification to generation: creating images, text, and data from noise.

## Roadmap
| Lesson | Description | Status |
|--------|-------------|--------|
| 01. Autoencoders (AE & VAE) | Latent space representations. | ✅ |
| 02. GANs | Generative Adversarial Networks. | ⬚ |
| 03. Diffusion Models | DDPM, Stable Diffusion fundamentals. | ⬚ |

## Code Example: Simple GAN Generator
```python
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, img_dim),
            nn.Tanh() # Output values between -1 and 1
        )
        
    def forward(self, x):
        return self.gen(x)
```
