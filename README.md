# Okkhor-Diffusion
Code related to the paper **Okkhor-Diffusion: Class Guided Generation of Bangla Isolated Handwritten Characters using Denoising Diffusion Probabilistic Model (DDPM)**.

Pretrained models are uploaded to [huggingface](https://huggingface.co/gr33nr1ng3r/OkkhorDiffusion).
There are three pretrained models:<br>
[Okkhor-Diffusion-Banglalekha-64x64]()<br>
[Okkhor-Diffusion-CmaterDB-64x64]()<br>
[Okkhor-Diffusion-Ekushey-64x64]()<br>

# Inference

```py
from diffusers import DiffusionPipeline
from typing import List, Optional, Tuple, Union
import torch
pipeline = DiffusionPipeline.from_pretrained("gr33nr1ng3r/OkkhorDiffusion",custom_pipeline="gr33nr1ng3r/OkkhorDiffusion",embedding=torch.float16)
pipeline.to("cuda")
pipeline.embedding=torch.tensor([10-1]) # 'ও': 9
pipeline(batch_size=1,num_inference_steps=1000).images[0]

```

