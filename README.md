# Okkhor-Diffusion
Code related to the paper **Okkhor-Diffusion: Class Guided Generation of Bangla Isolated Handwritten Characters using Denoising Diffusion Probabilistic Model (DDPM)**.

Pretrained models are uploaded to [huggingface](https://huggingface.co/ahmedfaiyaz/OkkhorDiffusion).
There are three pretrained models:<br>
[Okkhor-Diffusion-Banglalekha](https://huggingface.co/ahmedfaiyaz/OkkhorDiffusion)<br>
[Okkhor-Diffusion-CmaterDB]()<br>
[Okkhor-Diffusion-Ekushey]()<br>

# Inference

```py
from diffusers import DiffusionPipeline
from typing import List, Optional, Tuple, Union
import torch
pipeline = DiffusionPipeline.from_pretrained("ahmedfaiyaz/OkkhorDiffusion",custom_pipeline="ahmedfaiyaz/OkkhorDiffusion",embedding=torch.float16)
pipeline.to("cuda")
pipeline.embedding=torch.tensor([10-1]) # 'ও': 9
pipeline(batch_size=1,num_inference_steps=1000).images[0]

```

