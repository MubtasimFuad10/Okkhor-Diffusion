# Okkhor-Diffusion
Code related to the paper **Okkhor-Diffusion: Class Guided Generation of Bangla Isolated Handwritten Characters using Denoising Diffusion Probabilistic Model (DDPM)**.

Pretrained models are uploaded to [huggingface](https://huggingface.co/models?other=diffusers%3AOkkhorDiffusionPipeline).

| Pretrained Model                   | Hugging Face Link                                      |
| ---------------------------------- | ------------------------------------------------------- |
| Okkhor-Diffusion-Banglalekha       | [ahmedfaiyaz/OkkhorDiffusion](https://huggingface.co/ahmedfaiyaz/OkkhorDiffusion) |
| Okkhor-Diffusion-CmaterDB          | Coming soon                                |
| Okkhor-Diffusion-Ekushey           | Coming soon                                   |


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

