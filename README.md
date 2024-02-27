# Okkhor-Diffusion
Code related to the paper **Okkhor-Diffusion: Class Guided Generation of Bangla Isolated Handwritten Characters using Denoising Diffusion Probabilistic Model (DDPM)**.

Pretrained models are uploaded to [huggingface](https://huggingface.co/models?other=diffusers%3AOkkhorDiffusionPipeline).

| Pretrained Model                   | Hugging Face Link                                      |                          API and web frontend          |
| ---------------------------------- | ------------------------------------------------------- |-------------------------------------------------------|
| Okkhor-Diffusion-Banglalekha       | [ahmedfaiyaz/OkkhorDiffusion](https://huggingface.co/ahmedfaiyaz/OkkhorDiffusion) | [OkkhorDiffusion-Space](https://huggingface.co/spaces/ahmedfaiyaz/OkkhorDiffusion) |
| Okkhor-Diffusion-CMATERdb          | [ahmedfaiyaz/OkkhorDiffusion-CMATERdb](https://huggingface.co/ahmedfaiyaz/OkkhorDiffusion-CMATERdb)                                |        
| Okkhor-Diffusion-Ekush             | [ahmedfaiyaz/OkkhorDiffusion-Ekush](https://huggingface.co/ahmedfaiyaz/OkkhorDiffusion-Ekush)                                   |


# Inference

```py
from diffusers import DiffusionPipeline
from typing import List, Optional, Tuple, Union
import torch
pipeline = DiffusionPipeline.from_pretrained("ahmedfaiyaz/OkkhorDiffusion",custom_pipeline="ahmedfaiyaz/OkkhorDiffusion",embedding=torch.float16)
pipeline.to("cuda")
pipeline.embedding=torch.tensor([10-1]) # 'ও': 9
pipeline(batch_size=1,num_inference_steps=1000).images[0]
# Citation
```
@ARTICLE{10445466,

  author={Fuad, Md Mubtasim and Faiyaz, A. and Arnob, Noor Mairukh Khan and Mridha, M.F. and Saha, Aloke Kumar and Aung, Zeyar},

  journal={IEEE Access}, 

  title={Okkhor-Diffusion: Class Guided Generation of Bangla Isolated Handwritten Characters using Denoising Diffusion Probabilistic Model (DDPM)}, 

  year={2024},

  volume={},

  number={},

  pages={1-1},

  keywords={Deep learning;Handwritten character generation;Generative Model;Denoising Diffusion Probabilistic Model},

  doi={10.1109/ACCESS.2024.3370674}}
```

