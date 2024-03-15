# Okkhor-Diffusion
Code related to the paper **Okkhor-Diffusion: Class Guided Generation of Bangla Isolated Handwritten Characters using Denoising Diffusion Probabilistic Model (DDPM)**.

Pretrained models are uploaded to [huggingface](https://huggingface.co/models?other=diffusers%3AOkkhorDiffusionPipeline).

| Pretrained Model                   | Hugging Face Link                                      |                          API and web frontend          |
| ---------------------------------- | ------------------------------------------------------- |-------------------------------------------------------|
| Okkhor-Diffusion-Banglalekha       | [ahmedfaiyaz/OkkhorDiffusion](https://huggingface.co/ahmedfaiyaz/OkkhorDiffusion) | [OkkhorDiffusion-Space](https://huggingface.co/spaces/ahmedfaiyaz/OkkhorDiffusion) |
| Okkhor-Diffusion-CMATERdb          | [ahmedfaiyaz/OkkhorDiffusion-CMATERdb](https://huggingface.co/ahmedfaiyaz/OkkhorDiffusion-CMATERdb)                                |        
| Okkhor-Diffusion-Ekush             | [ahmedfaiyaz/OkkhorDiffusion-Ekush](https://huggingface.co/ahmedfaiyaz/OkkhorDiffusion-Ekush)                                   |


# Generating Samples
You can try out this [Colab Notebook](https://colab.research.google.com/drive/1rXafKwmYOwh5YOJD9EEPn2sDv0faUN6d?usp=sharing).
```py
from diffusers import DiffusionPipeline
import torch
device="cuda"
pipeline = DiffusionPipeline.from_pretrained(
              "ahmedfaiyaz/OkkhorDiffusion",
              custom_pipeline="ahmedfaiyaz/OkkhorDiffusion",
              embedding=torch.int16
            )
pipeline.to(device)
pipeline.embedding=torch.tensor([9],device=device) # 'ও': 9
pipeline(batch_size=1,num_inference_steps=100).images[0]

```
# Run the App
Install requirements
```
pip install -r requirements.txt
```
Run
```
python Okkhor_Diffusion_Gradio_App.py
``` 
This will automatically download all the models and run a web app with api endpoints in the local machine.This app is also hosted here at [Huggingface spaces.](https://huggingface.co/spaces/ahmedfaiyaz/OkkhorDiffusion)
# Citation
```
@ARTICLE{10445466,
  author={Fuad, Md. Mubtasim and Faiyaz, A. and Arnob, Noor Mairukh Khan and Mridha, M. F. and Saha, Aloke Kumar and Aung, Zeyar},
  journal={IEEE Access}, 
  title={Okkhor-Diffusion: Class Guided Generation of Bangla Isolated Handwritten Characters Using Denoising Diffusion Probabilistic Model (DDPM)}, 
  year={2024},
  volume={12},
  number={},
  pages={37521-37539},
  keywords={Data models;Shape measurement;Probabilistic logic;Generative adversarial networks;Character generation;Noise reduction;Mathematical models;Deep learning;Handwriting recognition;Character generation;Deep learning;handwritten character generation;generative model;denoising diffusion probabilistic model},
  doi={10.1109/ACCESS.2024.3370674}}
```

