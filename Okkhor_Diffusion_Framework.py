from diffusers import DiffusionPipeline
from typing import List, Optional, Tuple, Union
import torch
import gradio as gr
css="""
#input-panel{
align-items:center;
justify-content:center
    
}

"""
modelname="ahmedfaiyaz/OkkhorDiffusion"
pipeline = DiffusionPipeline.from_pretrained(modelname,custom_pipeline="ahmedfaiyaz/OkkhorDiffusion",embedding=torch.float16)
character_mappings = {
    'অ': 1,
    'আ': 2,
    'ই': 3,
    'ঈ': 4,
    'উ': 5,
    'ঊ': 6,
    'ঋ': 7,
    'এ': 8,
    'ঐ': 9,
    'ও': 10,
    'ঔ': 11,
    'ক': 12,
    'খ': 13,
    'গ': 14,
    'ঘ': 15,
    'ঙ': 16,
    'চ': 17,
    'ছ': 18,
    'জ': 19,
    'ঝ': 20,
    'ঞ': 21,
    'ট': 22,
    'ঠ': 23,
    'ড': 24,
    'ঢ': 25,
    'ণ': 26,
    'ত': 27,
    'থ': 28,
    'দ': 29,
    'ধ': 30,
    'ন': 31,
    'প': 32,
    'ফ': 33,
    'ব': 34,
    'ভ': 35,
    'ম': 36,
    'য': 37,
    'র': 38,
    'ল': 39,
    'শ': 40,
    'ষ': 41,
    'স': 42,
    'হ': 43,
    'ড়': 44,
    'ঢ়': 45,
    'য়': 46,
    'ৎ': 47,
    'ং': 48,
    'ঃ': 49,
    'ঁ': 50,
    '০': 51,
    '১': 52,
    '২': 53,
    '৩': 54,
    '৪': 55,
    '৫': 56,
    '৬': 57,
    '৭': 58,
    '৮': 59,
    '৯': 60,
    'ক্ষ(ksa)': 61,
    'ব্দ(bda)': 62,
    'ঙ্গ': 63,
    'স্ক': 64,
    'স্ফ': 65,
    'স্থ': 66,
    'চ্ছ': 67,
    'ক্ত': 68,
    'স্ন': 69,
    'ষ্ণ': 70,
    'ম্প': 71,
    'হ্ম': 72,
    'প্ত': 73,
    'ম্ব': 74,
    'ন্ড': 75,
    'দ্ভ': 76,
    'ত্থ': 77,
    'ষ্ঠ': 78,
    'ল্প': 79,
    'ষ্প': 80,
    'ন্দ': 81,
    'ন্ধ': 82,
    'ম্ম': 83,
    'ন্ঠ': 84,
}

def generate(input_text:str,batch_size:int,inference_steps:int):
    batch_size=int(batch_size)
    inference_steps=int(inference_steps)
    print(f"Generating image with label:{character_mappings[input_text]} batch size:{batch_size}")
    label=int(character_mappings[input_text])
    label-=1
    pipeline.embedding=torch.tensor([label])
    generate_image=pipeline(batch_size=batch_size,num_inference_steps=inference_steps).images
    return generate_image



with gr.Blocks(css=css,elem_id="panel") as od_app:
    with gr.Column(min_width=100):
        text=gr.HTML("""
         <div style="text-align: center; margin: 0 auto;">
              <div style="display: inline-flex;align-items: center;gap: 0.8rem;font-size: 1.75rem;">
                <h1> Okkhor Diffusion </h1>
               </div>
        </div>

""")
    #input panel 
    
    with gr.Row(elem_id="input-panel"):
        with gr.Column(variant="panel",scale=0,elem_id="input-panel-items"):
            dropdown = gr.Dropdown(label="Select Character",choices=list(character_mappings.keys()))
            batch_size = gr.Number(label="Batch Size", minimum=0, maximum=100)
            inference_steps= gr.Slider(label="Steps",value=100,minimum=100,maximum=1000,step=100)
            btn = gr.Button("Generate",size="sm")  
      
        
    gallery = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery"
    , columns=[10], rows=[10], object_fit="contain", height="auto",scale=1,min_width=80)

         
    btn.click(fn=generate,inputs=[dropdown,batch_size,inference_steps],outputs=[gallery])

if __name__=='__main__':
    od_app.queue(max_size=20).launch(show_error=True)





