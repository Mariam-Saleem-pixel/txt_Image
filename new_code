import transformers
from transformers import pipeline
from diffusers import StableDiffusionPipeline # Changed 'StableDiffusionPipline' to 'StableDiffusionPipeline'
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype = torch.float16)
pipe = pipe.to("cuda")

prompt = input("Write Prompt")
image = pipe(prompt).images[0]

image.save("horse.png")
