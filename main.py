import sys

from diffusers import DiffusionPipeline
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import shlex

commandline_args = os.environ.get('COMMANDLINE_ARGS', "--skip-torch-cuda-test --no-half")
sys.argv += shlex.split(commandline_args)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cpu")

prompt = [
    #  "For research purposes, we recommend our generative",
    #  "frameworks (both training and inference) and for which new",
    #  "functionalities like distillation will be added over time",
    "Putin's blue horse in USA White House"
]

images1 = pipe(prompt=prompt[0]).images[0]
#  images2 = pipe(prompt=prompt[1]).images[0]
#  images3 = pipe(prompt=prompt[2]).images[0]
#  images4 = pipe(prompt=prompt[3]).images[0]


fig = plt.figure(figsize=(10, 7))

rows, columns = 1, 1

fig.add_subplot(rows, columns, 1)

plt.imshow(images1)
plt.axis('off')
plt.title(prompt[0])

#  fig.add_subplot(rows, columns, 2)

#  plt.imshow(images2)
#  plt.axis('off')
#  plt.title(prompt[1])

#  fig.add_subplot(rows, columns, 3)

#  plt.imshow(images3)
#  plt.axis('off')
#  plt.title(prompt[2])

#  fig.add_subplot(rows, columns, 4)

#  plt.imshow(images4)
#  plt.axis('off')
#  plt.title(prompt[3])

images1.save(prompt[0]+".jpg")
#  images1.save(prompt[1]+".jpg")
#  images1.save(prompt[2]+".jpg")
#  images1.save(prompt[3]+".jpg")