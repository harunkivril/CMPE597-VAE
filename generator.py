from model import VAE
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms import Resize

config = {
    "zdim":64,
    "image_size":28,
    "bidirect":True,
    "fc_out_size":64,
    "channels":(64, 32, 16, 1),
    "kernel_sizes":(4,4,4,4),
    "pads":(0,0,1,1),
    "strides":(1,1,2,2),
    "max_epoch":15,
    "early_stop_steps":10,
    "batch_size":64,
    "lr":1e-3,
}

model = VAE(**config)
model.load_state_dict(torch.load("./best_model.pkl"))

resizer = Resize(800)
latent = torch.Tensor(np.ones((100, 64)))
latent = torch.randn_like(latent)
decoded = model.decode(latent)
images =resizer(make_grid(decoded, nrow=10)).permute(1, 2, 0).numpy()
plt.figure(figsize=(10,10))
plt.imsave("./generated_images.png", images)
print("Generated images saved to ./generated_images.png")
