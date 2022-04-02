import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import sampler
from torchvision.utils import make_grid
from tqdm import tqdm

from model import VAE

config = {
    "zdim":64,
    "image_size":28,
    "bidirect":True,
    "fc_out_size":64,
    "channels":(64, 32, 16, 1),
    "kernel_sizes":(4,4,4,4),
    "pads":(0,0,1,1),
    "strides":(1,1,2,2),
    "max_epoch":50,
    "early_stop_steps":10,
    "batch_size":64,
    "lr":1e-3
}

save_folder = "./models/deneme/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_EPOCH = config["max_epoch"]
early_stop_steps = config["early_stop_steps"]
batch_size= config["batch_size"]
lr = config["lr"]


os.makedirs(save_folder, exist_ok=True)
with open(save_folder + "config.json", "w") as file:
    json.dump(config, file)

transformations = transforms.Compose([transforms.ToTensor()])
resizer = transforms.Resize(800)

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transformations)

valset = torchvision.datasets.MNIST(root='./data', train=True,
                                         download=True, transform=transformations)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transformations)

split_ratio = 0.1
seed = 3136
n_obs = len(trainset)
indices = list(range(n_obs))
split_idx = int(np.floor(split_ratio * n_obs))
np.random.seed(seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split_idx:], indices[:split_idx]
train_sampler = sampler.SubsetRandomSampler(train_idx)
valid_sampler = sampler.SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=valid_sampler)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


model = VAE(**config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.BCELoss(reduction='sum')

def loss_fn(bce_loss, mu, logvar, gamma=0):
    kldiv = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + (1+gamma)*kldiv

print(model)

val_total_losses = []
val_bce_losses = []
train_total_losses = []
train_bce_losses = []

best_val_loss = 1000
early_stop_count = 0

for epoch in range(MAX_EPOCH):
    train_total_loss = 0
    train_bce_loss = 0
    for images, __ in tqdm(trainloader):
        images = images.to(device)
        images_in = images.squeeze(dim=1)
        out, mu, logvar = model(images_in)
        bce_loss = criterion(out, images)
        loss = loss_fn(bce_loss, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_total_loss += loss.item()
        train_bce_loss += bce_loss.item()

    val_total_loss = 0
    val_bce_loss = 0
    print(f"Validating epoch: {epoch}")
    with torch.no_grad():
        for val_images, __ in tqdm(valloader):
            val_images = val_images.to(device)
            val_images_in = val_images.squeeze(dim=1)
            val_out, val_mu, val_logvar = model(val_images_in)
            val_bce = criterion(val_out, val_images)
            vloss = loss_fn(val_bce, val_mu, val_logvar)
            val_total_loss += vloss.item()
            val_bce_loss += val_bce.item()

    images = resizer(make_grid(val_out)).permute(1, 2, 0).cpu().numpy()
    plt.imsave(save_folder + f"epoch{epoch}.png", images)

    train_total_loss = train_total_loss/(len(trainloader)*batch_size)
    train_total_losses.append(train_total_loss)

    train_bce_loss = train_bce_loss/(len(trainloader)*batch_size)
    train_bce_losses.append(train_bce_loss)

    val_total_loss = val_total_loss/(len(valloader)*batch_size)
    val_total_losses.append(val_total_loss)

    val_bce_loss = val_bce_loss/(len(valloader)*batch_size)
    val_bce_losses.append(val_bce_loss)

    if val_total_loss < best_val_loss:
        early_stop_count=0
        best_val_loss = val_total_loss
        torch.save(model.state_dict(), save_folder + "best_model.pkl")
    else:
        early_stop_count+=1

    print(f"""
    Epoch {epoch}: Train Total Loss: {train_total_loss}, Val Total Loss: {val_total_loss}
    Epoch {epoch}: Train BCE Loss: {train_bce_loss}, Val BCE Loss: {val_bce_loss}
    """)

train_kl_losses = np.array(train_total_losses) - train_bce_losses
val_kl_losses = np.array(val_total_losses) - val_bce_losses

plt.style.use("ggplot")
plt.figure(figsize=(12,8))
plt.plot(list(range(len(train_total_losses))), train_total_losses, label='train_total_loss')
plt.plot(list(range(len(val_total_losses))), val_total_losses, label='val_total_loss')
plt.plot(list(range(len(train_bce_losses))), train_bce_losses, label='train_bce_loss')
plt.plot(list(range(len(val_bce_losses))), val_bce_losses, label='val_bce_loss')
plt.plot(list(range(len(train_kl_losses))), train_kl_losses, label='train_kl_loss')
plt.plot(list(range(len(val_kl_losses))), val_kl_losses, label='val_kl_loss')
plt.xlabel("# of Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{save_folder}/loss_plot.png")

test_total_loss = 0
test_bce_loss = 0
for images, __ in tqdm(testloader):
    images = images.to(device)
    images_in = images.squeeze(dim=1)
    out, mu, logvar = model(images_in)
    bce_loss = criterion(out, images)
    loss = loss_fn(bce_loss, mu, logvar)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    test_total_loss += loss.item()
    test_bce_loss += bce_loss.item()

test_results ={}
test_results["total_loss"] = test_total_loss/(len(testloader)*batch_size)
test_results["bce_loss"] = test_bce_loss/(len(testloader)*batch_size)
test_results["kl_loss"] = test_results["total_loss"] - test_results["bce_loss"]

images = resizer(make_grid(out)).permute(1, 2, 0).cpu().numpy()
plt.imsave(save_folder + f"test_sample.png", images)

with open(f"{save_folder}/test_results.json", "w") as file:
    json.dump(test_results, file)
