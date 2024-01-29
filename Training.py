import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os

from DDPM import *
from ContextUnet_Model import *
from Utils import *


# hyperparameters

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02



# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 64 # 64 hidden dimension feature
n_cfeat = 5 # context vector is of size 5
height = 16 # 16x16 image
save_dir = './weights/'


# Initialise DDPM noise scheduler
ddpm = DDPM(timesteps=timesteps, beta1=beta1, beta2=beta2, height=height)


# training hyperparameters
batch_size = 100
n_epoch = 32
lrate=1e-3

# construct model
nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

# load dataset and construct optimizer
sprites = './Data/sprites_1788_16x16.npy'
labeles = './Data/sprite_labels_nc_1788_16x16.npy'
dataset = CustomDataset(sprites, labeles, transform, null_context=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


# training without context code

# set into train mode
nn_model.train()

for ep in range(n_epoch):
    print(f'epoch {ep}')
    
    # linearly decay learning rate
    optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
    
    pbar = tqdm(dataloader, mininterval=2 )

    for x, _ in pbar:   # x: images

        optim.zero_grad()

        x = x.to(device)
        n = x.shape[0]
        
        # perturb data
        noise = torch.randn_like(x)
        t = ddpm.sample_timesteps(n)
        x_pert = ddpm.noise_image(x, t, noise)

        # use network to recover noise
        pred_noise = nn_model(x_pert, t / timesteps)
        
        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        
        optim.step()

    # save model periodically
    if ep%4==0 or ep == int(n_epoch-1):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
        print('saved model at ' + save_dir + f"model_{ep}.pth")