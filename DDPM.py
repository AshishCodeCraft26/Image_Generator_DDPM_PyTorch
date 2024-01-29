import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class DDPM():
    def __init__(self, timesteps = 500, beta1 = 1e-4, beta2 = 0.02, height=28):

        self.timesteps = timesteps
        self.beta1 = beta1
        self.beta2 = beta2
        self.h = height

        # construct DDPM noise schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumsum(self.alpha.log(), dim=0).exp() 
        self.alpha_hat[0] = 1

    def prepare_noise_schedule(self):
        return (self.beta2 - self.beta1) * torch.linspace(0, 1, self.timesteps + 1, device=device) + self.beta1
    
    def noise_image(self, x, t, noise): #perturbs an image to a specified noise level
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat)[t, None, None, None]
        # sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)[t, None, None, None]
        one_minus_alpha_hat = (1 - self.alpha_hat[t, None, None, None])

        return sqrt_alpha_hat * x + one_minus_alpha_hat * noise
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps + 1, size=(n,)).to(device)
    
    # helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
    def denoise_add_noise(self, x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        
        beta = self.beta[t]
        alpha = self.alpha[t]
        alpha_hat = self.alpha_hat[t]

        noise = torch.sqrt(beta) * z

        mean =  1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) 

        return mean + noise

    # sample using standard algorithm
    def sampling(self, n_sample, model, save_rate=20):
        model.eval()
        with torch.no_grad():
            
            # x_T ~ N(0, 1), sample initial noise
            samples = torch.randn(n_sample, 3, self.h, self.h).to(device) 

            intermediate = []
            for i in range(self.timesteps, 0, -1):
                print(f'sampling timestep {i:3d}', end='\r')

                # reshape time tensor
                t = torch.tensor([i / self.timesteps])[:, None, None, None].to(device)

                # sample some random noise to inject back in. For i = 1, don't add back in noise
                z = torch.randn_like(samples) if i > 1 else 0

                predicted_noise = model(samples, t)    # predict noise e_(x_t,t)

                samples = self.denoise_add_noise(samples, i, predicted_noise, z)

                if i % save_rate ==0 or i==self.timesteps or i<8:
                    intermediate.append(samples.detach().cpu().numpy())

            model.train()
            intermediate = np.stack(intermediate)
            return samples, intermediate


    




        
