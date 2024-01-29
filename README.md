<h1>
	The Diffusion Model
</h1>


We'll take a deeper look into **Denoising Diffusion Probabilistic Models** (also known as DDPMs, diffusion models, score-based generative models or simply [autoencoders](https://benanne.github.io/2022/01/31/diffusion.html)) as researchers have been able to achieve remarkable results with them for (un)conditional image/audio/video generation. Popular examples (at the time of writing) include [GLIDE](https://arxiv.org/abs/2112.10741) and [DALL-E 2](https://openai.com/dall-e-2/) by OpenAI, [Latent Diffusion](https://github.com/CompVis/latent-diffusion) by the University of Heidelberg and [ImageGen](https://imagen.research.google/) by Google Brain.

We'll go over the original DDPM paper by ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)), implementing it step-by-step in PyTorch, based on Phil Wang's [implementation](https://github.com/lucidrains/denoising-diffusion-pytorch) - which itself is based on the [original TensorFlow implementation](https://github.com/hojonathanho/diffusion). Note that the idea of diffusion for generative modeling was actually already introduced in ([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)). However, it took until ([Song et al., 2019](https://arxiv.org/abs/1907.05600)) (at Stanford University), and then ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) (at Google Brain) who independently improved the approach.

Note that there are [several perspectives](https://twitter.com/sedielem/status/1530894256168222722?s=20&t=mfv4afx1GcNQU5fZklpACw) on diffusion models. Here, we employ the discrete-time (latent variable model) perspective, but be sure to check out the other perspectives as well.

Diffusion model is based on [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) and [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)


<p align="center">
<img src='https://drive.google.com/uc?id=11C3cBUfz7_vrkj_4CWCyePaQyr-0m85_' width=500>
</p>

