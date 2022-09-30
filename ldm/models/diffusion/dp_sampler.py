'''
Common ancestor of both the DDIM and PLM samplers.
It basically implements the make_schedule() method
that they both use.
'''

import torch
import numpy as np
from ldm.models.diffusion.sampler import Sampler
from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor,
)

class DPSampler(Sampler):

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device(self.device):
                attr = attr.to(dtype=torch.float32, device=self.device)
        setattr(self, name, attr)

    def make_schedule(
            self,
            ddim_num_steps,
            ddim_discretize='uniform',
            ddim_eta=0.0,
            verbose=False,
    ):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), 'alphas have to be defined for each timestep'
        to_torch = (
            lambda x: x.clone()
            .detach()
            .to(torch.float32)
            .to(self.model.device)
        )

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer(
            'alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu()))
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod',
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            'log_one_minus_alphas_cumprod',
            to_torch(np.log(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            'sqrt_recip_alphas_cumprod',
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())),
        )
        self.register_buffer(
            'sqrt_recipm1_alphas_cumprod',
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        # ddim sampling parameters
        (
            ddim_sigmas,
            ddim_alphas,
            ddim_alphas_prev,
        ) = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer(
            'ddim_sqrt_one_minus_alphas', np.sqrt(1.0 - ddim_alphas)
        )
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            'ddim_sigmas_for_original_num_steps',
            sigmas_for_original_sampling_steps,
        )

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape)
            * noise
        )
