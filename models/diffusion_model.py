"""
Diffusion Model Implementation for Financial Text Refinement.

This module implements a diffusion-based model that operates in the embedding space
to refine financial text by gradually removing noise through a learned denoising process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class DiffusionModel(nn.Module):
    """
    A diffusion model for text refinement in embedding space.
    
    The model learns to predict and remove noise from text embeddings,
    effectively refining the semantic content of financial documents.
    """
    
    def __init__(self, embedding_dim: int = 384, num_steps: int = 100, hidden_dim: int = 512):
        """
        Initialize the diffusion model.
        
        Args:
            embedding_dim: Dimension of the input embeddings (default: 384 for all-MiniLM-L6-v2)
            num_steps: Number of diffusion steps for the noise schedule
            hidden_dim: Hidden dimension of the neural network
        """
        super(DiffusionModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        
        # Time embedding layer to encode the timestep
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Main noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Initialize noise schedule
        self.register_buffer('betas', self._cosine_beta_schedule(num_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - self.alphas_cumprod))
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Create a cosine beta schedule for the diffusion process.
        
        Args:
            timesteps: Number of diffusion steps
            s: Small offset to prevent beta from being too small
            
        Returns:
            Beta schedule tensor
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model.
        
        Args:
            x: Noisy embeddings [batch_size, embedding_dim]
            t: Timesteps [batch_size]
            
        Returns:
            Predicted noise [batch_size, embedding_dim]
        """
        # Normalize timesteps to [0, 1]
        t_normalized = t.float() / self.num_steps
        
        # Create time embeddings
        time_emb = self.time_embedding(t_normalized.unsqueeze(-1))
        
        # Concatenate input with time embedding
        x_with_time = torch.cat([x, time_emb], dim=-1)
        
        # Predict noise
        predicted_noise = self.noise_predictor(x_with_time)
        
        return predicted_noise
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to the input embeddings according to the noise schedule.
        
        Args:
            x: Original embeddings [batch_size, embedding_dim]
            t: Timesteps [batch_size]
            
        Returns:
            Tuple of (noisy_embeddings, noise)
        """
        noise = torch.randn_like(x)
        
        # Get noise schedule values for timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        
        # Add noise according to the schedule
        noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noisy_x, noise
    
    def denoise_step(self, noisy_x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Perform a single denoising step.
        
        Args:
            noisy_x: Noisy embeddings [batch_size, embedding_dim]
            t: Current timestep [batch_size]
            
        Returns:
            Slightly denoised embeddings
        """
        with torch.no_grad():
            # Predict noise
            predicted_noise = self.forward(noisy_x, t)
            
            # Get schedule values
            alpha_t = self.alphas[t].unsqueeze(-1)
            alpha_cumprod_t = self.alphas_cumprod[t].unsqueeze(-1)
            beta_t = self.betas[t].unsqueeze(-1)
            
            # Calculate denoised prediction
            pred_original = (noisy_x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            # Calculate denoised step
            if t[0] > 0:
                alpha_cumprod_prev = self.alphas_cumprod[t - 1].unsqueeze(-1)
                noise = torch.randn_like(noisy_x)
                denoised = torch.sqrt(alpha_cumprod_prev) * pred_original + \
                          torch.sqrt(1 - alpha_cumprod_prev) * noise
            else:
                denoised = pred_original
                
            return denoised
    
    def sample(self, shape: Tuple[int, ...], device: str = 'cpu') -> torch.Tensor:
        """
        Generate samples by running the reverse diffusion process.
        
        Args:
            shape: Shape of the samples to generate
            device: Device to run sampling on
            
        Returns:
            Generated samples
        """
        # Start with pure noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion process
        for i in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.denoise_step(x, t)
            
        return x
    
    def refine_embedding(self, embedding: torch.Tensor, num_inference_steps: int = 50) -> torch.Tensor:
        """
        Refine a given embedding by applying the diffusion process.
        
        Args:
            embedding: Input embedding to refine [batch_size, embedding_dim]
            num_inference_steps: Number of refinement steps
            
        Returns:
            Refined embedding
        """
        device = embedding.device
        batch_size = embedding.shape[0]
        
        # Add some initial noise
        noise_level = 0.3
        noise = torch.randn_like(embedding) * noise_level
        noisy_embedding = embedding + noise
        
        # Perform refinement steps
        step_size = self.num_steps // num_inference_steps
        
        with torch.no_grad():
            for i in range(num_inference_steps):
                t = torch.full((batch_size,), 
                             max(0, self.num_steps - 1 - i * step_size), 
                             device=device, dtype=torch.long)
                
                # Predict and remove noise
                predicted_noise = self.forward(noisy_embedding, t)
                
                # Update embedding
                alpha = 1.0 - (i / num_inference_steps) * 0.5
                noisy_embedding = noisy_embedding - alpha * predicted_noise
        
        return noisy_embedding
    
    def compute_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the training loss for the diffusion model.
        
        Args:
            x: Clean embeddings [batch_size, embedding_dim]
            t: Random timesteps [batch_size]
            
        Returns:
            MSE loss between predicted and actual noise
        """
        # Add noise to embeddings
        noisy_x, noise = self.add_noise(x, t)
        
        # Predict noise
        predicted_noise = self.forward(noisy_x, t)
        
        # Compute MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture and parameters.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'embedding_dim': self.embedding_dim,
            'num_steps': self.num_steps,
            'hidden_dim': self.hidden_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
