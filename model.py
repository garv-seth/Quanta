"""
Quasar d-SLM - Core Model Architecture
This file defines the main components of the quantum-inspired diffusion model.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict

# Try importing optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class EnvironmentDetector:
    """Automatically detect and optimize for available hardware."""

    @staticmethod
    def get_system_info() -> Dict:
        """Get comprehensive system information."""
        info = {
            'cpu_count': os.cpu_count() or 4,
            'has_cuda': torch.cuda.is_available(),
            'cuda_devices': 0,
            'ram_gb': 8.0,  # Default fallback
            'vram_gb': 0.0,
            'gpu_name': 'N/A'
        }

        if torch.cuda.is_available():
            info['cuda_devices'] = torch.cuda.device_count()
            try:
                props = torch.cuda.get_device_properties(0)
                info['vram_gb'] = props.total_memory / (1024**3)
                info['gpu_name'] = props.name
            except Exception:
                # Fallback if properties can't be fetched
                info['vram_gb'] = 8.0

        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                info['ram_gb'] = mem.total / (1024**3)
            except Exception:
                pass

        return info

    @staticmethod
    def get_optimal_config() -> Dict:
        """Return optimal training parameters based on detected hardware."""
        system_info = EnvironmentDetector.get_system_info()
        
        # Start with a base configuration for low-spec hardware
        config = {
            'batch_size': 4,
            'accumulate_gradients': 8,
            'd_model': 256,
            'num_layers': 4,
            'nhead': 4,
            'max_seq_len': 128,
            'use_mixed_precision': False,
            'num_diffusion_steps': 250,
            'num_paths': 4 # Number of Feynman paths
        }
        
        # Scale up based on available RAM
        if system_info['ram_gb'] >= 60: # High-end system
            config.update({
                'batch_size': 32, 'accumulate_gradients': 2, 'd_model': 768, 
                'num_layers': 8, 'nhead': 12, 'max_seq_len': 512
            })
        elif system_info['ram_gb'] >= 32:
            config.update({
                'batch_size': 16, 'accumulate_gradients': 4, 'd_model': 512, 
                'num_layers': 6, 'nhead': 8, 'max_seq_len': 256
            })
        elif system_info['ram_gb'] >= 16:
            config.update({
                'batch_size': 8, 'accumulate_gradients': 4, 'd_model': 384, 
                'num_layers': 5, 'nhead': 6, 'max_seq_len': 192
            })
        
        # Optimize for GPU
        if system_info['has_cuda']:
            config['use_mixed_precision'] = True
            if system_info['vram_gb'] >= 15: # e.g., 16GB+ VRAM
                config['batch_size'] *= 2
                config['num_diffusion_steps'] = 1000
            elif system_info['vram_gb'] >= 8:
                config['batch_size'] = max(8, config['batch_size']) # Ensure at least 8
                config['num_diffusion_steps'] = 500

        return config


class QuantumInspiredDiffusionModel(nn.Module):
    """
    Quantum-inspired diffusion model implementing Feynman path integral principles.
    Explores multiple denoising paths simultaneously and combines them probabilistically.
    """
    
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, 
                 max_seq_len: int, num_diffusion_steps: int, num_paths: int):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_diffusion_steps = num_diffusion_steps
        self.num_paths = num_paths
        
        # This layer is used outside the forward pass to get initial embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.time_embedding = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Path-specific transformers (parallel path processing)
        self.path_transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                    dropout=0.1, batch_first=True, activation='gelu'
                ),
                num_layers=num_layers
            ) for _ in range(num_paths)
        ])
        
        # Path weighting network (quantum amplitude calculation)
        self.path_weighting_net = nn.Sequential(
            nn.Linear(d_model * num_paths, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_paths),
        )
        
        # Output layer now predicts noise, so it maps back to d_model
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Noise schedule
        betas = self._cosine_beta_schedule(num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _time_steps_embedding(self, t, device):
        """Create embedding for timesteps."""
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embedding(emb)

    def forward(self, x_embeds, t):
        """
        Forward pass with Feynman path integral.
        Accepts noisy embeddings and predicts the noise.
        """
        batch_size, seq_len, _ = x_embeds.shape
        device = x_embeds.device
        
        # Embeddings (time and position)
        time_emb = self._time_steps_embedding(t, device)
        
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.position_embedding(positions)
        
        # Initial input to all paths is the noisy embedding + pos/time info
        x = x_embeds + pos_emb.unsqueeze(0) + time_emb.unsqueeze(1)
        x = self.dropout(x)
        
        # Explore multiple denoising paths simultaneously
        path_outputs = [transformer(x) for transformer in self.path_transformers]
        
        # Extract path features (e.g., from CLS token or mean pooling)
        path_features = [p_out.mean(dim=1) for p_out in path_outputs] # List of [batch_size, d_model]
        
        # Combine path features for weighting
        combined_features = torch.cat(path_features, dim=1) # [batch_size, d_model * num_paths]
        
        # Calculate path weights (quantum amplitudes)
        path_weights = self.path_weighting_net(combined_features)
        path_weights = F.softmax(path_weights, dim=-1) # [batch_size, num_paths]
        
        # Weighted combination of path outputs (the "path integral")
        # Reshape weights for broadcasting: [batch_size, num_paths, 1, 1]
        reshaped_weights = path_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Stack outputs: [num_paths, batch_size, seq_len, d_model]
        stacked_outputs = torch.stack(path_outputs, dim=0)
        
        # Weighted sum: [batch_size, seq_len, d_model]
        final_output = torch.sum(stacked_outputs * reshaped_weights.permute(1, 0, 2, 3), dim=0)
        
        # Final projection to predict noise
        predicted_noise = self.output_projection(final_output)
        
        return predicted_noise

    def add_noise_to_embeddings(self, x_start_embeds, t, noise=None):
        """Add noise to continuous embeddings."""
        if noise is None:
            noise = torch.randn_like(x_start_embeds)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        noisy_embeds = sqrt_alpha_t * x_start_embeds + sqrt_one_minus_alpha_t * noise
        return noisy_embeds
    
    def get_token_embeddings(self, token_ids):
        """Helper to get token embeddings."""
        return self.token_embedding(token_ids) 