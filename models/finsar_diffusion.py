"""
FinSar Diffusion Model - Quantum-Inspired Financial Path Exploration

Implements the breakthrough FinSar model using Feynman path integral principles
for financial text diffusion. This model explores multiple financial scenarios
before converging on the most probable outcome.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
import math

from .diffusion_model import DiffusionModel


class FeynmanPathExplorer(nn.Module):
    """
    Quantum-inspired path exploration module implementing Feynman path integral principles.
    
    This module explores multiple financial scenarios (paths) and uses quantum-inspired
    interference patterns to select the most probable financial outcome.
    """
    
    def __init__(self, embedding_dim: int, num_paths: int = 50, path_dim: int = 256):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_paths = num_paths
        self.path_dim = path_dim
        
        # Path generation network
        self.path_generator = nn.Sequential(
            nn.Linear(embedding_dim, path_dim),
            nn.ReLU(),
            nn.Linear(path_dim, path_dim),
            nn.ReLU(),
            nn.Linear(path_dim, embedding_dim * num_paths)
        )
        
        # Amplitude calculation (quantum-inspired)
        self.amplitude_calculator = nn.Sequential(
            nn.Linear(embedding_dim, path_dim // 2),
            nn.Tanh(),
            nn.Linear(path_dim // 2, 2)  # Real and imaginary parts
        )
        
        # Path interference network
        self.interference_net = nn.Sequential(
            nn.Linear(embedding_dim * num_paths, path_dim),
            nn.ReLU(),
            nn.Linear(path_dim, num_paths),
            nn.Softmax(dim=-1)
        )
        
        # Path convergence network
        self.convergence_net = nn.Sequential(
            nn.Linear(embedding_dim * num_paths, path_dim),
            nn.ReLU(),
            nn.Linear(path_dim, embedding_dim)
        )
    
    def generate_paths(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate multiple financial scenario paths from a query embedding.
        
        Args:
            query_embedding: Input financial query [batch_size, embedding_dim]
            
        Returns:
            Generated paths [batch_size, num_paths, embedding_dim]
        """
        batch_size = query_embedding.shape[0]
        
        # Generate raw paths
        raw_paths = self.path_generator(query_embedding)
        paths = raw_paths.view(batch_size, self.num_paths, self.embedding_dim)
        
        # Add noise for exploration diversity
        noise = torch.randn_like(paths) * 0.1
        paths = paths + noise
        
        # Normalize paths
        paths = F.normalize(paths, dim=-1)
        
        return paths
    
    def calculate_path_amplitudes(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Calculate quantum-inspired amplitudes for each path.
        
        Args:
            paths: Financial scenario paths [batch_size, num_paths, embedding_dim]
            
        Returns:
            Complex amplitudes [batch_size, num_paths, 2] (real, imaginary)
        """
        batch_size, num_paths, _ = paths.shape
        
        # Calculate amplitudes for each path
        amplitudes = []
        for i in range(num_paths):
            path_amplitude = self.amplitude_calculator(paths[:, i, :])
            amplitudes.append(path_amplitude)
        
        amplitudes = torch.stack(amplitudes, dim=1)  # [batch_size, num_paths, 2]
        
        return amplitudes
    
    def apply_quantum_interference(self, paths: torch.Tensor, 
                                 amplitudes: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum interference between paths to determine probabilities.
        
        Args:
            paths: Financial scenario paths [batch_size, num_paths, embedding_dim]
            amplitudes: Complex amplitudes [batch_size, num_paths, 2]
            
        Returns:
            Path probabilities [batch_size, num_paths]
        """
        batch_size, num_paths, _ = paths.shape
        
        # Calculate probability from complex amplitudes |ψ|²
        real_parts = amplitudes[:, :, 0]
        imag_parts = amplitudes[:, :, 1]
        probabilities = real_parts**2 + imag_parts**2
        
        # Apply interference effects using path similarity
        flat_paths = paths.view(batch_size, -1)
        interference_weights = self.interference_net(flat_paths)
        
        # Combine quantum probabilities with interference
        final_probabilities = probabilities * interference_weights
        
        # Normalize to valid probability distribution
        final_probabilities = F.softmax(final_probabilities, dim=-1)
        
        return final_probabilities
    
    def select_convergent_path(self, paths: torch.Tensor, 
                             probabilities: torch.Tensor) -> torch.Tensor:
        """
        Select the most probable path using quantum convergence.
        
        Args:
            paths: Financial scenario paths [batch_size, num_paths, embedding_dim]
            probabilities: Path probabilities [batch_size, num_paths]
            
        Returns:
            Convergent financial scenario [batch_size, embedding_dim]
        """
        batch_size, num_paths, embedding_dim = paths.shape
        
        # Weighted combination of paths based on probabilities
        weighted_paths = paths * probabilities.unsqueeze(-1)
        path_sum = weighted_paths.sum(dim=1)
        
        # Apply convergence network for final refinement
        flat_paths = paths.view(batch_size, -1)
        convergent_path = self.convergence_net(flat_paths)
        
        # Blend weighted sum with convergence network output
        final_path = 0.7 * path_sum + 0.3 * convergent_path
        
        return F.normalize(final_path, dim=-1)
    
    def forward(self, query_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete FinSar path exploration process.
        
        Args:
            query_embedding: Input financial query [batch_size, embedding_dim]
            
        Returns:
            Dictionary containing paths, probabilities, and final outcome
        """
        # Generate multiple financial scenario paths
        paths = self.generate_paths(query_embedding)
        
        # Calculate quantum-inspired amplitudes
        amplitudes = self.calculate_path_amplitudes(paths)
        
        # Apply quantum interference
        probabilities = self.apply_quantum_interference(paths, amplitudes)
        
        # Select convergent path
        final_outcome = self.select_convergent_path(paths, probabilities)
        
        return {
            'paths': paths,
            'amplitudes': amplitudes,
            'probabilities': probabilities,
            'final_outcome': final_outcome,
            'most_probable_path_idx': torch.argmax(probabilities, dim=-1)
        }


class FinSarDiffusion(DiffusionModel):
    """
    FinSar Diffusion Model combining quantum path exploration with diffusion processes.
    
    This breakthrough model implements Feynman path integral principles for financial
    text processing, exploring multiple scenarios before converging on optimal outcomes.
    """
    
    def __init__(self, embedding_dim: int = 384, num_steps: int = 100, 
                 hidden_dim: int = 512, num_paths: int = 50):
        super().__init__(embedding_dim, num_steps, hidden_dim)
        
        self.num_paths = num_paths
        
        # Quantum path explorer
        self.path_explorer = FeynmanPathExplorer(
            embedding_dim=embedding_dim,
            num_paths=num_paths,
            path_dim=hidden_dim // 2
        )
        
        # Enhanced noise predictor with path conditioning
        self.path_conditioned_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2 + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Financial volatility predictor
        self.volatility_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward_with_paths(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with quantum path exploration.
        
        Args:
            x: Noisy embeddings [batch_size, embedding_dim]
            t: Timesteps [batch_size]
            
        Returns:
            Dictionary with predicted noise and path information
        """
        # Standard diffusion prediction
        standard_noise = super().forward(x, t)
        
        # Path exploration
        path_results = self.path_explorer(x)
        final_outcome = path_results['final_outcome']
        
        # Time embedding
        t_normalized = t.float() / self.num_steps
        time_emb = self.time_embedding(t_normalized.unsqueeze(-1))
        
        # Path-conditioned prediction
        x_with_path_and_time = torch.cat([x, final_outcome, time_emb], dim=-1)
        path_conditioned_noise = self.path_conditioned_predictor(x_with_path_and_time)
        
        # Predict financial volatility
        volatility = self.volatility_predictor(x)
        
        # Blend predictions based on volatility
        final_noise = (1 - volatility) * standard_noise + volatility * path_conditioned_noise
        
        return {
            'predicted_noise': final_noise,
            'standard_noise': standard_noise,
            'path_conditioned_noise': path_conditioned_noise,
            'volatility': volatility,
            'path_results': path_results
        }
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for training compatibility.
        
        Args:
            x: Noisy embeddings [batch_size, embedding_dim]
            t: Timesteps [batch_size]
            
        Returns:
            Predicted noise [batch_size, embedding_dim]
        """
        results = self.forward_with_paths(x, t)
        return results['predicted_noise']
    
    def explore_financial_scenarios(self, query_embedding: torch.Tensor, 
                                  num_scenarios: int = 10) -> Dict[str, any]:
        """
        Explore multiple financial scenarios for a given query.
        
        Args:
            query_embedding: Financial query embedding [1, embedding_dim]
            num_scenarios: Number of scenarios to explore
            
        Returns:
            Dictionary with scenario analysis
        """
        with torch.no_grad():
            # Run path exploration multiple times for scenario diversity
            scenarios = []
            all_probabilities = []
            
            for _ in range(num_scenarios):
                path_results = self.path_explorer(query_embedding)
                scenarios.append(path_results['final_outcome'])
                all_probabilities.append(path_results['probabilities'])
            
            scenarios = torch.stack(scenarios, dim=1).squeeze(0)  # [num_scenarios, embedding_dim]
            probabilities = torch.stack(all_probabilities, dim=1).squeeze(0)  # [num_scenarios, num_paths]
            
            # Calculate scenario statistics
            scenario_variance = torch.var(scenarios, dim=0).mean().item()
            avg_max_probability = probabilities.max(dim=-1)[0].mean().item()
            
            return {
                'scenarios': scenarios,
                'probabilities': probabilities,
                'scenario_variance': scenario_variance,
                'confidence': avg_max_probability,
                'num_scenarios': num_scenarios
            }
    
    def compute_path_integral_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute path integral loss for FinSar training.
        
        Args:
            x: Clean embeddings [batch_size, embedding_dim]
            t: Random timesteps [batch_size]
            
        Returns:
            Combined loss including path exploration
        """
        # Standard diffusion loss
        noisy_x, noise = self.add_noise(x, t)
        results = self.forward_with_paths(noisy_x, t)
        
        # Main diffusion loss
        diffusion_loss = F.mse_loss(results['predicted_noise'], noise)
        
        # Path exploration loss (encourage diverse but coherent paths)
        path_results = results['path_results']
        paths = path_results['paths']
        probabilities = path_results['probabilities']
        
        # Diversity loss - encourage exploration of different paths
        path_variance = torch.var(paths, dim=1).mean()
        diversity_loss = torch.exp(-path_variance)  # Penalize low diversity
        
        # Concentration loss - encourage probability concentration
        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=-1).mean()
        concentration_loss = entropy / math.log(self.num_paths)  # Normalized entropy
        
        # Volatility consistency loss
        volatility = results['volatility']
        volatility_loss = F.mse_loss(volatility, torch.mean(volatility).expand_as(volatility))
        
        # Combined loss
        total_loss = (diffusion_loss + 
                     0.1 * diversity_loss + 
                     0.1 * concentration_loss + 
                     0.05 * volatility_loss)
        
        return total_loss
    
    def get_model_info(self) -> dict:
        """
        Get comprehensive information about the FinSar model.
        
        Returns:
            Dictionary with model information
        """
        base_info = super().get_model_info()
        
        finsar_info = {
            'model_type': 'FinSar Quantum Diffusion',
            'num_paths': self.num_paths,
            'quantum_features': [
                'Feynman Path Exploration',
                'Quantum Amplitude Calculation',
                'Path Interference Patterns',
                'Financial Volatility Modeling'
            ],
            'capabilities': [
                'Multi-scenario Financial Analysis',
                'Quantum-inspired Uncertainty Quantification',
                'Path-conditioned Text Generation',
                'Financial Volatility Prediction'
            ]
        }
        
        return {**base_info, **finsar_info}