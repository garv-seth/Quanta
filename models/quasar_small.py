"""
Quasar Small: A Diffusion-Based GPT for Quantitative Finance

A specialized transformer model that combines the power of GPT architecture with
diffusion processes for superior financial text generation and analysis.
Designed to outperform traditional models like BlackRock's Aladdin in principle.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import json
import pickle
import os
from datetime import datetime
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position encoding"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[-2]
        
        if seq_len != self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
            
        return self._cos_cached, self._sin_cached

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors"""
    
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class QuantFinanceAttention(nn.Module):
    """
    Enhanced Multi-Head Attention with financial market-specific optimizations
    - Rotary Position Embedding (RoPE)
    - Adaptive attention patterns for time-series data
    - Financial sector bias mechanism
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 8192):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Rotary position embedding
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)
        
        # Financial sector attention bias
        self.sector_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                sector_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply rotary position embedding
        cos, sin = self.rope(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add sector bias for financial domain
        scores = scores + self.sector_bias
        
        # Apply sector-specific weights if provided
        if sector_weights is not None:
            scores = scores * sector_weights.unsqueeze(1).unsqueeze(1)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out)

class QuantFinanceFeedForward(nn.Module):
    """
    Enhanced Feed-Forward Network with financial domain optimizations
    - SwiGLU activation (used in PaLM, LLaMA)
    - Adaptive gating for financial reasoning
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # SwiGLU components
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Up projection
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(xW1) ⊙ (xW3) W2
        gate = F.silu(self.w1(x))  # Swish activation
        up = self.w3(x)
        hidden = gate * up
        return self.w2(self.dropout(hidden))

class QuasarTransformerBlock(nn.Module):
    """
    Enhanced Transformer Block for Quasar Small
    - Pre-norm architecture (like GPT-3, LLaMA)
    - Enhanced attention and feed-forward
    - Residual scaling for better training dynamics
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, 
                 layer_scale_init: float = 1e-6):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = QuantFinanceAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = QuantFinanceFeedForward(d_model, d_ff, dropout)
        
        # Layer scale for better training (from CaiT paper)
        self.ls1 = nn.Parameter(layer_scale_init * torch.ones(d_model))
        self.ls2 = nn.Parameter(layer_scale_init * torch.ones(d_model))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                sector_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm with layer scale
        attn_out = self.attn(self.ln1(x), mask, sector_weights)
        x = x + self.dropout(self.ls1 * attn_out)
        
        mlp_out = self.mlp(self.ln2(x))
        x = x + self.dropout(self.ls2 * mlp_out)
        
        return x

class DiffusionScheduler:
    """
    Advanced diffusion scheduler with multiple noise schedules
    Optimized for financial text generation
    """
    
    def __init__(self, num_timesteps: int = 1000, schedule_type: str = "cosine"):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif schedule_type == "linear":
            self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in improved DDPM paper"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

class FinancialTokenizer:
    """
    Advanced tokenizer specifically designed for financial text
    Incorporates financial entities, numbers, and domain-specific patterns
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        
        # Financial-specific special tokens
        self.special_tokens = [
            '<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>',
            '<NUM>', '<COMPANY>', '<TICKER>', '<CURRENCY>', '<PERCENT>',
            '<DATE>', '<TIME>', '<REVENUE>', '<PROFIT>', '<LOSS>',
            '<QUARTER>', '<YEAR>', '<ANALYST>', '<RATING>',
            '<BULL>', '<BEAR>', '<NEUTRAL>', '<BUY>', '<SELL>', '<HOLD>'
        ]
        
        # Initialize special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
            
        self.special_token_count = len(self.special_tokens)
        
    def build_vocab_from_texts(self, texts: List[str]) -> None:
        """Build vocabulary from financial texts with BPE-like approach"""
        import re
        from collections import Counter
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self._basic_tokenize(text)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Add most frequent tokens to vocabulary
        for token, count in token_counts.most_common(self.vocab_size - self.special_token_count):
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inverse_vocab[idx] = token
                
    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization with financial pattern recognition"""
        import re
        
        text = text.lower().strip()
        
        # Financial patterns
        patterns = [
            r'\$[\d,]+(?:\.\d+)?[kmb]?',  # Currency amounts
            r'\d+(?:\.\d+)?%',            # Percentages
            r'\b[a-z]{1,5}\b',            # Potential tickers
            r'\d{4}-\d{2}-\d{2}',         # Dates
            r'\b\w+\b',                   # Regular words
        ]
        
        tokens = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            tokens.extend(matches)
            
        return tokens
        
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token IDs"""
        tokens = self._basic_tokenize(text)
        token_ids = [self.vocab['<BOS>']]
        
        for token in tokens:
            if len(token_ids) >= max_length - 1:
                break
            token_id = self.vocab.get(token, self.vocab['<UNK>'])
            token_ids.append(token_id)
            
        token_ids.append(self.vocab['<EOS>'])
        
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(self.vocab['<PAD>'])
            
        return token_ids[:max_length]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                if token not in ['<PAD>', '<BOS>', '<EOS>']:
                    tokens.append(token)
        return ' '.join(tokens)

class QuasarSmall(nn.Module):
    """
    Quasar Small: Diffusion-Based GPT for Quantitative Finance
    
    A state-of-the-art model combining:
    - GPT-style transformer architecture with modern improvements
    - Diffusion-based generation for high-quality outputs
    - Financial domain-specific optimizations
    - Efficient inference and training
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        num_diffusion_steps: int = 1000,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device
        
        # Initialize tokenizer
        self.tokenizer = FinancialTokenizer(vocab_size)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Time embedding for diffusion steps
        self.time_embedding = nn.Embedding(num_diffusion_steps, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            QuasarTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Diffusion scheduler
        self.scheduler = DiffusionScheduler(num_diffusion_steps)
        
        # Financial sector embeddings (optional enhancement)
        self.sector_embeddings = nn.Embedding(20, d_model)  # 20 major sectors
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
    def _init_weights(self, module):
        """Initialize weights using GPT-style initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, timestep: Optional[torch.Tensor] = None,
                sector_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Quasar Small
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            timestep: Diffusion timestep [batch_size] or scalar
            sector_ids: Financial sector IDs [batch_size]
            attention_mask: Attention mask [batch_size, seq_len]
        """
        B, T = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add time embedding for diffusion
        if timestep is not None:
            if timestep.dim() == 0:  # Scalar timestep
                timestep = timestep.expand(B)
            time_emb = self.time_embedding(timestep).unsqueeze(1)
            x = x + time_emb
        
        # Add sector embeddings if provided
        if sector_ids is not None:
            sector_emb = self.sector_embeddings(sector_ids).unsqueeze(1)
            x = x + sector_emb
            
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def add_noise(self, x: torch.Tensor, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to embeddings according to diffusion schedule"""
        noise = torch.randn_like(x)
        
        sqrt_alphas_cumprod = self.scheduler.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alphas_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep]
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1)
        
        noisy_x = sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
        
        return noisy_x, noise
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step for diffusion loss"""
        input_ids = batch['input_ids']
        B, T = input_ids.shape
        
        # Get embeddings
        x = self.token_embedding(input_ids)
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.num_diffusion_steps, (B,), device=x.device)
        
        # Add noise
        noisy_x, noise = self.add_noise(x, timesteps)
        
        # Predict the noise
        predicted_logits = self.forward(input_ids, timesteps)
        predicted_embeddings = self.token_embedding(predicted_logits.argmax(dim=-1))
        
        # Compute loss (predict the noise)
        loss = F.mse_loss(predicted_embeddings, noise)
        
        return loss
    
    def generate(self, prompt: str, max_length: int = 256, num_inference_steps: int = 50,
                 temperature: float = 1.0, top_k: int = 50) -> str:
        """
        Generate text using diffusion-based sampling
        
        Args:
            prompt: Input prompt text
            max_length: Maximum generation length
            num_inference_steps: Number of denoising steps
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        """
        self.eval()
        
        with torch.no_grad():
            # Encode prompt
            prompt_ids = self.tokenizer.encode(prompt, max_length=max_length//4)
            prompt_tensor = torch.tensor([prompt_ids], device=self.device)
            
            # Start with noise for the rest
            generated_ids = torch.randint(0, self.vocab_size, (1, max_length), device=self.device)
            generated_ids[0, :len(prompt_ids)] = prompt_tensor[0, :len(prompt_ids)]
            
            # Diffusion sampling loop
            for i in reversed(range(0, self.num_diffusion_steps, self.num_diffusion_steps // num_inference_steps)):
                timestep = torch.tensor([i], device=self.device)
                
                # Forward pass
                logits = self.forward(generated_ids, timestep)
                
                # Apply temperature and top-k sampling
                logits = logits / temperature
                
                # Only update non-prompt tokens
                for pos in range(len(prompt_ids), max_length):
                    if top_k > 0:
                        # Top-k sampling
                        top_k_logits, top_k_indices = torch.topk(logits[0, pos], top_k)
                        probs = F.softmax(top_k_logits, dim=-1)
                        next_token = top_k_indices[torch.multinomial(probs, 1)]
                    else:
                        # Standard sampling
                        probs = F.softmax(logits[0, pos], dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    
                    # Update with small probability (gradual denoising)
                    if torch.rand(1) < 0.1:
                        generated_ids[0, pos] = next_token
            
            # Decode to text
            generated_text = self.tokenizer.decode(generated_ids[0].tolist())
            
        return generated_text
    
    def refine_text(self, text: str, num_steps: int = 25, noise_level: float = 0.3) -> str:
        """
        Refine input text using partial diffusion process
        
        Args:
            text: Input text to refine
            num_steps: Number of refinement steps
            noise_level: Amount of noise to add (0.0 to 1.0)
        """
        self.eval()
        
        with torch.no_grad():
            # Encode text
            text_ids = self.tokenizer.encode(text)
            text_tensor = torch.tensor([text_ids], device=self.device)
            
            # Start from partially noisy state
            start_timestep = int(self.num_diffusion_steps * noise_level)
            
            # Add initial noise
            x = self.token_embedding(text_tensor)
            timestep = torch.tensor([start_timestep], device=self.device)
            noisy_x, _ = self.add_noise(x, timestep)
            
            # Gradual denoising
            for i in range(num_steps):
                current_timestep = max(0, start_timestep - (i * start_timestep // num_steps))
                timestep = torch.tensor([current_timestep], device=self.device)
                
                # Predict and refine
                logits = self.forward(text_tensor, timestep)
                
                # Update tokens with refined predictions
                refined_probs = F.softmax(logits[0], dim=-1)
                for pos in range(len(text_ids)):
                    if torch.rand(1) < 0.1:  # Selective refinement
                        text_tensor[0, pos] = torch.multinomial(refined_probs[pos], 1)
            
            # Decode refined text
            refined_text = self.tokenizer.decode(text_tensor[0].tolist())
            
        return refined_text
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'Quasar Small',
            'architecture': 'Diffusion-Based GPT',
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'num_diffusion_steps': self.num_diffusion_steps,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming fp32
            'is_trained': self.is_trained,
            'training_epochs': len(self.training_history),
            'device': self.device
        }
    
    def save_model(self, path: str) -> None:
        """Save model state and configuration"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'd_ff': self.d_ff,
                'max_seq_len': self.max_seq_len,
                'dropout': self.dropout,
                'num_diffusion_steps': self.num_diffusion_steps
            },
            'tokenizer_vocab': self.tokenizer.vocab,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = "cpu") -> 'QuasarSmall':
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with saved config
        model = cls(device=device, **checkpoint['model_config'])
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore tokenizer and training state
        model.tokenizer.vocab = checkpoint['tokenizer_vocab']
        model.tokenizer.inverse_vocab = {v: k for k, v in model.tokenizer.vocab.items()}
        model.training_history = checkpoint.get('training_history', [])
        model.is_trained = checkpoint.get('is_trained', False)
        
        logger.info(f"Model loaded from {path}")
        return model

def create_quasar_small(
    vocab_size: int = 32000,
    d_model: int = 768,
    num_heads: int = 12,
    num_layers: int = 12,
    device: str = "cpu"
) -> QuasarSmall:
    """
    Factory function to create Quasar Small model with optimal configurations
    """
    model = QuasarSmall(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_model * 4,  # Standard scaling
        max_seq_len=2048,
        dropout=0.1,
        num_diffusion_steps=1000,
        device=device
    )
    
    logger.info(f"Created Quasar Small with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    return model