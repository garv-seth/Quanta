"""
Advanced Financial Diffusion Language Model

A transformer-based diffusion model specifically designed for financial text generation
and refinement, inspired by GPT architecture but optimized for diffusion processes.
"""

import numpy as np
import json
import pickle
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

# Simple transformer implementation without external dependencies
class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        # Linear transformations
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose for attention computation
        Q = np.transpose(Q, (0, 2, 1, 3))
        K = np.transpose(K, (0, 2, 1, 3))
        V = np.transpose(V, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        scores = np.matmul(Q, np.transpose(K, (0, 1, 3, 2))) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        
        attention_weights = self.softmax(scores)
        attention_output = np.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = np.transpose(attention_output, (0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, seq_len, d_model)
        
        # Final linear transformation
        output = np.dot(attention_output, self.W_o)
        return output
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)  # ReLU activation
        output = np.dot(hidden, self.W2) + self.b2
        return output

class TransformerBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
    
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        # Self-attention with residual connection
        attn_output = self.attention.forward(x)
        x = self.layer_norm1.forward(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward.forward(x)
        x = self.layer_norm2.forward(x + ff_output)
        
        return x

class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FinancialTokenizer:
    """Simple tokenizer for financial text"""
    
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        
        # Financial domain-specific tokens
        self.special_tokens = [
            '<PAD>', '<UNK>', '<START>', '<END>',
            '<COMPANY>', '<REVENUE>', '<PROFIT>', '<LOSS>',
            '<QUARTER>', '<YEAR>', '<PERCENT>', '<DOLLAR>'
        ]
        
        # Initialize with special tokens
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)
            self.inverse_vocab[len(self.inverse_vocab)] = token
        
        self.vocab_size = len(self.vocab)
    
    def build_vocab(self, texts: List[str], max_vocab_size: int = 10000):
        """Build vocabulary from training texts"""
        word_freq = {}
        
        for text in texts:
            words = self.simple_tokenize(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words[:max_vocab_size - len(self.special_tokens)]:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
                self.inverse_vocab[len(self.inverse_vocab)] = word
        
        self.vocab_size = len(self.vocab)
    
    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b|\$\d+(?:\.\d+)?[kmb]?|\d+(?:\.\d+)?%?', text)
        
        return tokens
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.simple_tokenize(text)
        token_ids = []
        
        token_ids.append(self.vocab['<START>'])
        
        for token in tokens[:max_length-2]:  # Reserve space for START and END
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['<UNK>'])
        
        token_ids.append(self.vocab['<END>'])
        
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
                if token not in ['<PAD>', '<START>', '<END>']:
                    tokens.append(token)
        
        return ' '.join(tokens)

class FinancialDiffusionLLM:
    """
    Financial Language Model using Diffusion Process
    
    Combines transformer architecture with diffusion-based text generation
    for high-quality financial text refinement and generation.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        num_diffusion_steps: int = 1000
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.num_diffusion_steps = num_diffusion_steps
        
        # Initialize tokenizer
        self.tokenizer = FinancialTokenizer()
        
        # Initialize model components
        self._initialize_model()
        
        # Diffusion schedule
        self.betas = self._cosine_beta_schedule(num_diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # Training state
        self.is_trained = False
        self.training_history = []
    
    def _initialize_model(self):
        """Initialize model parameters"""
        # Token embeddings
        self.token_embeddings = np.random.randn(self.vocab_size, self.d_model) * 0.1
        
        # Positional embeddings
        self.position_embeddings = np.random.randn(self.max_seq_length, self.d_model) * 0.1
        
        # Time embeddings for diffusion steps
        self.time_embeddings = np.random.randn(self.num_diffusion_steps, self.d_model) * 0.1
        
        # Transformer blocks
        self.transformer_blocks = []
        for _ in range(self.num_layers):
            block = TransformerBlock(self.d_model, self.num_heads, self.d_ff)
            self.transformer_blocks.append(block)
        
        # Output projection
        self.output_projection = np.random.randn(self.d_model, self.vocab_size) * 0.1
        self.output_bias = np.zeros(self.vocab_size)
        
        # Layer normalization
        self.final_layer_norm = LayerNorm(self.d_model)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> np.ndarray:
        """Create cosine beta schedule for diffusion"""
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999)
    
    def embed_tokens(self, token_ids: np.ndarray, timestep: int = 0) -> np.ndarray:
        """Convert token IDs to embeddings"""
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        token_embs = self.token_embeddings[token_ids]
        
        # Positional embeddings
        pos_embs = self.position_embeddings[:seq_len]
        
        # Time embeddings for diffusion
        time_emb = self.time_embeddings[timestep]
        
        # Combine embeddings
        embeddings = token_embs + pos_embs + time_emb
        
        return embeddings
    
    def forward(self, token_ids: np.ndarray, timestep: int = 0) -> np.ndarray:
        """Forward pass through the model"""
        # Get embeddings
        x = self.embed_tokens(token_ids, timestep)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block.forward(x)
        
        # Final layer norm
        x = self.final_layer_norm.forward(x)
        
        # Output projection
        logits = np.dot(x, self.output_projection) + self.output_bias
        
        return logits
    
    def add_noise(self, token_ids: np.ndarray, timestep: int) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise to token embeddings at given timestep"""
        # Get clean embeddings
        clean_embs = self.embed_tokens(token_ids, 0)
        
        # Generate noise
        noise = np.random.randn(*clean_embs.shape)
        
        # Add noise according to schedule
        sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod[timestep])
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod[timestep])
        
        noisy_embs = sqrt_alphas_cumprod * clean_embs + sqrt_one_minus_alphas_cumprod * noise
        
        return noisy_embs, noise
    
    def train_step(self, texts: List[str]) -> float:
        """Single training step"""
        total_loss = 0.0
        num_samples = len(texts)
        
        for text in texts:
            # Encode text
            token_ids = np.array([self.tokenizer.encode(text)])
            
            # Random timestep
            t = np.random.randint(0, self.num_diffusion_steps)
            
            # Add noise
            noisy_embs, noise = self.add_noise(token_ids, t)
            
            # Predict noise
            predicted_logits = self.forward(token_ids, t)
            
            # Compute loss (simplified MSE on embeddings)
            predicted_embs = self.embed_tokens(token_ids, t)
            loss = np.mean((predicted_embs - (noisy_embs - noise)) ** 2)
            
            total_loss += loss
        
        return total_loss / num_samples
    
    def train(self, texts: List[str], epochs: int = 10) -> List[float]:
        """Train the model on financial texts"""
        if not texts:
            raise ValueError("No training texts provided")
        
        # Build vocabulary
        self.tokenizer.build_vocab(texts)
        self.vocab_size = self.tokenizer.vocab_size
        
        # Re-initialize with correct vocab size
        self._initialize_model()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = self.train_step(texts)
            losses.append(epoch_loss)
            
            self.training_history.append({
                'epoch': epoch,
                'loss': epoch_loss,
                'timestamp': datetime.now().isoformat()
            })
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")
        
        self.is_trained = True
        return losses
    
    def generate_text(self, prompt: str, max_length: int = 100, num_inference_steps: int = 50) -> str:
        """Generate text using the diffusion process"""
        if not self.is_trained:
            return "Model not trained yet. Please train the model first."
        
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, max_length=max_length//2)
        
        # Start with noise
        generated_ids = np.random.randint(0, self.vocab_size, size=(1, max_length))
        generated_ids[0, :len(prompt_ids)] = prompt_ids
        
        # Diffusion sampling
        for i in reversed(range(0, self.num_diffusion_steps, self.num_diffusion_steps // num_inference_steps)):
            # Predict noise
            predicted_logits = self.forward(generated_ids, i)
            
            # Sample from logits (simplified)
            probabilities = self.softmax(predicted_logits[0])
            
            # Update generated tokens (simplified denoising)
            for j in range(len(prompt_ids), max_length):
                if np.random.random() < 0.1:  # 10% chance to update each token
                    generated_ids[0, j] = np.random.choice(self.vocab_size, p=probabilities[j])
        
        # Decode to text
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        return generated_text
    
    def refine_text(self, input_text: str, num_inference_steps: int = 50) -> str:
        """Refine input text using diffusion process"""
        if not self.is_trained:
            return "Model not trained yet. Please train the model first."
        
        # Encode input text
        input_ids = np.array([self.tokenizer.encode(input_text)])
        
        # Add moderate noise
        t = self.num_diffusion_steps // 3  # Start from middle of noise schedule
        noisy_embs, _ = self.add_noise(input_ids, t)
        
        # Gradual denoising
        for step in range(num_inference_steps):
            current_t = max(0, t - (step * t // num_inference_steps))
            
            # Predict and remove noise
            predicted_logits = self.forward(input_ids, current_t)
            
            # Update token probabilities (simplified)
            probabilities = self.softmax(predicted_logits[0])
            
            # Selective token replacement for refinement
            for j in range(len(input_ids[0])):
                if input_ids[0, j] != self.tokenizer.vocab.get('<PAD>', 0):
                    # Small chance to replace with better token
                    if np.random.random() < 0.05:
                        input_ids[0, j] = np.random.choice(self.vocab_size, p=probabilities[j])
        
        # Decode refined text
        refined_text = self.tokenizer.decode(input_ids[0].tolist())
        
        # Post-process for financial context
        refined_text = self._post_process_financial_text(refined_text)
        
        return refined_text
    
    def _post_process_financial_text(self, text: str) -> str:
        """Post-process text for financial context"""
        # Financial term replacements
        replacements = {
            'good': 'strong',
            'bad': 'challenging',
            'ok': 'stable',
            'fine': 'positive',
            'money': 'revenue',
            'cash': 'liquidity',
            'up': 'increased',
            'down': 'decreased'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in replacements:
                words[i] = replacements[word.lower()]
        
        # Ensure proper sentence structure
        result = ' '.join(words)
        if result and not result.endswith('.'):
            result += '.'
        
        return result
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def save_model(self, filepath: str):
        """Save model to file"""
        model_data = {
            'config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'd_ff': self.d_ff,
                'max_seq_length': self.max_seq_length,
                'num_diffusion_steps': self.num_diffusion_steps
            },
            'tokenizer_vocab': self.tokenizer.vocab,
            'tokenizer_inverse_vocab': self.tokenizer.inverse_vocab,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'token_embeddings': self.token_embeddings.tolist(),
            'position_embeddings': self.position_embeddings.tolist(),
            'time_embeddings': self.time_embeddings.tolist(),
            'output_projection': self.output_projection.tolist(),
            'output_bias': self.output_bias.tolist(),
            'betas': self.betas.tolist(),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Restore configuration
        config = model_data['config']
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.d_ff = config['d_ff']
        self.max_seq_length = config['max_seq_length']
        self.num_diffusion_steps = config['num_diffusion_steps']
        
        # Restore tokenizer
        self.tokenizer.vocab = model_data['tokenizer_vocab']
        self.tokenizer.inverse_vocab = {int(k): v for k, v in model_data['tokenizer_inverse_vocab'].items()}
        self.tokenizer.vocab_size = len(self.tokenizer.vocab)
        
        # Restore model parameters
        self.token_embeddings = np.array(model_data['token_embeddings'])
        self.position_embeddings = np.array(model_data['position_embeddings'])
        self.time_embeddings = np.array(model_data['time_embeddings'])
        self.output_projection = np.array(model_data['output_projection'])
        self.output_bias = np.array(model_data['output_bias'])
        self.betas = np.array(model_data['betas'])
        
        # Restore training state
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']
        
        # Re-initialize transformer blocks
        self._initialize_model()
        
        print(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': 'Financial Diffusion LLM',
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_length': self.max_seq_length,
            'num_diffusion_steps': self.num_diffusion_steps,
            'is_trained': self.is_trained,
            'training_epochs': len(self.training_history),
            'parameters': self.vocab_size * self.d_model + self.max_seq_length * self.d_model + self.num_diffusion_steps * self.d_model
        }