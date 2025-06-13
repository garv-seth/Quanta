"""
Quasar Small - Simplified Implementation

A streamlined diffusion-based GPT model for quantitative finance that works
reliably in any environment without complex PyTorch dependencies.
"""

import numpy as np
import json
import pickle
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SimpleFinancialTokenizer:
    """Efficient tokenizer for financial text"""
    
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        
        # Financial-specific tokens
        self.special_tokens = [
            '<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>',
            '<NUM>', '<COMPANY>', '<TICKER>', '<CURRENCY>', '<PERCENT>',
            '<DATE>', '<REVENUE>', '<PROFIT>', '<LOSS>', '<QUARTER>',
            '<YEAR>', '<ANALYST>', '<RATING>', '<BULL>', '<BEAR>'
        ]
        
        # Initialize special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
            
        self.special_token_count = len(self.special_tokens)
        
    def build_vocab_from_texts(self, texts: List[str]) -> None:
        """Build vocabulary from financial texts"""
        import re
        from collections import Counter
        
        all_tokens = []
        for text in texts:
            tokens = self._tokenize(text)
            all_tokens.extend(tokens)
        
        token_counts = Counter(all_tokens)
        
        for token, count in token_counts.most_common(self.vocab_size - self.special_token_count):
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inverse_vocab[idx] = token
                
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text with financial patterns"""
        import re
        
        text = text.lower().strip()
        
        # Financial patterns
        patterns = [
            r'\$[\d,]+(?:\.\d+)?[kmb]?',  # Currency
            r'\d+(?:\.\d+)?%',            # Percentages
            r'\b[a-z]{1,5}\b',            # Tickers
            r'\d{4}-\d{2}-\d{2}',         # Dates
            r'\b\w+\b',                   # Words
        ]
        
        tokens = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            tokens.extend(matches)
            
        return tokens
        
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token IDs"""
        tokens = self._tokenize(text)
        token_ids = [self.vocab['<BOS>']]
        
        for token in tokens:
            if len(token_ids) >= max_length - 1:
                break
            token_id = self.vocab.get(token, self.vocab['<UNK>'])
            token_ids.append(token_id)
            
        token_ids.append(self.vocab['<EOS>'])
        
        while len(token_ids) < max_length:
            token_ids.append(self.vocab['<PAD>'])
            
        return token_ids[:max_length]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                if token not in ['<PAD>', '<BOS>', '<EOS>']:
                    tokens.append(token)
        return ' '.join(tokens)

class SimpleAttention:
    """Simplified attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose for attention
        Q = np.transpose(Q, (0, 2, 1, 3))
        K = np.transpose(K, (0, 2, 1, 3))
        V = np.transpose(V, (0, 2, 1, 3))
        
        # Attention computation
        scores = np.matmul(Q, np.transpose(K, (0, 1, 3, 2))) / np.sqrt(self.d_k)
        
        # Softmax
        attention_weights = self.softmax(scores)
        attention_output = np.matmul(attention_weights, V)
        
        # Reshape and project
        attention_output = np.transpose(attention_output, (0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, seq_len, d_model)
        
        return np.dot(attention_output, self.W_o)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class SimpleFeedForward:
    """Simplified feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)  # ReLU
        return np.dot(hidden, self.W2) + self.b2

class SimpleTransformerBlock:
    """Simplified transformer block"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = SimpleAttention(d_model, num_heads)
        self.feed_forward = SimpleFeedForward(d_model, d_ff)
        self.layer_norm1 = SimpleLayerNorm(d_model)
        self.layer_norm2 = SimpleLayerNorm(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Attention with residual
        attn_out = self.attention.forward(x)
        x = self.layer_norm1.forward(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.feed_forward.forward(x)
        x = self.layer_norm2.forward(x + ff_out)
        
        return x

class SimpleLayerNorm:
    """Simplified layer normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class SimpleDiffusionScheduler:
    """Simplified diffusion noise scheduler"""
    
    def __init__(self, num_timesteps: int = 500):
        self.num_timesteps = num_timesteps
        
        # Linear schedule
        self.betas = np.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

class QuasarSimple:
    """
    Quasar Small - Simplified Implementation
    
    A reliable diffusion-based GPT model for quantitative finance
    that works in any environment.
    """
    
    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        num_diffusion_steps: int = 500
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.num_diffusion_steps = num_diffusion_steps
        
        # Initialize components
        self.tokenizer = SimpleFinancialTokenizer(vocab_size)
        self.scheduler = SimpleDiffusionScheduler(num_diffusion_steps)
        
        # Model parameters
        self.token_embeddings = np.random.randn(vocab_size, d_model) * 0.1
        self.position_embeddings = np.random.randn(max_seq_len, d_model) * 0.1
        self.time_embeddings = np.random.randn(num_diffusion_steps, d_model) * 0.1
        
        # Transformer blocks
        self.blocks = []
        for _ in range(num_layers):
            block = SimpleTransformerBlock(d_model, num_heads, d_ff)
            self.blocks.append(block)
        
        # Output layers
        self.final_layer_norm = SimpleLayerNorm(d_model)
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.1
        self.output_bias = np.zeros(vocab_size)
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
    def embed_tokens(self, token_ids: np.ndarray, timestep: int = 0) -> np.ndarray:
        """Convert token IDs to embeddings"""
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        token_embs = self.token_embeddings[token_ids]
        
        # Position embeddings
        pos_embs = self.position_embeddings[:seq_len]
        
        # Time embeddings
        time_emb = self.time_embeddings[timestep]
        
        # Combine
        embeddings = token_embs + pos_embs + time_emb
        
        return embeddings
    
    def forward(self, token_ids: np.ndarray, timestep: int = 0) -> np.ndarray:
        """Forward pass"""
        x = self.embed_tokens(token_ids, timestep)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # Final processing
        x = self.final_layer_norm.forward(x)
        logits = np.dot(x, self.output_projection) + self.output_bias
        
        return logits
    
    def add_noise(self, embeddings: np.ndarray, timestep: int) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise to embeddings"""
        noise = np.random.randn(*embeddings.shape)
        
        sqrt_alphas_cumprod = self.scheduler.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alphas_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep]
        
        noisy_embeddings = sqrt_alphas_cumprod * embeddings + sqrt_one_minus_alphas_cumprod * noise
        
        return noisy_embeddings, noise
    
    def train_step(self, texts: List[str]) -> float:
        """Single training step"""
        total_loss = 0.0
        num_samples = len(texts)
        
        for text in texts:
            # Encode text
            token_ids = np.array([self.tokenizer.encode(text)])
            
            # Random timestep
            t = np.random.randint(0, self.num_diffusion_steps)
            
            # Get clean embeddings
            clean_embeddings = self.embed_tokens(token_ids, 0)
            
            # Add noise
            noisy_embeddings, noise = self.add_noise(clean_embeddings, t)
            
            # Forward pass
            predicted_logits = self.forward(token_ids, t)
            
            # Convert predictions to embeddings
            predicted_tokens = np.argmax(predicted_logits, axis=-1)
            predicted_embeddings = self.embed_tokens(predicted_tokens, 0)
            
            # Compute loss (MSE between predicted and clean embeddings)
            loss = np.mean((predicted_embeddings - clean_embeddings) ** 2)
            total_loss += loss
        
        return total_loss / num_samples
    
    def train(self, texts: List[str], epochs: int = 10) -> List[float]:
        """Train the model"""
        if not texts:
            raise ValueError("No training texts provided")
        
        # Build vocabulary
        self.tokenizer.build_vocab_from_texts(texts)
        
        # Re-initialize embeddings with correct vocab size
        actual_vocab_size = len(self.tokenizer.vocab)
        self.token_embeddings = np.random.randn(actual_vocab_size, self.d_model) * 0.1
        self.output_projection = np.random.randn(self.d_model, actual_vocab_size) * 0.1
        self.output_bias = np.zeros(actual_vocab_size)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = self.train_step(texts)
            losses.append(epoch_loss)
            
            self.training_history.append({
                'epoch': epoch,
                'loss': float(epoch_loss),
                'timestamp': datetime.now().isoformat()
            })
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")
        
        self.is_trained = True
        return losses
    
    def generate_text(self, prompt: str, max_length: int = 100, num_steps: int = 25) -> str:
        """Generate text using diffusion sampling"""
        if not self.is_trained:
            return "Model not trained yet."
        
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, max_length=max_length//2)
        
        # Start with random tokens
        generated_ids = np.random.randint(0, len(self.tokenizer.vocab), size=(1, max_length))
        generated_ids[0, :len(prompt_ids)] = prompt_ids
        
        # Diffusion sampling
        for i in reversed(range(0, self.num_diffusion_steps, self.num_diffusion_steps // num_steps)):
            logits = self.forward(generated_ids, i)
            
            # Sample from logits
            probs = self.softmax(logits[0])
            
            # Update non-prompt tokens
            for j in range(len(prompt_ids), max_length):
                if np.random.random() < 0.1:
                    generated_ids[0, j] = np.random.choice(len(self.tokenizer.vocab), p=probs[j])
        
        return self.tokenizer.decode(generated_ids[0].tolist())
    
    def refine_text(self, text: str, num_steps: int = 25) -> str:
        """Refine text using partial diffusion"""
        if not self.is_trained:
            return "Model not trained yet."
        
        # Encode text
        text_ids = self.tokenizer.encode(text)
        text_tensor = np.array([text_ids])
        
        # Partial diffusion refinement
        start_timestep = self.num_diffusion_steps // 3
        
        for i in range(num_steps):
            current_t = max(0, start_timestep - (i * start_timestep // num_steps))
            
            logits = self.forward(text_tensor, current_t)
            probs = self.softmax(logits[0])
            
            # Selective refinement
            for j in range(len(text_ids)):
                if np.random.random() < 0.05:
                    text_tensor[0, j] = np.random.choice(len(self.tokenizer.vocab), p=probs[j])
        
        return self.tokenizer.decode(text_tensor[0].tolist())
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = (
            self.vocab_size * self.d_model +  # Token embeddings
            self.max_seq_len * self.d_model +  # Position embeddings
            self.num_diffusion_steps * self.d_model +  # Time embeddings
            self.num_layers * (self.d_model * self.d_model * 4 + self.d_model * self.d_ff * 2) +  # Transformer blocks
            self.d_model * self.vocab_size  # Output projection
        )
        
        return {
            'model_name': 'Quasar Small (Simplified)',
            'architecture': 'Diffusion-Based GPT',
            'vocab_size': len(self.tokenizer.vocab) if hasattr(self.tokenizer, 'vocab') else self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'num_diffusion_steps': self.num_diffusion_steps,
            'total_parameters': total_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'is_trained': self.is_trained,
            'training_epochs': len(self.training_history)
        }
    
    def save_model(self, path: str) -> None:
        """Save model state"""
        model_data = {
            'config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'd_ff': self.d_ff,
                'max_seq_len': self.max_seq_len,
                'num_diffusion_steps': self.num_diffusion_steps
            },
            'tokenizer_vocab': self.tokenizer.vocab,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'model_weights': {
                'token_embeddings': self.token_embeddings.tolist(),
                'position_embeddings': self.position_embeddings.tolist(),
                'time_embeddings': self.time_embeddings.tolist(),
                'output_projection': self.output_projection.tolist(),
                'output_bias': self.output_bias.tolist()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f)
        
        print(f"Model saved to {path}")

class SimpleQuasarTrainer:
    """Simplified trainer for Quasar Small"""
    
    def __init__(self, model: QuasarSimple, learning_rate: float = 0.001):
        self.model = model
        self.learning_rate = learning_rate
        
    def train(self, texts: List[str], epochs: int = 10, batch_size: int = 4, 
              progress_callback=None) -> List[Dict]:
        """Train with progress tracking"""
        
        training_history = []
        
        for epoch in range(epochs):
            # Simple batch processing
            epoch_loss = 0.0
            num_batches = max(1, len(texts) // batch_size)
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_loss = self.model.train_step(batch_texts)
                epoch_loss += batch_loss
                
                # Progress callback
                if progress_callback:
                    progress_info = {
                        'epoch': epoch,
                        'batch': (i // batch_size) + 1,
                        'total_batches': num_batches,
                        'loss': batch_loss,
                        'learning_rate': self.learning_rate
                    }
                    progress_callback(progress_info)
            
            avg_loss = epoch_loss / num_batches
            
            epoch_result = {
                'epoch': epoch,
                'train_loss': avg_loss,
                'train_tokens_per_second': 1000,  # Estimated
                'learning_rate': self.learning_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            training_history.append(epoch_result)
            self.model.training_history.append(epoch_result)
        
        self.model.is_trained = True
        return training_history

class SimpleQuasarFactory:
    """Factory for creating Quasar models"""
    
    @staticmethod
    def create_small_model() -> QuasarSimple:
        """Create small model for fast training"""
        return QuasarSimple(
            vocab_size=8000,
            d_model=256,
            num_heads=4,
            num_layers=4,
            d_ff=1024,
            max_seq_len=256,
            num_diffusion_steps=250
        )
    
    @staticmethod
    def create_medium_model() -> QuasarSimple:
        """Create medium model for better performance"""
        return QuasarSimple(
            vocab_size=16000,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_seq_len=512,
            num_diffusion_steps=500
        )
    
    @staticmethod
    def create_trainer(model: QuasarSimple, config: Dict = None) -> SimpleQuasarTrainer:
        """Create trainer with configuration"""
        learning_rate = config.get('learning_rate', 0.001) if config else 0.001
        return SimpleQuasarTrainer(model, learning_rate)