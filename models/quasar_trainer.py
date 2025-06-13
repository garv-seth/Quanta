"""
Quasar Small Training Manager

Advanced training pipeline for the Quasar Small model with:
- Efficient batch processing
- Real-time metrics
- Progressive training strategies
- Database integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import time
import logging
from datetime import datetime
import json

from .quasar_small import QuasarSmall, FinancialTokenizer

logger = logging.getLogger(__name__)

class FinancialTextDataset(Dataset):
    """Dataset for financial text training"""
    
    def __init__(self, texts: List[str], tokenizer: FinancialTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts for efficiency
        self.tokenized_texts = []
        for text in texts:
            tokens = tokenizer.encode(text, max_length)
            self.tokenized_texts.append(tokens)
    
    def __len__(self):
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.tokenized_texts[idx], dtype=torch.long)

class QuasarTrainer:
    """
    Advanced trainer for Quasar Small model
    
    Features:
    - Adaptive learning rate scheduling
    - Gradient accumulation
    - Mixed precision training
    - Real-time monitoring
    - Checkpointing
    """
    
    def __init__(
        self,
        model: QuasarSmall,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        accumulation_steps: int = 1,
        use_mixed_precision: bool = False
    ):
        self.model = model
        self.device = model.device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        
        # Optimizer (AdamW with weight decay)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)  # GPT-style betas
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=warmup_steps,
            eta_min=learning_rate * 0.1
        )
        
        # Mixed precision scaler
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'learning_rate': [],
            'grad_norm': [],
            'tokens_per_second': [],
            'memory_usage': []
        }
        
    def prepare_data(self, texts: List[str], batch_size: int = 8, max_length: int = 512) -> DataLoader:
        """Prepare training data with efficient batching"""
        
        # Build vocabulary if not already built
        if not hasattr(self.model.tokenizer, 'vocab') or len(self.model.tokenizer.vocab) <= len(self.model.tokenizer.special_tokens):
            logger.info("Building vocabulary from training texts...")
            self.model.tokenizer.build_vocab_from_texts(texts)
            logger.info(f"Vocabulary size: {len(self.model.tokenizer.vocab)}")
        
        # Create dataset
        dataset = FinancialTextDataset(texts, self.model.tokenizer, max_length)
        
        # Create dataloader with efficient settings
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Single process for stability
            pin_memory=True if self.device != 'cpu' else False,
            drop_last=True
        )
        
        return dataloader
    
    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute diffusion training loss"""
        B, T = batch.shape
        
        # Move to device
        batch = batch.to(self.device)
        
        # Get embeddings for the input tokens
        with torch.no_grad():
            target_embeddings = self.model.token_embedding(batch)
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.model.num_diffusion_steps, (B,), device=self.device)
        
        # Add noise to embeddings
        noisy_embeddings, noise = self.model.add_noise(target_embeddings, timesteps)
        
        # Forward pass with noisy input
        logits = self.model.forward(batch, timesteps)
        
        # Convert logits back to embeddings for loss computation
        predicted_embeddings = self.model.token_embedding(logits.argmax(dim=-1))
        
        # Compute MSE loss between predicted and target noise
        loss = nn.functional.mse_loss(predicted_embeddings, noise)
        
        # Add auxiliary language modeling loss for stability
        lm_loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch.view(-1),
            ignore_index=self.model.tokenizer.vocab.get('<PAD>', 0)
        )
        
        # Combine losses
        total_loss = loss + 0.1 * lm_loss
        
        return total_loss
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step with gradient accumulation"""
        
        # Compute loss
        if self.use_mixed_precision and self.scaler:
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(batch)
            loss = loss / self.accumulation_steps
            self.scaler.scale(loss).backward()
        else:
            loss = self.compute_loss(batch)
            loss = loss / self.accumulation_steps
            loss.backward()
        
        step_metrics = {'loss': loss.item() * self.accumulation_steps}
        
        # Update parameters every accumulation_steps
        if (self.step + 1) % self.accumulation_steps == 0:
            
            # Gradient clipping
            if self.use_mixed_precision and self.scaler:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            step_metrics.update({
                'grad_norm': grad_norm.item(),
                'learning_rate': self.scheduler.get_last_lr()[0]
            })
        
        self.step += 1
        return step_metrics
    
    def train_epoch(self, dataloader: DataLoader, progress_callback=None) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        epoch_metrics = {'loss': 0.0, 'grad_norm': 0.0, 'tokens_processed': 0}
        num_batches = len(dataloader)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start = time.time()
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Update epoch metrics
            epoch_metrics['loss'] += step_metrics['loss']
            epoch_metrics['tokens_processed'] += batch.numel()
            
            if 'grad_norm' in step_metrics:
                epoch_metrics['grad_norm'] += step_metrics['grad_norm']
            
            # Compute tokens per second
            batch_time = time.time() - batch_start
            tokens_per_second = batch.numel() / batch_time
            
            # Progress callback
            if progress_callback:
                progress_info = {
                    'epoch': self.epoch,
                    'batch': batch_idx + 1,
                    'total_batches': num_batches,
                    'loss': step_metrics['loss'],
                    'tokens_per_second': tokens_per_second,
                    'learning_rate': step_metrics.get('learning_rate', self.learning_rate)
                }
                progress_callback(progress_info)
        
        # Average metrics
        epoch_metrics['loss'] /= num_batches
        epoch_metrics['grad_norm'] /= max(1, num_batches // self.accumulation_steps)
        epoch_metrics['epoch_time'] = time.time() - start_time
        epoch_metrics['tokens_per_second'] = epoch_metrics['tokens_processed'] / epoch_metrics['epoch_time']
        
        return epoch_metrics
    
    def train(
        self,
        texts: List[str],
        epochs: int = 10,
        batch_size: int = 8,
        max_length: int = 512,
        validation_split: float = 0.1,
        save_every: int = 5,
        progress_callback=None
    ) -> List[Dict]:
        """
        Complete training pipeline
        
        Args:
            texts: Training texts
            epochs: Number of training epochs
            batch_size: Batch size for training
            max_length: Maximum sequence length
            validation_split: Fraction of data for validation
            save_every: Save checkpoint every N epochs
            progress_callback: Function to call with training progress
        """
        
        if not texts:
            raise ValueError("No training texts provided")
        
        logger.info(f"Starting Quasar Small training with {len(texts)} texts")
        
        # Split data
        split_idx = int(len(texts) * (1 - validation_split))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:] if validation_split > 0 else []
        
        # Prepare data
        train_dataloader = self.prepare_data(train_texts, batch_size, max_length)
        val_dataloader = self.prepare_data(val_texts, batch_size, max_length) if val_texts else None
        
        logger.info(f"Training batches: {len(train_dataloader)}")
        if val_dataloader:
            logger.info(f"Validation batches: {len(val_dataloader)}")
        
        # Training loop
        training_history = []
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader, progress_callback)
            
            # Validation
            val_metrics = {}
            if val_dataloader:
                val_metrics = self.validate(val_dataloader)
            
            # Combine metrics
            epoch_result = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_tokens_per_second': train_metrics['tokens_per_second'],
                'learning_rate': self.scheduler.get_last_lr()[0],
                'timestamp': datetime.now().isoformat()
            }
            
            if val_metrics:
                epoch_result.update({
                    'val_loss': val_metrics['loss'],
                    'val_tokens_per_second': val_metrics['tokens_per_second']
                })
            
            training_history.append(epoch_result)
            self.training_history.append(epoch_result)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = f"quasar_small_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path, epoch_result)
            
            # Update best model
            current_loss = val_metrics.get('loss', train_metrics['loss'])
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint("quasar_small_best.pt", epoch_result, is_best=True)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_metrics['loss']:.6f}, "
                       f"Tokens/sec = {train_metrics['tokens_per_second']:.0f}")
        
        # Mark as trained
        self.model.is_trained = True
        self.model.training_history = self.training_history
        
        logger.info("Training completed!")
        return training_history
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation step"""
        
        self.model.eval()
        val_metrics = {'loss': 0.0, 'tokens_processed': 0}
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in dataloader:
                loss = self.compute_loss(batch)
                val_metrics['loss'] += loss.item()
                val_metrics['tokens_processed'] += batch.numel()
        
        val_metrics['loss'] /= len(dataloader)
        val_time = time.time() - start_time
        val_metrics['tokens_per_second'] = val_metrics['tokens_processed'] / val_time
        
        self.model.train()
        return val_metrics
    
    def save_checkpoint(self, path: str, metrics: Dict, is_best: bool = False):
        """Save training checkpoint"""
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'current_metrics': metrics,
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'model_config': self.model.get_model_info(),
            'is_best': is_best
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_history = checkpoint.get('training_history', [])
        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {path}")
    
    def estimate_training_time(self, num_texts: int, batch_size: int, epochs: int) -> Dict[str, float]:
        """Estimate training time and resource requirements"""
        
        # Rough estimates based on model size and typical performance
        tokens_per_text = 200  # Average
        total_tokens = num_texts * tokens_per_text * epochs
        
        # Performance estimates (tokens per second)
        if self.device == 'cpu':
            tokens_per_second = 500
        elif 'cuda' in self.device:
            tokens_per_second = 2000  # Rough estimate for GPU
        else:
            tokens_per_second = 1000
        
        estimated_time_seconds = total_tokens / tokens_per_second
        
        # Memory estimates
        model_params = sum(p.numel() for p in self.model.parameters())
        memory_per_param = 4  # bytes for fp32
        base_memory_mb = (model_params * memory_per_param) / (1024 * 1024)
        training_memory_mb = base_memory_mb * 3  # Rough estimate including gradients and activations
        
        return {
            'estimated_time_hours': estimated_time_seconds / 3600,
            'estimated_time_minutes': estimated_time_seconds / 60,
            'total_tokens': total_tokens,
            'tokens_per_second': tokens_per_second,
            'estimated_memory_mb': training_memory_mb,
            'estimated_disk_space_mb': base_memory_mb * 2  # Model + checkpoint
        }

class QuasarModelFactory:
    """Factory for creating and managing Quasar models"""
    
    @staticmethod
    def create_small_model(device: str = "cpu") -> QuasarSmall:
        """Create Quasar Small model optimized for local training"""
        return QuasarSmall(
            vocab_size=16000,  # Smaller vocab for faster training
            d_model=512,       # Reduced from 768
            num_heads=8,       # Reduced from 12
            num_layers=8,      # Reduced from 12
            d_ff=2048,         # Reduced from 3072
            max_seq_len=1024,  # Reduced from 2048
            dropout=0.1,
            num_diffusion_steps=500,  # Reduced for faster training
            device=device
        )
    
    @staticmethod
    def create_medium_model(device: str = "cpu") -> QuasarSmall:
        """Create Quasar Medium model for better performance"""
        return QuasarSmall(
            vocab_size=32000,
            d_model=768,
            num_heads=12,
            num_layers=12,
            d_ff=3072,
            max_seq_len=2048,
            dropout=0.1,
            num_diffusion_steps=1000,
            device=device
        )
    
    @staticmethod
    def create_trainer(model: QuasarSmall, config: Dict = None) -> QuasarTrainer:
        """Create trainer with optimal configuration"""
        
        default_config = {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_steps': 500,
            'max_grad_norm': 1.0,
            'accumulation_steps': 1,
            'use_mixed_precision': False
        }
        
        if config:
            default_config.update(config)
        
        return QuasarTrainer(model, **default_config)