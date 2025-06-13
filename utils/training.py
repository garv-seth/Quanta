"""
Training utilities for the Financial Text Diffusion Model.

This module provides training loops, optimization strategies, and monitoring
utilities for training the diffusion model on financial text data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
import time
from tqdm import tqdm
import logging


class FinancialTextDataset(Dataset):
    """
    Dataset class for financial text embeddings.
    
    Handles loading and preprocessing of financial text data for training
    the diffusion model.
    """
    
    def __init__(self, texts: List[str], text_processor):
        """
        Initialize the dataset.
        
        Args:
            texts: List of financial text strings
            text_processor: TextProcessor instance for generating embeddings
        """
        self.texts = texts
        self.text_processor = text_processor
        self.embeddings = None
        self._generate_embeddings()
    
    def _generate_embeddings(self):
        """Generate embeddings for all texts."""
        print("Generating embeddings for training data...")
        self.embeddings = self.text_processor.batch_process_texts(self.texts)
        print(f"Generated {len(self.embeddings)} embeddings")
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]


class ModelTrainer:
    """
    Trainer class for the diffusion model.
    
    Handles the complete training pipeline including optimization,
    loss computation, and progress monitoring.
    """
    
    def __init__(self, model, text_processor, device: str = None):
        """
        Initialize the trainer.
        
        Args:
            model: The diffusion model to train
            text_processor: TextProcessor instance
            device: Device to use for training (auto-detected if None)
        """
        self.model = model
        self.text_processor = text_processor
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        
        # Metrics tracking
        self.training_history = []
        self.best_loss = float('inf')
        
        print(f"Trainer initialized on device: {self.device}")
    
    def setup_optimizer(self, learning_rate: float = 1e-3, 
                       weight_decay: float = 1e-5) -> None:
        """
        Setup optimizer and learning rate scheduler.
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
        """
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,  # Will be adjusted based on training steps
            eta_min=learning_rate * 0.01
        )
    
    def train_step(self, batch: torch.Tensor) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of embeddings
            
        Returns:
            Loss value for this step
        """
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        batch_size = batch.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.model.num_steps, (batch_size,), device=self.device)
        
        # Compute loss
        if self.scaler is not None:
            # Use automatic mixed precision for GPU training
            with torch.cuda.amp.autocast():
                loss = self.model.compute_loss(batch, t)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # CPU training or full precision
            loss = self.model.compute_loss(batch, t)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.item()
    
    def validate_step(self, batch: torch.Tensor) -> float:
        """
        Perform a validation step.
        
        Args:
            batch: Batch of embeddings
            
        Returns:
            Validation loss
        """
        with torch.no_grad():
            batch = batch.to(self.device)
            batch_size = batch.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, self.model.num_steps, (batch_size,), device=self.device)
            
            # Compute validation loss
            loss = self.model.compute_loss(batch, t)
            
            return loss.item()
    
    def train_epoch(self, dataloader: DataLoader, 
                   progress_callback: Optional[Callable] = None) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            loss = self.train_step(batch)
            total_loss += loss
            
            # Update progress
            if progress_callback:
                progress_callback(batch_idx, loss)
        
        avg_loss = total_loss / num_batches
        
        return {
            'avg_loss': avg_loss,
            'total_loss': total_loss,
            'num_batches': num_batches
        }
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with validation statistics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch in dataloader:
            loss = self.validate_step(batch)
            total_loss += loss
        
        avg_loss = total_loss / num_batches
        
        return {
            'avg_loss': avg_loss,
            'total_loss': total_loss,
            'num_batches': num_batches
        }
    
    def train(self, texts: List[str], 
              learning_rate: float = 1e-3,
              batch_size: int = 4,
              num_epochs: int = 10,
              validation_split: float = 0.1,
              progress_callback: Optional[Callable] = None) -> List[Dict[str, float]]:
        """
        Complete training loop.
        
        Args:
            texts: List of training texts
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of training history dictionaries
        """
        print(f"Starting training with {len(texts)} texts")
        print(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {learning_rate}")
        
        # Setup optimizer
        self.setup_optimizer(learning_rate)
        
        # Create dataset and split
        dataset = FinancialTextDataset(texts, self.text_processor)
        
        # Split into train and validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders with memory optimization
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=(self.device == 'cuda')
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device == 'cuda')
        ) if val_size > 0 else None
        
        # Training loop
        training_history = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            train_stats = self.train_epoch(
                train_loader,
                lambda batch_idx, loss: progress_callback(epoch, batch_idx, loss) if progress_callback else None
            )
            
            # Validation phase
            val_stats = None
            if val_loader is not None:
                val_stats = self.validate_epoch(val_loader)
            
            # Record statistics
            epoch_stats = {
                'epoch': epoch,
                'train_loss': train_stats['avg_loss'],
                'val_loss': val_stats['avg_loss'] if val_stats else None,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': time.time() - start_time
            }
            
            training_history.append(epoch_stats)
            self.training_history.append({
                'loss': train_stats['avg_loss'],
                'epoch': epoch,
                'timestamp': time.time()
            })
            
            # Update best loss
            if train_stats['avg_loss'] < self.best_loss:
                self.best_loss = train_stats['avg_loss']
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_stats['avg_loss']:.6f}")
            if val_stats:
                print(f"                     Val Loss: {val_stats['avg_loss']:.6f}")
            
            # Memory cleanup
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"Training completed! Best loss: {self.best_loss:.6f}")
        return training_history
    
    def save_checkpoint(self, filepath: str, epoch: int = None) -> None:
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history': self.training_history,
            'best_loss': self.best_loss,
            'epoch': epoch,
            'model_config': {
                'embedding_dim': self.model.embedding_dim,
                'num_steps': self.model.num_steps,
                'hidden_dim': self.model.hidden_dim
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        self.training_history = checkpoint.get('training_history', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        if self.device == 'cuda' and torch.cuda.is_available():
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'cached_mb': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2
            }
        else:
            return {'allocated_mb': 0, 'cached_mb': 0, 'max_allocated_mb': 0}
    
    def optimize_for_hardware(self, target_vram_gb: float = 8.0) -> Dict[str, int]:
        """
        Suggest optimal training parameters for given hardware constraints.
        
        Args:
            target_vram_gb: Target VRAM usage in GB
            
        Returns:
            Dictionary with suggested parameters
        """
        model_size_mb = sum(p.numel() * 4 for p in self.model.parameters()) / 1024**2
        
        # Estimate memory usage per sample (rough approximation)
        embedding_size_mb = self.model.embedding_dim * 4 / 1024**2
        
        # Account for gradients (roughly 2x model size) and optimizer states (roughly 2x)
        base_memory_mb = model_size_mb * 4
        
        # Calculate maximum batch size
        available_memory_mb = target_vram_gb * 1024 - base_memory_mb
        max_batch_size = max(1, int(available_memory_mb / (embedding_size_mb * 4)))
        
        # Suggest reasonable batch size (power of 2, but not too large)
        suggested_batch_size = min(max_batch_size, 8)
        suggested_batch_size = max(1, 2 ** int(np.log2(suggested_batch_size)))
        
        return {
            'suggested_batch_size': suggested_batch_size,
            'max_batch_size': max_batch_size,
            'model_size_mb': model_size_mb,
            'estimated_memory_usage_mb': base_memory_mb + suggested_batch_size * embedding_size_mb * 4
        }
