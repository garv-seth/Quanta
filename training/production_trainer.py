"""
Production Training System for Financial Diffusion Models

Optimized for RTX 4060 with 32GB RAM system. Implements efficient training
with memory management, checkpoint handling, and real financial data integration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable, Union
from pathlib import Path
import psutil
import gc

from models.diffusion_model import DiffusionModel
from models.finsar_diffusion import FinSarDiffusion
from utils.text_processor import TextProcessor
from data_collection.financial_data_collector import FinancialDataCollector
from database.real_data_manager import RealFinancialDataManager


class OptimizedFinancialDataset(Dataset):
    """
    Memory-optimized dataset for financial text embeddings.
    
    Uses pre-computed embeddings stored on disk to minimize memory usage
    and enable training on large datasets with limited VRAM.
    """
    
    def __init__(self, embedding_files: List[str], max_length: Optional[int] = None):
        """
        Initialize dataset with pre-computed embedding files.
        
        Args:
            embedding_files: List of .pt files containing embeddings
            max_length: Maximum number of samples to load (for memory management)
        """
        self.embedding_files = embedding_files
        self.embeddings = []
        self.file_indices = []
        
        # Load embeddings from files
        total_loaded = 0
        for file_idx, file_path in enumerate(embedding_files):
            try:
                embeddings = torch.load(file_path, map_location='cpu')
                if isinstance(embeddings, list):
                    embeddings = torch.stack(embeddings)
                
                # Add to dataset
                self.embeddings.append(embeddings)
                self.file_indices.extend([file_idx] * len(embeddings))
                total_loaded += len(embeddings)
                
                print(f"Loaded {len(embeddings)} embeddings from {file_path}")
                
                if max_length and total_loaded >= max_length:
                    break
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Concatenate all embeddings
        if self.embeddings:
            self.embeddings = torch.cat(self.embeddings, dim=0)
            print(f"Total dataset size: {len(self.embeddings)} embeddings")
        else:
            raise ValueError("No valid embeddings found in provided files")
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]


class HardwareOptimizer:
    """
    Hardware optimization utilities for RTX 4060 systems.
    
    Provides memory management, batch size optimization, and performance monitoring
    specifically tuned for 8GB VRAM and 32GB system RAM.
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.total_vram_gb = self._get_total_vram() if self.device == 'cuda' else 0
        self.system_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"Hardware detected: {self.device}")
        if self.device == 'cuda':
            print(f"VRAM: {self.total_vram_gb:.1f}GB, System RAM: {self.system_ram_gb:.1f}GB")
    
    def _get_total_vram(self) -> float:
        """Get total VRAM in GB."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0
    
    def optimize_training_config(self, model: nn.Module, 
                                target_vram_usage: float = 0.85) -> Dict[str, int]:
        """
        Calculate optimal training configuration for RTX 4060.
        
        Args:
            model: Model to optimize for
            target_vram_usage: Target VRAM utilization (0.85 = 85%)
            
        Returns:
            Dictionary with optimized parameters
        """
        # Model memory footprint
        model_params = sum(p.numel() for p in model.parameters())
        model_memory_mb = model_params * 4 / (1024**2)  # 4 bytes per float32
        
        # Account for gradients (2x model size) and optimizer states (2x model size)
        training_overhead = model_memory_mb * 4
        
        # Available memory for batch processing
        available_vram_mb = (self.total_vram_gb * 1024 * target_vram_usage) - training_overhead
        
        # Estimate memory per sample (embedding + forward/backward passes)
        embedding_dim = 384  # Standard for sentence transformers
        memory_per_sample_mb = embedding_dim * 4 * 8 / (1024**2)  # 8x for forward/backward
        
        # Calculate optimal batch size
        max_batch_size = max(1, int(available_vram_mb / memory_per_sample_mb))
        
        # Clamp to reasonable values for RTX 4060
        optimal_batch_size = min(max_batch_size, 64)
        optimal_batch_size = max(optimal_batch_size, 1)
        
        # Gradient accumulation for effective larger batch sizes
        effective_batch_size = 256  # Target effective batch size
        grad_accumulation_steps = max(1, effective_batch_size // optimal_batch_size)
        
        return {
            'batch_size': optimal_batch_size,
            'grad_accumulation_steps': grad_accumulation_steps,
            'effective_batch_size': optimal_batch_size * grad_accumulation_steps,
            'estimated_vram_usage_mb': training_overhead + optimal_batch_size * memory_per_sample_mb,
            'model_memory_mb': model_memory_mb,
            'num_workers': min(4, os.cpu_count())  # For data loading
        }
    
    def monitor_memory(self) -> Dict[str, float]:
        """Monitor current memory usage."""
        memory_info = {
            'system_ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'system_ram_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent()
        }
        
        if self.device == 'cuda' and torch.cuda.is_available():
            memory_info.update({
                'vram_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'vram_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'vram_percent': (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
            })
        
        return memory_info
    
    def cleanup_memory(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class ProductionTrainer:
    """
    Production-ready trainer optimized for RTX 4060 hardware.
    
    Features:
    - Memory-efficient training with gradient accumulation
    - Automatic mixed precision (AMP) for faster training
    - Real-time monitoring and checkpointing
    - Financial data integration
    - Hardware-optimized batch sizing
    """
    
    def __init__(self, model_type: str = 'finsar', device: Optional[str] = None):
        """
        Initialize production trainer.
        
        Args:
            model_type: Type of model to train ('diffusion' or 'finsar')
            device: Training device (auto-detected if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Initialize components
        self.hardware_optimizer = HardwareOptimizer()
        self.text_processor = TextProcessor()
        self.data_manager = RealFinancialDataManager()
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        
        # Metrics tracking
        self.training_history = []
        self.best_loss = float('inf')
        self.start_time = None
        
        # Create directories
        self.checkpoint_dir = Path("checkpoints")
        self.embedding_dir = Path("embeddings")
        self.logs_dir = Path("logs")
        
        for dir_path in [self.checkpoint_dir, self.embedding_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"ProductionTrainer initialized on {self.device}")
    
    def create_model(self, embedding_dim: int = 384, num_steps: int = 100, 
                    hidden_dim: int = 512) -> nn.Module:
        """
        Create and initialize the model.
        
        Args:
            embedding_dim: Embedding dimension
            num_steps: Number of diffusion steps
            hidden_dim: Hidden layer dimension
            
        Returns:
            Initialized model
        """
        if self.model_type == 'finsar':
            self.model = FinSarDiffusion(
                embedding_dim=embedding_dim,
                num_steps=num_steps,
                hidden_dim=hidden_dim,
                num_paths=50
            )
        else:
            self.model = DiffusionModel(
                embedding_dim=embedding_dim,
                num_steps=num_steps,
                hidden_dim=hidden_dim
            )
        
        self.model.to(self.device)
        
        # Get hardware-optimized configuration
        self.training_config = self.hardware_optimizer.optimize_training_config(self.model)
        
        print(f"Model created: {self.model_type}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Optimized config: {self.training_config}")
        
        return self.model
    
    def collect_training_data(self, num_companies: int = 100) -> str:
        """
        Collect and prepare real financial training data.
        
        Args:
            num_companies: Number of companies to collect data for
            
        Returns:
            Path to saved embeddings file
        """
        print("Collecting real financial data...")
        
        # Collect live financial data
        collection_stats = self.data_manager.collect_all_live_data()
        print(f"Data collection stats: {collection_stats}")
        
        # Extract training texts from database
        training_texts = self.data_manager.prepare_training_texts_from_db()
        print(f"Extracted {len(training_texts)} training texts")
        
        if not training_texts:
            raise ValueError("No training texts found. Check data collection process.")
        
        # Generate embeddings in batches to manage memory
        embeddings_file = self.embedding_dir / f"financial_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        print("Generating embeddings...")
        batch_size = 50  # Process in small batches
        all_embeddings = []
        
        for i in range(0, len(training_texts), batch_size):
            batch_texts = training_texts[i:i + batch_size]
            batch_embeddings = self.text_processor.batch_process_texts(batch_texts)
            all_embeddings.append(batch_embeddings.cpu())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_texts)}/{len(training_texts)} texts")
        
        # Combine and save embeddings
        final_embeddings = torch.cat(all_embeddings, dim=0)
        torch.save(final_embeddings, embeddings_file)
        
        print(f"Saved {len(final_embeddings)} embeddings to {embeddings_file}")
        return str(embeddings_file)
    
    def setup_training(self, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        """
        Setup optimizer and scheduler for training.
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # AdamW optimizer with hardware-optimized settings
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=100,  # Initial restart period
            T_mult=2,  # Multiply restart period by this factor
            eta_min=learning_rate * 0.01
        )
        
        print("Training setup complete")
    
    def train_step(self, batch: torch.Tensor, accumulate_gradients: bool = False) -> float:
        """
        Perform optimized training step with gradient accumulation.
        
        Args:
            batch: Training batch
            accumulate_gradients: Whether to accumulate gradients
            
        Returns:
            Loss value
        """
        if not accumulate_gradients:
            self.optimizer.zero_grad()
        
        batch = batch.to(self.device, non_blocking=True)
        batch_size = batch.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.model.num_steps, (batch_size,), device=self.device)
        
        # Forward pass with automatic mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                if isinstance(self.model, FinSarDiffusion):
                    loss = self.model.compute_path_integral_loss(batch, t)
                else:
                    loss = self.model.compute_loss(batch, t)
                
                # Scale loss for gradient accumulation
                loss = loss / self.training_config['grad_accumulation_steps']
            
            self.scaler.scale(loss).backward()
        else:
            if isinstance(self.model, FinSarDiffusion):
                loss = self.model.compute_path_integral_loss(batch, t)
            else:
                loss = self.model.compute_loss(batch, t)
            
            loss = loss / self.training_config['grad_accumulation_steps']
            loss.backward()
        
        return loss.item() * self.training_config['grad_accumulation_steps']
    
    def train_epoch(self, dataloader: DataLoader, epoch: int,
                   progress_callback: Optional[Callable] = None) -> Dict[str, float]:
        """
        Train for one epoch with hardware optimization.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            progress_callback: Progress update callback
            
        Returns:
            Epoch statistics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(dataloader)
        grad_accumulation_steps = self.training_config['grad_accumulation_steps']
        
        for batch_idx, batch in enumerate(dataloader):
            # Check if this is the last step in accumulation
            is_accumulation_step = (batch_idx + 1) % grad_accumulation_steps != 0
            is_last_batch = batch_idx == num_batches - 1
            
            # Training step
            loss = self.train_step(batch, accumulate_gradients=is_accumulation_step)
            total_loss += loss
            
            # Optimizer step when accumulation is complete
            if not is_accumulation_step or is_last_batch:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Progress callback
            if progress_callback and batch_idx % 10 == 0:
                progress_callback(epoch, batch_idx, loss, num_batches)
            
            # Memory cleanup every 50 batches
            if batch_idx % 50 == 0:
                self.hardware_optimizer.cleanup_memory()
        
        avg_loss = total_loss / num_batches
        return {
            'avg_loss': avg_loss,
            'total_loss': total_loss,
            'num_batches': num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def train(self, embedding_files: List[str], num_epochs: int = 10,
             validation_split: float = 0.1, save_every: int = 5,
             progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Complete training pipeline with hardware optimization.
        
        Args:
            embedding_files: List of embedding files to train on
            num_epochs: Number of training epochs
            validation_split: Validation split ratio
            save_every: Save checkpoint every N epochs
            progress_callback: Progress callback function
            
        Returns:
            Training history
        """
        self.start_time = time.time()
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Hardware config: {self.training_config}")
        
        # Create dataset
        dataset = OptimizedFinancialDataset(embedding_files)
        
        # Split dataset
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create optimized data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=self.training_config['num_workers'],
            pin_memory=(self.device == 'cuda'),
            persistent_workers=(self.training_config['num_workers'] > 0)
        )
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_stats = self.train_epoch(train_loader, epoch, progress_callback)
            
            # Memory monitoring
            memory_stats = self.hardware_optimizer.monitor_memory()
            
            # Record statistics
            epoch_stats = {
                'epoch': epoch,
                'train_loss': train_stats['avg_loss'],
                'learning_rate': train_stats['learning_rate'],
                'epoch_time': time.time() - epoch_start,
                'memory_stats': memory_stats
            }
            
            self.training_history.append(epoch_stats)
            
            # Update best loss and save checkpoint
            if train_stats['avg_loss'] < self.best_loss:
                self.best_loss = train_stats['avg_loss']
                self.save_checkpoint('best_model.pth', epoch)
            
            # Regular checkpoint saving
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Loss: {train_stats['avg_loss']:.6f}, "
                  f"LR: {train_stats['learning_rate']:.6f}, "
                  f"Time: {epoch_stats['epoch_time']:.1f}s")
            print(f"Memory: VRAM {memory_stats.get('vram_percent', 0):.1f}%, "
                  f"RAM {memory_stats['system_ram_percent']:.1f}%")
            
            # Cleanup
            self.hardware_optimizer.cleanup_memory()
        
        total_time = time.time() - self.start_time
        print(f"Training completed in {total_time:.1f}s")
        print(f"Best loss: {self.best_loss:.6f}")
        
        return self.training_history
    
    def save_checkpoint(self, filename: str, epoch: int):
        """Save training checkpoint with comprehensive state."""
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'training_config': self.training_config,
            'model_config': {
                'model_type': self.model_type,
                'embedding_dim': self.model.embedding_dim,
                'num_steps': self.model.num_steps,
                'hidden_dim': self.model.hidden_dim
            },
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Checkpoint loaded: {filepath}")
        return checkpoint


def quick_start_training(model_type: str = 'finsar', num_epochs: int = 10) -> ProductionTrainer:
    """
    Quick start function for immediate training.
    
    Args:
        model_type: Type of model to train ('finsar' or 'diffusion')
        num_epochs: Number of training epochs
        
    Returns:
        Configured trainer instance
    """
    # Initialize trainer
    trainer = ProductionTrainer(model_type=model_type)
    
    # Create model with RTX 4060 optimized settings
    trainer.create_model(
        embedding_dim=384,
        num_steps=100,
        hidden_dim=512
    )
    
    # Setup training with conservative learning rate
    trainer.setup_training(learning_rate=5e-4, weight_decay=1e-5)
    
    # Collect real financial data
    embedding_file = trainer.collect_training_data(num_companies=50)
    
    # Start training
    def progress_callback(epoch, batch_idx, loss, total_batches):
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}/{total_batches}, Loss: {loss:.6f}")
    
    trainer.train(
        embedding_files=[embedding_file],
        num_epochs=num_epochs,
        validation_split=0.1,
        save_every=2,
        progress_callback=progress_callback
    )
    
    return trainer