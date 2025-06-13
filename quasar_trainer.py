"""
Quanta Quasar Financial Diffusion Model Trainer
Optimized for 62GB RAM Replit environment and RTX 4060 local hardware
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import json
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc

# Try importing optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import core utilities
from data.sample_texts import get_sample_financial_texts, get_sample_text_pairs
from models.diffusion_model import DiffusionModel


class SimpleEmbedder:
    """Simple text embedding processor for financial data."""
    
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Convert texts to embeddings using simple word-based approach."""
        embeddings = []
        
        for text in texts:
            # Simple tokenization and embedding
            words = text.lower().split()
            
            # Create embedding based on word characteristics
            embedding = np.zeros(self.embedding_dim)
            
            for i, word in enumerate(words[:50]):  # Limit to 50 words
                # Use word hash for reproducible embeddings
                word_hash = hash(word) % self.embedding_dim
                embedding[word_hash] += 1.0
                
                # Add positional encoding
                if i < self.embedding_dim // 2:
                    embedding[i] += 0.1
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)


class QuasarTrainer:
    """Production-optimized trainer for Quasar financial diffusion model."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.training_history = []
        self.best_loss = float('inf')
        
        # Optimized for 62GB RAM environment
        self.batch_size = 32
        self.accumulate_gradients = 2
        self.mixed_precision = True
        
    def create_model(self, embedding_dim=384, num_steps=100, hidden_dim=512):
        """Create optimized diffusion model."""
        self.model = DiffusionModel(
            embedding_dim=embedding_dim,
            num_steps=num_steps,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        st.success(f"Model created with {total_params:,} parameters")
        
        return self.model
    
    def setup_optimizer(self, learning_rate=5e-4):
        """Setup optimizer and scheduler."""
        if self.model is None:
            raise ValueError("Model must be created first")
            
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=learning_rate * 0.01
        )
    
    def train_step(self, batch, accumulate_gradients=False):
        """Optimized training step."""
        if self.model is None or self.optimizer is None:
            raise ValueError("Model and optimizer must be set up first")
        
        if not accumulate_gradients:
            self.optimizer.zero_grad()
        
        # Mixed precision training
        if self.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                # Generate random timesteps
                timesteps = torch.randint(0, self.model.num_steps, (batch.size(0),)).to(self.device)
                loss = self.model.compute_loss(batch, timesteps)
        else:
            timesteps = torch.randint(0, self.model.num_steps, (batch.size(0),)).to(self.device)
            loss = self.model.compute_loss(batch, timesteps)
        
        # Backward pass
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item()
    
    def train_epoch(self, dataloader, epoch, progress_callback=None):
        """Train one epoch with optimization."""
        if self.model is None:
            raise ValueError("Model must be created first")
        
        self.model.train()
        epoch_losses = []
        total_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            
            # Gradient accumulation
            accumulate = (batch_idx + 1) % self.accumulate_gradients != 0
            loss = self.train_step(batch, accumulate_gradients=accumulate)
            epoch_losses.append(loss)
            
            # Optimizer step
            if not accumulate or batch_idx == total_batches - 1:
                if self.mixed_precision and self.scaler is not None:
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
            if progress_callback:
                progress_callback(epoch, batch_idx, loss, total_batches)
        
        return np.mean(epoch_losses)
    
    def train_model(self, training_data, num_epochs=10, progress_callback=None, status_callback=None):
        """Complete training pipeline."""
        if status_callback:
            status_callback("Preparing training data...")
        
        # Create dataloader
        dataloader = DataLoader(
            training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()
        )
        
        if status_callback:
            status_callback("Starting training...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            avg_loss = self.train_epoch(dataloader, epoch, progress_callback)
            
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update training history
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            })
            
            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint("checkpoints/best_model.pth", epoch)
            
            if status_callback:
                status_callback(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.6f}, Time={epoch_time:.1f}s")
        
        if status_callback:
            status_callback("Training completed successfully!")
    
    def save_checkpoint(self, filename, epoch):
        """Save training checkpoint."""
        os.makedirs("checkpoints", exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
    
    def get_memory_stats(self):
        """Get memory usage statistics."""
        stats = {'available': True}
        
        # System memory
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                stats['ram_percent'] = mem.percent
                stats['ram_available_gb'] = mem.available / (1024**3)
                stats['cpu_percent'] = psutil.cpu_percent()
            except:
                stats['ram_percent'] = 50.0
                stats['cpu_percent'] = 25.0
        else:
            stats['ram_percent'] = 50.0
            stats['cpu_percent'] = 25.0
        
        # GPU memory
        if torch.cuda.is_available():
            try:
                stats['vram_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
                stats['vram_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                stats['vram_percent'] = (stats['vram_allocated_gb'] / stats['vram_total_gb']) * 100
            except:
                stats['vram_allocated_gb'] = 0.0
                stats['vram_total_gb'] = 8.0
                stats['vram_percent'] = 0.0
        else:
            stats['vram_allocated_gb'] = 0.0
            stats['vram_total_gb'] = 0.0
            stats['vram_percent'] = 0.0
        
        return stats


class FinancialDataset(Dataset):
    """Optimized dataset for financial training."""
    
    def __init__(self, embeddings):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]


def collect_financial_data(progress_callback=None):
    """Collect and prepare financial training data."""
    if progress_callback:
        progress_callback(0.1, "Loading sample financial texts...")
    
    # Get sample financial texts
    texts = get_sample_financial_texts()
    text_pairs = get_sample_text_pairs()
    
    # Combine all texts
    all_texts = texts + [pair['draft'] for pair in text_pairs] + [pair['refined'] for pair in text_pairs]
    
    if progress_callback:
        progress_callback(0.5, f"Processing {len(all_texts)} financial texts...")
    
    # Create embeddings
    embedder = SimpleEmbedder(embedding_dim=384)
    embeddings = embedder.encode(all_texts)
    
    if progress_callback:
        progress_callback(1.0, f"Dataset ready: {len(embeddings)} samples")
    
    return {
        'embeddings': embeddings,
        'texts': all_texts,
        'count': len(embeddings)
    }


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'trainer' not in st.session_state:
        st.session_state.trainer = QuasarTrainer()
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False


def main():
    st.set_page_config(
        page_title="Quanta Quasar Trainer",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Header with Quanta branding
    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            st.image("attached_assets/QuantaLogo_1749842610909.png", width=100)
        except:
            st.markdown("**Q**")
    with col2:
        st.title("Quanta - Quasar Financial Diffusion Model Trainer")
        st.markdown("Production training system optimized for 62GB RAM environment")
    
    # Sidebar - System Information
    with st.sidebar:
        st.header("System Information")
        
        # GPU detection
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            st.success(f"GPU: {gpu_props.name}")
            st.info(f"VRAM: {gpu_props.total_memory / (1024**3):.1f} GB")
        else:
            st.warning("No GPU detected - using CPU")
        
        # RAM information
        if PSUTIL_AVAILABLE:
            ram_info = psutil.virtual_memory()
            st.info(f"RAM: {ram_info.total / (1024**3):.1f} GB")
            st.metric("RAM Usage", f"{ram_info.percent:.1f}%")
        
        st.markdown("**Optimized Settings:**")
        st.write("- Batch Size: 32 (with 62GB RAM)")
        st.write("- Mixed Precision: Enabled")
        st.write("- Gradient Accumulation: 2 steps")
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["🏋️ Training", "📊 Monitoring", "💾 Management"])
    
    with tab1:
        training_interface()
    
    with tab2:
        monitoring_interface()
    
    with tab3:
        management_interface()


def training_interface():
    """Main training interface."""
    st.header("🏋️ Quasar Model Training")
    
    trainer = st.session_state.trainer
    
    # Training configuration
    with st.expander("Training Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            embedding_dim = st.selectbox("Embedding Dimension", [256, 384, 512], index=1)
            num_steps = st.selectbox("Diffusion Steps", [50, 100, 200], index=1)
            hidden_dim = st.selectbox("Hidden Dimension", [256, 512, 768], index=1)
        
        with col2:
            num_epochs = st.slider("Training Epochs", 1, 50, 15)
            learning_rate = st.selectbox("Learning Rate", [1e-4, 5e-4, 1e-3], index=1)
    
    # Training controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚀 Start Training", disabled=st.session_state.is_training, use_container_width=True):
            start_training(embedding_dim, num_steps, hidden_dim, num_epochs, learning_rate)
    
    with col2:
        if st.button("🛑 Stop Training", disabled=not st.session_state.is_training):
            st.session_state.is_training = False
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.trainer = QuasarTrainer()
            st.session_state.training_data = None
            st.rerun()


def start_training(embedding_dim, num_steps, hidden_dim, num_epochs, learning_rate):
    """Start the training process."""
    st.session_state.is_training = True
    trainer = st.session_state.trainer
    
    # Progress tracking
    overall_progress = st.progress(0)
    status_text = st.empty()
    
    # Training metrics
    metrics_container = st.container()
    chart_container = st.container()
    
    try:
        # Collect data
        status_text.text("Collecting financial training data...")
        
        def data_progress(progress, message):
            overall_progress.progress(progress * 0.3)
            status_text.text(message)
        
        data_result = collect_financial_data(data_progress)
        dataset = FinancialDataset(data_result['embeddings'])
        
        # Create model
        status_text.text("Creating Quasar model...")
        model = trainer.create_model(embedding_dim, num_steps, hidden_dim)
        trainer.setup_optimizer(learning_rate)
        overall_progress.progress(0.4)
        
        # Training callbacks
        def progress_callback(epoch, batch_idx, loss, total_batches):
            epoch_prog = (epoch + 1) / num_epochs
            overall_progress.progress(0.4 + 0.6 * epoch_prog)
            
            # Update metrics in container
            with metrics_container:
                if trainer.training_history:
                    df = pd.DataFrame(trainer.training_history)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Epoch", f"{epoch+1}/{num_epochs}")
                    with col2:
                        st.metric("Current Loss", f"{loss:.6f}")
                    with col3:
                        st.metric("Best Loss", f"{trainer.best_loss:.6f}")
                    with col4:
                        memory_stats = trainer.get_memory_stats()
                        st.metric("VRAM Usage", f"{memory_stats.get('vram_percent', 0):.1f}%")
                    
                    # Training chart
                    if len(df) > 1:
                        with chart_container:
                            fig = px.line(df, x='epoch', y='loss', title='Training Progress')
                            st.plotly_chart(fig, use_container_width=True, key="live_training_chart")
        
        def status_callback(message):
            status_text.text(message)
        
        # Start training
        trainer.train_model(
            dataset,
            num_epochs=num_epochs,
            progress_callback=progress_callback,
            status_callback=status_callback
        )
        
        overall_progress.progress(1.0)
        st.success("🎉 Training completed successfully!")
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
    finally:
        st.session_state.is_training = False


def monitoring_interface():
    """Real-time training monitoring."""
    st.header("📊 Training Monitoring")
    
    trainer = st.session_state.trainer
    
    if trainer.training_history:
        df = pd.DataFrame(trainer.training_history)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Epochs", len(df))
        with col2:
            st.metric("Current Loss", f"{df['loss'].iloc[-1]:.6f}")
        with col3:
            st.metric("Best Loss", f"{df['loss'].min():.6f}")
        with col4:
            st.metric("Learning Rate", f"{df['learning_rate'].iloc[-1]:.6f}")
        
        # Training charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(df, x='epoch', y='loss', title='Training Loss')
            st.plotly_chart(fig, use_container_width=True, key="monitor_loss_chart")
        
        with col2:
            fig = px.line(df, x='epoch', y='learning_rate', title='Learning Rate Schedule')
            st.plotly_chart(fig, use_container_width=True, key="monitor_lr_chart")
        
        # Memory usage
        memory_stats = trainer.get_memory_stats()
        if memory_stats['available']:
            st.subheader("Hardware Usage")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RAM Usage", f"{memory_stats.get('ram_percent', 0):.1f}%")
            with col2:
                st.metric("CPU Usage", f"{memory_stats.get('cpu_percent', 0):.1f}%")
            with col3:
                st.metric("VRAM Usage", f"{memory_stats.get('vram_percent', 0):.1f}%")
    
    else:
        st.info("No training data available. Start training to see monitoring data.")


def management_interface():
    """Model management interface."""
    st.header("💾 Model Management")
    
    trainer = st.session_state.trainer
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Save Model")
        model_name = st.text_input("Model Name", "quasar_financial_model")
        
        if st.button("💾 Save Model"):
            if trainer.model is not None:
                filename = f"checkpoints/{model_name}.pth"
                trainer.save_checkpoint(filename, len(trainer.training_history))
                st.success(f"Model saved as {filename}")
            else:
                st.error("No model to save. Train a model first.")
    
    with col2:
        st.subheader("Model Information")
        if trainer.model is not None:
            total_params = sum(p.numel() for p in trainer.model.parameters())
            model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
            
            st.info(f"**Model Parameters:** {total_params:,}")
            st.info(f"**Model Size:** {model_size_mb:.1f} MB")
            st.info(f"**Training History:** {len(trainer.training_history)} epochs")
        else:
            st.warning("No model loaded")
    
    # Training history export
    if trainer.training_history:
        st.subheader("Export Training Data")
        df = pd.DataFrame(trainer.training_history)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Training History",
            data=csv,
            file_name="quasar_training_history.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()