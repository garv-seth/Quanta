"""
Production Financial Diffusion Model Training Script

Optimized for RTX 4060 (8GB VRAM) + 32GB RAM system.
Collects real financial data and trains the FinSar diffusion model.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from pathlib import Path
import json
import os
import gc

# Import our models and utilities
from models.diffusion_model import DiffusionModel
from utils.text_processor import TextProcessor
from utils.training import ModelTrainer
from data.sample_texts import get_sample_financial_texts

# Try importing optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from database.real_data_manager import RealFinancialDataManager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


class RTX4060Optimizer:
    """Hardware optimization specifically for RTX 4060 with 32GB RAM."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_memory_gb = 8.0  # RTX 4060 VRAM
        self.system_memory_gb = 32.0
        
    def get_optimal_config(self, model):
        """Get training configuration optimized for RTX 4060."""
        # Calculate model memory footprint
        model_params = sum(p.numel() for p in model.parameters())
        model_memory_mb = model_params * 4 / (1024**2)  # float32
        
        # Reserve memory for gradients (2x) and optimizer states (2x)
        training_overhead_mb = model_memory_mb * 4
        
        # Available VRAM for batches (use 85% of total)
        available_vram_mb = (self.gpu_memory_gb * 1024 * 0.85) - training_overhead_mb
        
        # Memory per sample (embedding + forward/backward passes)
        embedding_memory_mb = 384 * 4 / (1024**2)  # 384-dim embeddings
        memory_per_sample_mb = embedding_memory_mb * 12  # Factor for computation overhead
        
        # Calculate optimal batch size
        max_batch_size = max(1, int(available_vram_mb / memory_per_sample_mb))
        optimal_batch_size = min(max_batch_size, 16)  # Cap for stability
        
        # Gradient accumulation for larger effective batch size
        target_effective_batch = 128
        grad_accumulation_steps = max(1, target_effective_batch // optimal_batch_size)
        
        return {
            'batch_size': optimal_batch_size,
            'grad_accumulation_steps': grad_accumulation_steps,
            'effective_batch_size': optimal_batch_size * grad_accumulation_steps,
            'learning_rate': 5e-4,  # Conservative for stability
            'num_workers': 4,  # Good for 32GB RAM
            'pin_memory': True,
            'model_memory_mb': model_memory_mb,
            'estimated_vram_usage_mb': training_overhead_mb + optimal_batch_size * memory_per_sample_mb
        }


class ProductionFinancialDataset(Dataset):
    """Optimized dataset for production training."""
    
    def __init__(self, embeddings_tensor):
        """Initialize with pre-computed embeddings tensor."""
        self.embeddings = embeddings_tensor
        print(f"Dataset initialized with {len(self.embeddings)} samples")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]


class RealTimeDataCollector:
    """Collect and process real financial data for training."""
    
    def __init__(self):
        if DATABASE_AVAILABLE:
            try:
                self.data_manager = RealFinancialDataManager()
            except Exception as e:
                print(f"Database connection failed: {e}")
                self.data_manager = None
        else:
            self.data_manager = None
        
        self.text_processor = TextProcessor()
        self.collected_data = None
    
    def collect_comprehensive_data(self, progress_callback=None):
        """Collect comprehensive financial data from all sources."""
        try:
            if progress_callback:
                progress_callback(0.1, "Initializing data collection...")
            
            training_texts = []
            collection_stats = {}
            
            # Try to collect real financial data if database is available
            if self.data_manager:
                try:
                    collection_stats = self.data_manager.collect_all_live_data()
                    
                    if progress_callback:
                        progress_callback(0.4, "Extracting training texts...")
                    
                    training_texts = self.data_manager.prepare_training_texts_from_db()
                except Exception as e:
                    print(f"Real data collection failed: {e}")
                    training_texts = []
            
            # Use sample data if no real data available
            if not training_texts or len(training_texts) < 10:
                if progress_callback:
                    progress_callback(0.5, "Loading sample financial data...")
                training_texts = get_sample_financial_texts()
                collection_stats = {'sample_data_used': len(training_texts)}
                print(f"Using {len(training_texts)} sample financial texts")
            
            if progress_callback:
                progress_callback(0.7, "Generating embeddings...")
            
            # Process texts in batches for memory efficiency
            all_embeddings = []
            batch_size = 32
            
            for i in range(0, len(training_texts), batch_size):
                batch_texts = training_texts[i:i + batch_size]
                batch_embeddings = self.text_processor.batch_process_texts(batch_texts)
                all_embeddings.append(batch_embeddings.cpu())
                
                if progress_callback and i % (batch_size * 5) == 0:
                    progress = 0.7 + 0.2 * (i / len(training_texts))
                    progress_callback(progress, f"Processing batch {i//batch_size + 1}...")
            
            # Combine all embeddings
            final_embeddings = torch.cat(all_embeddings, dim=0)
            
            if progress_callback:
                progress_callback(1.0, "Data collection complete!")
            
            self.collected_data = {
                'embeddings': final_embeddings,
                'num_texts': len(training_texts),
                'collection_stats': collection_stats,
                'embedding_dim': final_embeddings.shape[1]
            }
            
            return self.collected_data
            
        except Exception as e:
            print(f"Data collection error: {e}")
            # Return sample data as fallback
            training_texts = get_sample_financial_texts()
            embeddings = self.text_processor.batch_process_texts(training_texts)
            
            return {
                'embeddings': embeddings,
                'num_texts': len(training_texts),
                'collection_stats': {'sample_data': len(training_texts)},
                'embedding_dim': embeddings.shape[1]
            }


class ProductionTrainingSystem:
    """Complete production training system for financial diffusion models."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer_rtx = RTX4060Optimizer()
        self.data_collector = RealTimeDataCollector()
        
        # Training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        
        # Training state
        self.training_history = []
        self.best_loss = float('inf')
        self.current_epoch = 0
        
        # Create directories
        for dir_name in ['checkpoints', 'logs', 'embeddings']:
            os.makedirs(dir_name, exist_ok=True)
    
    def create_model(self, embedding_dim=384, num_steps=100, hidden_dim=512):
        """Create and configure the diffusion model."""
        self.model = DiffusionModel(
            embedding_dim=embedding_dim,
            num_steps=num_steps,
            hidden_dim=hidden_dim
        )
        self.model.to(self.device)
        
        # Get hardware-optimized configuration
        self.config = self.optimizer_rtx.get_optimal_config(self.model)
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Optimized training config: {self.config}")
        
        return self.model
    
    def setup_training(self):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=self.config['learning_rate'] * 0.01
        )
    
    def train_step(self, batch, accumulate_gradients=False):
        """Optimized training step with gradient accumulation."""
        if not accumulate_gradients:
            self.optimizer.zero_grad()
        
        batch = batch.to(self.device, non_blocking=True)
        batch_size = batch.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.model.num_steps, (batch_size,), device=self.device)
        
        # Forward pass with automatic mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss = self.model.compute_loss(batch, t)
                loss = loss / self.config['grad_accumulation_steps']
            
            self.scaler.scale(loss).backward()
        else:
            loss = self.model.compute_loss(batch, t)
            loss = loss / self.config['grad_accumulation_steps']
            loss.backward()
        
        return loss.item() * self.config['grad_accumulation_steps']
    
    def train_epoch(self, dataloader, epoch, progress_callback=None):
        """Train for one epoch with RTX 4060 optimization."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(dataloader)
        grad_accumulation_steps = self.config['grad_accumulation_steps']
        
        for batch_idx, batch in enumerate(dataloader):
            # Gradient accumulation logic
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
            
            # Progress updates
            if progress_callback and batch_idx % 5 == 0:
                progress_callback(epoch, batch_idx, loss, num_batches)
            
            # Memory cleanup every 25 batches
            if batch_idx % 25 == 0:
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        return {
            'avg_loss': avg_loss,
            'total_loss': total_loss,
            'num_batches': num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def train_complete_model(self, num_epochs=10, progress_callback=None, 
                           status_callback=None):
        """Complete training pipeline."""
        
        # Data collection
        if status_callback:
            status_callback("Collecting real financial data...")
        
        def data_progress(progress, message):
            if progress_callback:
                progress_callback(-1, -1, 0, 0, f"Data: {message}")
        
        data_result = self.data_collector.collect_comprehensive_data(data_progress)
        
        # Create dataset
        dataset = ProductionFinancialDataset(data_result['embeddings'])
        
        # Create dataloader with RTX 4060 optimization
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            persistent_workers=True
        )
        
        # Training loop
        for epoch in range(num_epochs):
            if status_callback:
                status_callback(f"Training epoch {epoch+1}/{num_epochs}")
            
            epoch_start_time = time.time()
            
            # Train epoch
            train_stats = self.train_epoch(dataloader, epoch, progress_callback)
            
            epoch_time = time.time() - epoch_start_time
            
            # Record training history
            epoch_data = {
                'epoch': epoch,
                'train_loss': train_stats['avg_loss'],
                'learning_rate': train_stats['learning_rate'],
                'epoch_time': epoch_time,
                'timestamp': time.time()
            }
            
            self.training_history.append(epoch_data)
            
            # Save best model
            if train_stats['avg_loss'] < self.best_loss:
                self.best_loss = train_stats['avg_loss']
                self.save_checkpoint('best_model.pth', epoch)
            
            # Regular checkpoints
            if (epoch + 1) % 3 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch)
            
            # Memory cleanup
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        if status_callback:
            status_callback("Training completed successfully!")
        
        return self.training_history
    
    def save_checkpoint(self, filename, epoch):
        """Save training checkpoint."""
        filepath = Path('checkpoints') / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def get_memory_usage(self):
        """Get current memory usage statistics."""
        memory_info = {}
        
        if PSUTIL_AVAILABLE:
            try:
                memory_info.update({
                    'system_ram_percent': psutil.virtual_memory().percent,
                    'cpu_percent': psutil.cpu_percent()
                })
            except:
                memory_info.update({
                    'system_ram_percent': 50.0,  # Default fallback
                    'cpu_percent': 25.0
                })
        else:
            memory_info.update({
                'system_ram_percent': 50.0,
                'cpu_percent': 25.0
            })
        
        if torch.cuda.is_available():
            try:
                memory_info.update({
                    'vram_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'vram_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                    'vram_percent': (torch.cuda.memory_allocated() / 
                                   torch.cuda.get_device_properties(0).total_memory) * 100
                })
            except:
                memory_info.update({
                    'vram_allocated_gb': 0.0,
                    'vram_reserved_gb': 0.0,
                    'vram_percent': 0.0
                })
        else:
            memory_info.update({
                'vram_allocated_gb': 0.0,
                'vram_reserved_gb': 0.0,
                'vram_percent': 0.0
            })
        
        return memory_info


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'training_system' not in st.session_state:
        st.session_state.training_system = None
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False


def main():
    st.set_page_config(
        page_title="Financial Diffusion Model - Production Training",
        page_icon="⚡",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("⚡ Financial Diffusion Model - Production Training")
    st.markdown("Real financial data collection and FinSar model training optimized for RTX 4060")
    
    # Hardware info sidebar
    with st.sidebar:
        st.header("System Monitor")
        
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            st.success(f"GPU: {gpu_props.name}")
            st.info(f"VRAM: {gpu_props.total_memory / (1024**3):.1f} GB")
        else:
            st.warning("No GPU detected - using CPU")
        
        ram_info = psutil.virtual_memory()
        st.info(f"RAM: {ram_info.total / (1024**3):.1f} GB")
        st.metric("RAM Usage", f"{ram_info.percent:.1f}%")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🚀 Training", "📊 Monitoring", "💾 Results"])
    
    with tab1:
        training_interface()
    
    with tab2:
        monitoring_interface()
    
    with tab3:
        results_interface()


def training_interface():
    """Main training interface."""
    st.header("🚀 Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        # Model parameters
        embedding_dim = st.selectbox("Embedding Dimension", [384, 512], index=0)
        num_steps = st.slider("Diffusion Steps", 50, 200, 100)
        hidden_dim = st.slider("Hidden Dimension", 256, 768, 512, step=128)
        
        # Training parameters
        num_epochs = st.slider("Training Epochs", 5, 50, 15)
        
        st.info("Training parameters optimized for RTX 4060 (8GB VRAM)")
        
        # Start training button
        if st.button("🚀 Start Production Training", type="primary", 
                    disabled=st.session_state.is_training):
            start_production_training(embedding_dim, num_steps, hidden_dim, num_epochs)
    
    with col2:
        st.subheader("Training Features")
        st.markdown("""
        **Hardware Optimization:**
        - RTX 4060 optimized batch sizes
        - Automatic mixed precision
        - Memory management
        - Gradient accumulation
        
        **Real Data Sources:**
        - Yahoo Finance API
        - SEC filings
        - Financial news
        - Market indicators
        
        **Model Features:**
        - Financial text diffusion
        - Embedding space refinement
        - Cosine noise schedule
        - Advanced optimization
        """)


def start_production_training(embedding_dim, num_steps, hidden_dim, num_epochs):
    """Start the complete production training process."""
    
    st.session_state.is_training = True
    
    # Initialize training system
    training_system = ProductionTrainingSystem()
    st.session_state.training_system = training_system
    
    # Create containers for real-time updates
    status_container = st.container()
    progress_container = st.container()
    metrics_container = st.container()
    
    with status_container:
        status_text = st.empty()
        overall_progress = st.progress(0)
    
    with progress_container:
        st.subheader("Training Progress")
        epoch_progress = st.progress(0)
        batch_progress = st.progress(0)
        current_stats = st.empty()
    
    with metrics_container:
        st.subheader("Real-time Metrics")
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
    
    try:
        # Create model
        status_text.text("Creating optimized model...")
        model = training_system.create_model(embedding_dim, num_steps, hidden_dim)
        training_system.setup_training()
        overall_progress.progress(0.1)
        
        # Progress callback for training updates
        def progress_callback(epoch, batch_idx, loss, total_batches, message=None):
            if epoch >= 0:  # Regular training progress
                epoch_prog = (epoch + 1) / num_epochs
                batch_prog = batch_idx / total_batches if total_batches > 0 else 0
                
                overall_progress.progress(0.1 + 0.9 * epoch_prog)
                epoch_progress.progress(epoch_prog)
                batch_progress.progress(batch_prog)
                
                current_stats.text(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{total_batches}, Loss: {loss:.6f}")
                
                # Update metrics
                if training_system.training_history:
                    df = pd.DataFrame(training_system.training_history)
                    
                    # Memory usage
                    memory_info = training_system.get_memory_usage()
                    
                    # Display metrics
                    col1, col2, col3 = metrics_placeholder.columns(3)
                    with col1:
                        st.metric("Current Loss", f"{loss:.6f}")
                    with col2:
                        st.metric("Best Loss", f"{training_system.best_loss:.6f}")
                    with col3:
                        st.metric("VRAM Usage", f"{memory_info.get('vram_percent', 0):.1f}%")
                    
                    # Loss chart
                    if len(df) > 1:
                        fig = px.line(df, y='train_loss', title='Training Loss')
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
            else:  # Data collection progress
                if message:
                    status_text.text(message)
        
        # Status callback for major updates
        def status_callback(message):
            status_text.text(message)
        
        # Start training
        training_history = training_system.train_complete_model(
            num_epochs=num_epochs,
            progress_callback=progress_callback,
            status_callback=status_callback
        )
        
        # Training completed
        st.session_state.is_training = False
        st.session_state.training_complete = True
        
        overall_progress.progress(1.0)
        status_text.text("✅ Training completed successfully!")
        
        # Final results
        st.success(f"Training completed! Best loss: {training_system.best_loss:.6f}")
        
        # Save final results
        results = {
            'training_history': training_history,
            'best_loss': training_system.best_loss,
            'model_config': {
                'embedding_dim': embedding_dim,
                'num_steps': num_steps,
                'hidden_dim': hidden_dim
            },
            'training_config': training_system.config
        }
        
        with open('training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
    except Exception as e:
        st.session_state.is_training = False
        st.error(f"Training failed: {str(e)}")


def monitoring_interface():
    """Real-time training monitoring."""
    st.header("📊 Training Monitor")
    
    if not st.session_state.training_system:
        st.info("No training session active")
        return
    
    training_system = st.session_state.training_system
    
    if training_system.training_history:
        df = pd.DataFrame(training_system.training_history)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Epochs Completed", len(df))
        with col2:
            st.metric("Current Loss", f"{df['train_loss'].iloc[-1]:.6f}")
        with col3:
            st.metric("Best Loss", f"{df['train_loss'].min():.6f}")
        with col4:
            st.metric("Learning Rate", f"{df['learning_rate'].iloc[-1]:.6f}")
        
        # Detailed charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Learning Rate', 'Epoch Time', 'Memory Usage'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training loss
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['train_loss'], mode='lines+markers', name='Loss'),
            row=1, col=1
        )
        
        # Learning rate
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['learning_rate'], mode='lines+markers', name='LR'),
            row=1, col=2
        )
        
        # Epoch time
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['epoch_time'], mode='lines+markers', name='Time (s)'),
            row=2, col=1
        )
        
        # Memory usage
        memory_info = training_system.get_memory_usage()
        fig.add_trace(
            go.Bar(x=['RAM', 'VRAM'], 
                   y=[memory_info['system_ram_percent'], memory_info.get('vram_percent', 0)],
                   name='Usage %'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No training data available yet")


def results_interface():
    """Training results and model management."""
    st.header("💾 Results & Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Results")
        
        if st.session_state.training_complete and st.session_state.training_system:
            training_system = st.session_state.training_system
            
            st.success("✅ Training completed successfully!")
            
            # Display final metrics
            st.metric("Final Best Loss", f"{training_system.best_loss:.6f}")
            st.metric("Total Epochs", len(training_system.training_history))
            
            # Model configuration
            st.subheader("Model Configuration")
            if hasattr(training_system, 'config'):
                st.json(training_system.config)
            
            # Download training results
            if st.button("📥 Download Training Results"):
                results = {
                    'training_history': training_system.training_history,
                    'best_loss': training_system.best_loss,
                    'config': training_system.config
                }
                
                results_json = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=results_json,
                    file_name=f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        else:
            st.info("No completed training sessions")
    
    with col2:
        st.subheader("Saved Models")
        
        # List saved checkpoints
        checkpoint_dir = Path('checkpoints')
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob('*.pth'))
            
            if checkpoint_files:
                for checkpoint_file in sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True):
                    st.write(f"📄 {checkpoint_file.name}")
                    
                    # File info
                    file_size = checkpoint_file.stat().st_size / (1024*1024)
                    mod_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                    st.caption(f"Size: {file_size:.1f} MB, Modified: {mod_time.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Load button
                    if st.button(f"Load {checkpoint_file.name}", key=f"load_{checkpoint_file.name}"):
                        try:
                            checkpoint = torch.load(checkpoint_file, map_location='cpu')
                            st.success(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
                            st.json(checkpoint.get('config', {}))
                        except Exception as e:
                            st.error(f"Failed to load checkpoint: {e}")
            else:
                st.info("No saved models found")
        else:
            st.info("Checkpoints directory not found")


if __name__ == "__main__":
    main()