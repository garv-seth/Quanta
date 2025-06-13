"""
Quanta Quasar - Direct Financial Diffusion Model Training
Simplified approach that bypasses Streamlit/PyTorch compatibility issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import json
import math
import random
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Hardware detection
def detect_hardware():
    """Detect available hardware and optimize configuration."""
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 8,
        'model_dim': 384,
        'num_layers': 6,
        'num_heads': 8,
        'sequence_length': 256,
        'use_mixed_precision': torch.cuda.is_available(),
        'num_diffusion_steps': 1000
    }
    
    # Detect RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb >= 32:
            config.update({
                'batch_size': 16,
                'model_dim': 512,
                'num_layers': 8,
                'sequence_length': 384
            })
        if ram_gb >= 60:  # Replit environment
            config.update({
                'batch_size': 32,
                'model_dim': 768,
                'num_layers': 10,
                'sequence_length': 512
            })
    except:
        pass
    
    # GPU optimization
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        print(f"GPU: {props.name} ({vram_gb:.1f}GB VRAM)")
        
        if vram_gb >= 8:
            config['batch_size'] *= 2
            config['num_diffusion_steps'] = 1500
    
    return config

# Financial data generation
def generate_financial_data(num_samples=5000):
    """Generate comprehensive financial training data."""
    texts = []
    
    # SEC filing templates
    sec_templates = [
        "The Company's revenue increased by {pct}% year-over-year to ${amount} million in Q{quarter}.",
        "Operating expenses totaled ${amount} million, representing a {change} from the prior year period.",
        "Gross margin improved to {pct}% due to operational efficiency gains and cost optimization.",
        "Cash and cash equivalents of ${amount} million provide adequate liquidity for operations.",
        "The effective tax rate was {pct}%, compared to {old_pct}% in the previous year.",
        "Free cash flow of ${amount} million demonstrates strong operational performance.",
        "Depreciation and amortization expenses were ${amount} million for the period.",
        "Working capital increased by ${amount} million due to business expansion.",
        "Interest expense decreased to ${amount} million following debt restructuring.",
        "Capital expenditures of ${amount} million support growth initiatives."
    ]
    
    # Earnings call templates
    earnings_templates = [
        "Q{quarter} results exceeded expectations with {metric} growth of {pct}%.",
        "Our {segment} segment delivered strong performance with revenue up {pct}%.",
        "Market conditions remain favorable with {indicator} showing improvement.",
        "We continue to see momentum in {market} with growing demand.",
        "Margin expansion reflects our operational excellence initiatives.",
        "Integration of recent acquisitions is proceeding ahead of schedule.",
        "Investment in {technology} is generating positive returns.",
        "We remain confident in our full-year guidance of {target}%.",
        "Cash generation remains strong, supporting capital allocation strategy.",
        "Market share gains in {geography} continue to drive growth."
    ]
    
    # Financial news templates
    news_templates = [
        "{company} reported {period} earnings of ${eps} per share, {result} expectations.",
        "Shares of {company} {movement} {pct}% following {news} announcement.",
        "Federal Reserve decision to {action} rates impacts {sector} outlook.",
        "Market volatility continues as investors assess {event} implications.",
        "Analysts {action} {company} price target to ${target} citing {reason}.",
        "The {index} closed {direction} {pct}% led by {sector} stocks.",
        "Economic data showing {indicator} growth supports market sentiment.",
        "Merger activity accelerated with {company} announcing acquisition talks.",
        "Commodity prices {trend} on {factor} concerns affecting markets.",
        "Central bank divergence between regions affects currency trading."
    ]
    
    # Generate varied financial text
    for _ in range(num_samples):
        template_type = random.choice(['sec', 'earnings', 'news'])
        
        if template_type == 'sec':
            template = random.choice(sec_templates)
            text = template.format(
                pct=round(random.uniform(1, 25), 1),
                old_pct=round(random.uniform(1, 25), 1),
                amount=round(random.uniform(10, 5000), 1),
                quarter=random.choice([1, 2, 3, 4]),
                change=random.choice(["increase", "decrease"])
            )
        
        elif template_type == 'earnings':
            template = random.choice(earnings_templates)
            text = template.format(
                quarter=random.choice([1, 2, 3, 4]),
                metric=random.choice(["revenue", "earnings", "EBITDA"]),
                pct=round(random.uniform(2, 30), 1),
                segment=random.choice(["consumer", "enterprise", "international"]),
                indicator=random.choice(["demand", "pricing", "utilization"]),
                market=random.choice(["North America", "Europe", "Asia"]),
                technology=random.choice(["AI", "automation", "cloud"]),
                target=round(random.uniform(5, 20), 1),
                geography=random.choice(["China", "Europe", "Americas"])
            )
        
        else:  # news
            template = random.choice(news_templates)
            text = template.format(
                company=random.choice(["Apple", "Microsoft", "Tesla", "Amazon"]),
                period=random.choice(["Q1", "Q2", "Q3", "Q4"]),
                eps=round(random.uniform(0.5, 5.0), 2),
                result=random.choice(["exceeded", "met", "missed"]),
                movement=random.choice(["surged", "declined", "rose"]),
                pct=round(random.uniform(1, 15), 1),
                news=random.choice(["earnings", "product launch", "acquisition"]),
                action=random.choice(["raise", "lower", "maintain"]),
                target=round(random.uniform(100, 500)),
                reason=random.choice(["strong fundamentals", "market growth"]),
                index=random.choice(["S&P 500", "Nasdaq", "Dow Jones"]),
                direction=random.choice(["higher", "lower"]),
                sector=random.choice(["technology", "healthcare", "finance"]),
                event=random.choice(["earnings", "policy", "economic data"]),
                trend=random.choice(["rallied", "declined", "stabilized"]),
                factor=random.choice(["supply", "demand", "regulatory"])
            )
        
        texts.append(text)
    
    return texts

# Simple tokenizer
class FinancialTokenizer:
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        
    def build_vocab(self, texts):
        word_counts = {}
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Add special tokens
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        
        # Add most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.vocab_size - 4]:
            self.word_to_id[word] = len(self.word_to_id)
        
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
    
    def _tokenize(self, text):
        text = text.lower().replace('$', ' dollar ').replace('%', ' percent ')
        import re
        return re.findall(r'\b\w+\b|[.!?]', text)
    
    def encode(self, text, max_length=128):
        words = self._tokenize(text)
        token_ids = [2]  # START token
        
        for word in words[:max_length-2]:
            token_ids.append(self.word_to_id.get(word, 1))  # UNK if not found
        
        token_ids.append(3)  # END token
        
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(0)  # PAD
        
        return token_ids[:max_length]

# Diffusion model
class FinancialDiffusionModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 max_seq_len=256, num_diffusion_steps=1000):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_diffusion_steps = num_diffusion_steps
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
        # Noise schedule
        betas = self._cosine_beta_schedule(num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def forward(self, x, t):
        batch_size, seq_len = x.shape
        device = x.device
        
        # Embeddings
        token_emb = self.token_embedding(x)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        t_normalized = t.float().unsqueeze(-1) / self.num_diffusion_steps
        time_emb = self.time_embedding(t_normalized).unsqueeze(1)
        
        # Combine embeddings
        embeddings = token_emb + pos_emb + time_emb
        embeddings = self.dropout(embeddings)
        
        # Transformer
        output = self.transformer(embeddings)
        output = self.layer_norm(output)
        
        # Output projection
        logits = self.output_projection(output)
        
        return logits
    
    def compute_loss(self, x_start, t):
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Add noise to tokens (simplified approach)
        noise = torch.randn_like(x_start.float())
        
        # Get noise coefficients
        sqrt_alphas_cumprod_t = torch.gather(torch.sqrt(self.alphas_cumprod), 0, t).view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.gather(torch.sqrt(1.0 - self.alphas_cumprod), 0, t).view(-1, 1)
        
        # Create noisy version
        x_start_float = x_start.float()
        x_noisy = sqrt_alphas_cumprod_t * x_start_float + sqrt_one_minus_alphas_cumprod_t * noise
        x_noisy_discrete = torch.clamp(torch.round(x_noisy), 0, self.vocab_size - 1).long()
        
        # Predict original
        predicted_logits = self.forward(x_noisy_discrete, t)
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            predicted_logits.view(-1, self.vocab_size),
            x_start.view(-1),
            ignore_index=0  # Ignore padding
        )
        
        return loss

# Dataset
class FinancialDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.token_sequences = []
        for text in texts:
            tokens = tokenizer.encode(text, max_length)
            self.token_sequences.append(torch.tensor(tokens, dtype=torch.long))
    
    def __len__(self):
        return len(self.token_sequences)
    
    def __getitem__(self, idx):
        return self.token_sequences[idx]

# Training function
def train_model(config, dataset, num_epochs=50):
    print("Starting Quanta Quasar Financial Diffusion Model Training")
    print("=" * 60)
    
    device = config['device']
    print(f"Device: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Model dimension: {config['model_dim']}")
    print(f"Number of layers: {config['num_layers']}")
    print(f"Diffusion steps: {config['num_diffusion_steps']}")
    print()
    
    # Create model
    vocab_size = len(dataset.token_sequences[0].unique()) + 1000  # Estimate
    model = FinancialDiffusionModel(
        vocab_size=vocab_size,
        d_model=config['model_dim'],
        nhead=config['num_heads'],
        num_layers=config['num_layers'],
        max_seq_len=config['sequence_length'],
        num_diffusion_steps=config['num_diffusion_steps']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} parameters")
    print(f"Estimated model size: {total_params * 4 / (1024**2):.1f} MB")
    print()
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * 10)
    scaler = torch.cuda.amp.GradScaler() if config['use_mixed_precision'] else None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    # Training loop
    model.train()
    training_history = []
    best_loss = float('inf')
    
    print("Training started...")
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            batch_size = batch.shape[0]
            
            # Random timesteps
            t = torch.randint(0, config['num_diffusion_steps'], (batch_size,), device=device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if config['use_mixed_precision'] and scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = model.compute_loss(batch, t)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.compute_loss(batch, t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            epoch_losses.append(loss.item())
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        # Epoch statistics
        avg_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'learning_rate': current_lr,
            'epoch_time': epoch_time
        })
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, 'checkpoints/quasar_best.pth')
        
        total_time = time.time() - total_start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} completed:")
        print(f"  Average loss: {avg_loss:.6f}")
        print(f"  Epoch time: {epoch_time:.1f}s")
        print(f"  Total time: {total_time/60:.1f}m")
        print(f"  Learning rate: {current_lr:.8f}")
        print(f"  Best loss: {best_loss:.6f}")
        print("-" * 50)
    
    total_training_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_training_time/60:.1f} minutes")
    print(f"Final loss: {avg_loss:.6f}")
    print(f"Best loss: {best_loss:.6f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_history': training_history,
        'final_loss': avg_loss,
        'best_loss': best_loss
    }, 'checkpoints/quasar_final.pth')
    
    return model, training_history

def main():
    print("Quanta Quasar Financial Diffusion Model")
    print("Detecting hardware configuration...")
    
    # Detect hardware and configure
    config = detect_hardware()
    print(f"Configuration: {config}")
    print()
    
    # Generate training data
    print("Generating financial training data...")
    texts = generate_financial_data(num_samples=5000)
    print(f"Generated {len(texts)} financial text samples")
    
    # Build tokenizer
    print("Building financial vocabulary...")
    tokenizer = FinancialTokenizer(vocab_size=8000)
    tokenizer.build_vocab(texts)
    print(f"Vocabulary size: {len(tokenizer.word_to_id)}")
    
    # Create dataset
    print("Creating training dataset...")
    dataset = FinancialDataset(texts, tokenizer, config['sequence_length'])
    print(f"Dataset size: {len(dataset)} samples")
    print()
    
    # Train model
    model, history = train_model(config, dataset, num_epochs=50)
    
    print("Training completed successfully!")
    print("Model saved to checkpoints/quasar_final.pth")

if __name__ == "__main__":
    main()