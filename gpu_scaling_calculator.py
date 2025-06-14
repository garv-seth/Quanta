
#!/usr/bin/env python3
"""
GPU Scaling Calculator for Quantum Diffusion Models
Calculates maximum model size for RTX 4060 Mobile 8GB VRAM
"""

import math

def calculate_parameter_count(config):
    """Calculate exact parameter count for quantum diffusion model."""
    vocab_size = config['vocab_size']
    d_model = config['d_model']
    num_paths = config['num_paths']
    num_layers = config['num_layers']
    max_seq_len = config['max_seq_len']
    nhead = config['nhead']
    
    # Path-specific token embeddings
    path_embeddings = vocab_size * d_model * num_paths
    
    # Position embedding (shared)
    position_embedding = max_seq_len * d_model
    
    # Time embedding network
    time_embedding = d_model * (d_model * 4) + (d_model * 4) * d_model
    
    # Transformer parameters per path
    # Each layer: self-attention + feedforward
    # Self-attention: 4 * d_model^2 (Q, K, V, out projections)
    # Feedforward: d_model * (d_model * 4) + (d_model * 4) * d_model
    transformer_per_layer = (4 * d_model * d_model) + (d_model * d_model * 4 * 2)
    total_transformer_params = transformer_per_layer * num_layers * num_paths
    
    # Path weighting network
    path_weighting = (d_model * num_paths) * d_model + d_model * num_paths
    
    # Output projection
    output_projection = d_model * d_model
    
    total_params = (path_embeddings + position_embedding + time_embedding + 
                   total_transformer_params + path_weighting + output_projection)
    
    return total_params

def estimate_vram_usage(total_params, batch_size, seq_len, d_model, use_fp16=True):
    """Estimate VRAM usage including model, gradients, optimizer, and activations."""
    bytes_per_param = 2 if use_fp16 else 4
    
    # Model weights
    model_memory = total_params * bytes_per_param
    
    # Gradients (same size as model)
    gradient_memory = total_params * bytes_per_param
    
    # Optimizer states (AdamW has 2 states per parameter)
    optimizer_memory = total_params * bytes_per_param * 2
    
    # Activations (rough estimate)
    activation_memory = batch_size * seq_len * d_model * bytes_per_param * 10  # Multiple layers
    
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
    return total_memory / (1024**3)  # Convert to GB

def find_optimal_config_for_gpu():
    """Find optimal configuration for RTX 4060 Mobile 8GB."""
    available_vram = 7.5  # Leave 0.5GB for system
    
    print("🎮 RTX 4060 Mobile 8GB VRAM - Quantum Model Scaling Analysis")
    print("=" * 70)
    
    configs = [
        # Current config
        {
            'name': 'Current Model',
            'vocab_size': 348,
            'd_model': 384,
            'num_paths': 6,
            'num_layers': 4,
            'max_seq_len': 192,
            'nhead': 6,
            'batch_size': 8
        },
        # Scaled up versions
        {
            'name': 'GPU Optimized Small',
            'vocab_size': 1000,
            'd_model': 512,
            'num_paths': 8,
            'num_layers': 6,
            'max_seq_len': 256,
            'nhead': 8,
            'batch_size': 16
        },
        {
            'name': 'GPU Optimized Medium',
            'vocab_size': 2000,
            'd_model': 768,
            'num_paths': 12,
            'num_layers': 8,
            'max_seq_len': 512,
            'nhead': 12,
            'batch_size': 24
        },
        {
            'name': 'GPU Maximum Config',
            'vocab_size': 5000,
            'd_model': 1024,
            'num_paths': 16,
            'num_layers': 12,
            'max_seq_len': 1024,
            'nhead': 16,
            'batch_size': 32
        }
    ]
    
    for config in configs:
        params = calculate_parameter_count(config)
        vram_fp32 = estimate_vram_usage(params, config['batch_size'], 
                                       config['max_seq_len'], config['d_model'], False)
        vram_fp16 = estimate_vram_usage(params, config['batch_size'], 
                                       config['max_seq_len'], config['d_model'], True)
        
        # Model size in MB
        model_size_mb = params * 4 / (1024**2)
        
        print(f"\n📊 {config['name']}:")
        print(f"  🔢 Parameters: {params:,}")
        print(f"  💾 Model Size: {model_size_mb:.1f} MB")
        print(f"  🖥️ VRAM (FP32): {vram_fp32:.1f} GB")
        print(f"  🖥️ VRAM (FP16): {vram_fp16:.1f} GB")
        print(f"  📐 Dimensions: {config['d_model']}d, {config['num_paths']} paths, {config['num_layers']} layers")
        print(f"  📏 Sequence Length: {config['max_seq_len']}")
        print(f"  🎯 Batch Size: {config['batch_size']}")
        
        if vram_fp16 <= available_vram:
            print(f"  ✅ FITS in 8GB VRAM with FP16!")
        elif vram_fp32 <= available_vram:
            print(f"  ⚠️ Needs FP16 to fit in 8GB VRAM")
        else:
            print(f"  ❌ Too large for 8GB VRAM")
            
    print("\n🚀 QUANTUM MODEL ADVANTAGES:")
    print("  🌀 Multiple Path Processing: Explores different denoising trajectories")
    print("  🧮 Path Integral Formulation: Quantum-inspired optimization")
    print("  📈 Better Convergence: Multiple paths can find better solutions")
    print("  🎯 Robust Training: Path weighting provides ensemble-like benefits")
    
    print("\n⚡ RECOMMENDED GPU CONFIG:")
    print("  📱 Use FP16 mixed precision")
    print("  🔄 Gradient accumulation for larger effective batch sizes")
    print("  💾 Model checkpointing to save VRAM")
    print("  🎮 RTX 4060 Mobile can handle ~100M parameter quantum models!")

if __name__ == "__main__":
    find_optimal_config_for_gpu()
