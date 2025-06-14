
#!/usr/bin/env python3
"""
Verify the exact parameter count of the quantum diffusion model
"""

import torch
import torch.nn as nn
from collections import OrderedDict

def count_parameters_detailed():
    """Count parameters with detailed breakdown."""
    
    # Import the model
    from quasar_production import FeynmanPathIntegralDiffusionModel, QuantumFinancialTokenizer
    
    # Create a minimal tokenizer for vocab size
    tokenizer = QuantumFinancialTokenizer(vocab_size=348)  # From your logs
    
    # Create model with your exact config
    model = FeynmanPathIntegralDiffusionModel(
        vocab_size=348,
        d_model=384,
        nhead=6,
        num_layers=4,
        max_seq_len=192,
        num_diffusion_steps=500,
        num_quantum_paths=6
    )
    
    print("🔍 DETAILED PARAMETER BREAKDOWN")
    print("=" * 60)
    
    total_params = 0
    component_counts = OrderedDict()
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            params = module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                params += module.bias.numel()
            
            component_counts[name] = params
            total_params += params
            
            # Show major components
            if params > 10000:  # Only show significant components
                print(f"  {name:50} {params:>12,} parameters")
    
    print("=" * 60)
    print(f"🎯 TOTAL PARAMETERS: {total_params:,}")
    
    # Verify against PyTorch's built-in method
    pytorch_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ PyTorch verification: {pytorch_count:,}")
    print(f"✅ Trainable parameters: {trainable_count:,}")
    
    # Memory estimates
    model_size_mb = total_params * 4 / (1024**2)  # 4 bytes per float32
    print(f"💾 Model size: {model_size_mb:.1f} MB")
    
    # Break down by component type
    print("\n📊 BREAKDOWN BY COMPONENT TYPE:")
    embedding_params = sum(v for k, v in component_counts.items() if 'embedding' in k.lower())
    transformer_params = sum(v for k, v in component_counts.items() if 'transformer' in k.lower())
    other_params = total_params - embedding_params - transformer_params
    
    print(f"  🔤 Embeddings: {embedding_params:,} ({embedding_params/total_params*100:.1f}%)")
    print(f"  🔄 Transformers: {transformer_params:,} ({transformer_params/total_params*100:.1f}%)")
    print(f"  ⚙️ Other layers: {other_params:,} ({other_params/total_params*100:.1f}%)")
    
    # Validate if this matches your logged count
    logged_count = 43_747_014
    print(f"\n🎯 VALIDATION:")
    print(f"  Calculated: {total_params:,}")
    print(f"  Logged:     {logged_count:,}")
    print(f"  Match: {'✅ YES' if total_params == logged_count else '❌ NO'}")
    
    return total_params

if __name__ == "__main__":
    count_parameters_detailed()
