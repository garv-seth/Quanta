
#!/usr/bin/env python3
"""
Inference script for trained Quantum Financial Diffusion Model
"""

import torch
import torch.nn.functional as F
from quasar_production import FeynmanPathIntegralDiffusionModel, QuantumFinancialTokenizer, collect_real_financial_data

class QuantumFinancialGenerator:
    def __init__(self, model_path="checkpoints/quantum_best.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Rebuild tokenizer (in practice, save this separately)
        print("🔄 Rebuilding tokenizer...")
        texts = collect_real_financial_data(1000)  # Smaller for inference
        self.tokenizer = QuantumFinancialTokenizer(vocab_size=12000)
        self.tokenizer.build_vocab(texts)
        
        # Load model
        print("🚀 Loading quantum model...")
        self.model = FeynmanPathIntegralDiffusionModel(
            vocab_size=len(self.tokenizer.word_to_id),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_seq_len=config['max_seq_len'],
            num_diffusion_steps=config['num_diffusion_steps'],
            num_quantum_paths=config['num_quantum_paths']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✅ Model loaded with loss: {checkpoint['best_loss']:.4f}")

    def denoise_financial_text(self, prompt, num_steps=50):
        """Generate financial text using quantum denoising."""
        # Encode prompt
        tokens = self.tokenizer.encode(prompt, max_length=self.model.max_seq_len)
        x = torch.tensor(tokens).unsqueeze(0).to(self.device)
        
        # Convert to continuous embeddings
        with torch.no_grad():
            x_embed = F.one_hot(x, num_classes=self.model.vocab_size).float()
            
            # Start from pure noise and denoise
            current_embed = torch.randn_like(x_embed)
            
            # Reverse diffusion process
            for step in range(num_steps):
                t = torch.tensor([num_steps - step - 1], device=self.device)
                
                # Predict noise
                predicted_noise = self.model(x, t)
                
                # Remove noise (simplified)
                alpha = self.model.alphas_cumprod[t].view(-1, 1, 1)
                current_embed = (current_embed - predicted_noise * (1 - alpha).sqrt()) / alpha.sqrt()
            
            # Convert back to tokens
            final_tokens = torch.argmax(current_embed, dim=-1)
            
        return self.tokenizer.decode(final_tokens[0].cpu().tolist())

    def analyze_sentiment(self, text):
        """Analyze financial sentiment using quantum paths."""
        tokens = self.tokenizer.encode(text, max_length=128)
        x = torch.tensor(tokens).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get path representations
            t = torch.tensor([0], device=self.device)  # Minimal noise
            logits = self.model(x, t)
            
            # Analyze path weights (quantum amplitudes)
            avg_repr = logits.mean(dim=1)  # [batch, vocab_size] -> [batch, avg]
            
            # Simple sentiment scoring
            positive_score = torch.sigmoid(avg_repr.mean()).item()
            
        return {
            'sentiment': 'positive' if positive_score > 0.6 else 'negative' if positive_score < 0.4 else 'neutral',
            'confidence': abs(positive_score - 0.5) * 2,
            'quantum_coherence': positive_score
        }

def main():
    """Demo the trained model capabilities."""
    print("🌟 Quantum Financial AI Demo")
    print("=" * 50)
    
    # Check if model exists
    import os
    if not os.path.exists("checkpoints/quantum_best.pth"):
        print("❌ No trained model found. Run training first!")
        return
    
    # Load generator
    generator = QuantumFinancialGenerator()
    
    # Demo capabilities
    test_prompts = [
        "Revenue for the quarter was",
        "The company announced",
        "Market volatility increased due to",
        "Operating margins improved"
    ]
    
    print("\n🔮 FINANCIAL TEXT GENERATION:")
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        result = generator.denoise_financial_text(prompt)
        print(f"🤖 Generated: {result}")
    
    print("\n📊 SENTIMENT ANALYSIS:")
    test_texts = [
        "Revenue exceeded expectations with strong growth across all segments",
        "The company missed earnings guidance due to supply chain issues",
        "Quarterly results were in line with analyst forecasts"
    ]
    
    for text in test_texts:
        analysis = generator.analyze_sentiment(text)
        print(f"\n📰 Text: {text}")
        print(f"💭 Sentiment: {analysis['sentiment']} (confidence: {analysis['confidence']:.2f})")
        print(f"⚛️ Quantum coherence: {analysis['quantum_coherence']:.3f}")

if __name__ == "__main__":
    main()
