
#!/usr/bin/env python3
"""
Quick command to run the quantum financial diffusion model training.
This bypasses Streamlit completely and runs pure PyTorch training.
"""

import subprocess
import sys

def run_training():
    """Run the quantum training directly."""
    print("🚀 Starting Quantum Financial Diffusion Training...")
    print("This is the REAL Feynman Path Integral implementation.")
    print("-" * 60)
    
    # Run the training script
    try:
        result = subprocess.run([sys.executable, "quasar_production.py"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
        else:
            print(f"\n❌ Training failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running training: {str(e)}")

if __name__ == "__main__":
    run_training()
