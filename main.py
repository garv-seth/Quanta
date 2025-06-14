#!/usr/bin/env python3
"""
Standalone Quantum Financial Diffusion Model Training Script
Runs the production Feynman Path Integral implementation without Streamlit
"""

import sys
import os
import logging
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantum_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main training function without Streamlit dependencies."""
    logger.info("🚀 Starting Standalone Quantum Financial Training")
    logger.info("=" * 80)

    try:
        # Import and run the quantum training directly
        from quasar_production import main as run_quantum_training

        logger.info("✅ Successfully imported quantum training module")
        logger.info("🔄 Starting Feynman Path Integral training process...")

        # Run the actual training
        run_quantum_training()

        logger.info("🎉 Training completed successfully!")

    except ImportError as e:
        logger.error(f"❌ Failed to import training module: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()