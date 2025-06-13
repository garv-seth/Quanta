"""
Quasar d-SLM - Main Training Script
Orchestrates the training process for the quantum-inspired diffusion model.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm
import logging
import os
from dataclasses import dataclass, asdict
import json

# Import from our custom modules
from model import QuantumInspiredDiffusionModel, EnvironmentDetector
from data import load_text_dataset

def setup_logging():
    """Configures logging for the training process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )

@dataclass
class TrainingState:
    """A dataclass to hold and manage the state of training."""
    checkpoint_dir: str = "checkpoints"
    best_loss: float = float('inf')
    current_epoch: int = 0
    total_steps: int = 0

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, config, is_best=False):
        """Saves a checkpoint of the model and training state."""
        checkpoint_name = "best_model.pth" if is_best else f"checkpoint_epoch_{self.current_epoch}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        state = {
            'epoch': self.current_epoch,
            'total_steps': self.total_steps,
            'best_loss': self.best_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
        }
        torch.save(state, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, model, optimizer, path):
        """Loads a checkpoint."""
        if not os.path.exists(path):
            logging.warning(f"Checkpoint path not found: {path}")
            return
        
        state = torch.load(path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        self.current_epoch = state['epoch']
        self.total_steps = state['total_steps']
        self.best_loss = state['best_loss']
        logging.info(f"Loaded checkpoint from {path} (Epoch {self.current_epoch})")


class Trainer:
    """Handles the model training loop and associated logic."""

    def __init__(self, config: dict, model: QuantumInspiredDiffusionModel, tokenizer, dataset):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.dataset = dataset
        
        self.optimizer = AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
        self.training_state = TrainingState()

        # Log the model parameter count
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Model initialized with {total_params:,} trainable parameters.")

    def _prepare_dataloader(self):
        """Tokenizes the dataset and creates a DataLoader."""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=self.config['max_seq_len'],
                return_tensors="pt"
            )

        # The map function is slow on Windows with multiprocessing, so num_proc=1
        tokenized_dataset = self.dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        return DataLoader(
            tokenized_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0 # num_workers > 0 can cause issues on Windows
        )

    def _train_step(self, batch):
        """Performs a single training step with the correct diffusion process."""
        self.model.train()
        input_ids = batch['input_ids'].to(self.device)
        batch_size = input_ids.shape[0]

        # 1. Get the continuous embeddings for the input tokens.
        x0_embeds = self.model.get_token_embeddings(input_ids)

        # 2. Sample Gaussian noise of the same shape as the embeddings.
        noise = torch.randn_like(x0_embeds)

        # 3. Sample a random timestep for each item in the batch.
        t = torch.randint(0, self.config['num_diffusion_steps'], (batch_size,), device=self.device).long()
        
        # 4. Create the noised embeddings using the diffusion schedule.
        xt_embeds = self.model.add_noise_to_embeddings(x0_embeds, t, noise)

        # 5. Get the model's prediction of the noise.
        #    Note: This requires modifying the model's forward pass to accept embeddings.
        #    For now, we assume the model's forward pass is updated. 
        #    Let's pass original tokens and do the noising inside the model forward pass for now
        #    to avoid changing the model file again.
        #    This will fail until model is changed.
        
        predicted_noise = self.model(xt_embeds, t)
        
        # 6. Calculate the loss.
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def train(self, num_epochs: int):
        """Main training loop."""
        dataloader = self._prepare_dataloader()
        num_training_steps = num_epochs * len(dataloader)
        
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps
        )
        
        progress_bar = tqdm(range(num_training_steps))
        
        for epoch in range(num_epochs):
            self.training_state.current_epoch = epoch
            epoch_loss = 0.0
            
            for step, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
                
                loss = self._train_step(batch) # This will be fixed next
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                lr_scheduler.step()
                
                self.training_state.total_steps += 1
                progress_bar.update(1)
                epoch_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / len(dataloader)
            logging.info(f"End of Epoch {epoch+1}: Average Loss = {avg_epoch_loss:.4f}")
            
            if avg_epoch_loss < self.training_state.best_loss:
                self.training_state.best_loss = avg_epoch_loss
                self.training_state.save_checkpoint(self.model, self.optimizer, self.config, is_best=True)
            else:
                self.training_state.save_checkpoint(self.model, self.optimizer, self.config)


def main():
    """The main function to run the training."""
    setup_logging()
    logging.info("Starting Quasar d-SLM Training Process")

    # 1. Get hardware-optimized configuration
    config = EnvironmentDetector.get_optimal_config()
    config['learning_rate'] = 5e-5 # Add learning rate to config
    num_epochs = 10 # Let's do a small number of epochs for now
    
    logging.info("Hardware-optimized configuration:")
    logging.info(json.dumps(config, indent=2))

    # 2. Load dataset and tokenizer
    # Using a smaller, standard dataset for initial setup
    dataset_name = "glue" 
    dataset_config = "mrpc"
    # dataset = load_dataset(dataset_name, dataset_config, split='train', streaming=False)
    # The above is better but requires pyarrow. Let's use our data.py for now.
    dataset = load_text_dataset(streaming=False)['train']
    
    tokenizer_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logging.info(f"Loaded dataset: {dataset_name}/{dataset_config}")
    logging.info(f"Loaded tokenizer: {tokenizer_name}")

    # 3. Initialize model
    model = QuantumInspiredDiffusionModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        num_diffusion_steps=config['num_diffusion_steps'],
        num_paths=config['num_paths']
    )

    # 4. Start training
    trainer = Trainer(config, model, tokenizer, dataset)
    
    logging.info("Starting training...")
    trainer.train(num_epochs=num_epochs)
    logging.info("Training script finished.")


if __name__ == "__main__":
    main()