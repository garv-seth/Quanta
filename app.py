import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime

from models.diffusion_model import DiffusionModel
from utils.text_processor import TextProcessor
from utils.training import ModelTrainer
from utils.evaluation import ModelEvaluator

# Set page configuration
st.set_page_config(
    page_title="Financial Text Diffusion Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'text_processor' not in st.session_state or st.session_state.text_processor is None:
    st.session_state.text_processor = TextProcessor()
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'evaluator' not in st.session_state or st.session_state.evaluator is None:
    st.session_state.evaluator = ModelEvaluator()
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

def main():
    st.title("🏦 Financial Text Diffusion Model (dLLM)")
    st.markdown("---")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Text Refinement", "Model Training", "Model Evaluation", "Model Management"]
        )
        
        st.header("Model Status")
        if st.session_state.model is not None:
            st.success("✅ Model Loaded")
            st.info(f"Embedding Dim: {st.session_state.model.embedding_dim}")
            st.info(f"Diffusion Steps: {st.session_state.model.num_steps}")
        else:
            st.warning("⚠️ No Model Loaded")
    
    if page == "Text Refinement":
        text_refinement_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Model Evaluation":
        model_evaluation_page()
    elif page == "Model Management":
        model_management_page()

def text_refinement_page():
    st.header("📝 Financial Text Refinement")
    
    if st.session_state.model is None:
        st.error("Please load or train a model first using the Model Management page.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Text")
        input_text = st.text_area(
            "Enter draft financial text:",
            height=300,
            placeholder="Enter your draft financial report or text here..."
        )
        
        # Refinement parameters
        st.subheader("Refinement Parameters")
        noise_level = st.slider("Initial Noise Level", 0.1, 1.0, 0.5, 0.1)
        num_inference_steps = st.slider("Inference Steps", 10, 100, 50, 10)
        
        refine_button = st.button("🔄 Refine Text", type="primary")
    
    with col2:
        st.subheader("Refined Text")
        
        if refine_button and input_text.strip():
            with st.spinner("Refining text..."):
                try:
                    # Process the text through the diffusion model
                    refined_text = refine_text(
                        input_text, 
                        noise_level, 
                        num_inference_steps
                    )
                    
                    st.text_area(
                        "Refined output:",
                        value=refined_text,
                        height=300,
                        disabled=True
                    )
                    
                    # Show improvement metrics
                    st.subheader("Quality Metrics")
                    metrics = st.session_state.evaluator.calculate_text_metrics(
                        input_text, refined_text
                    )
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Length Improvement", f"{metrics['length_ratio']:.2f}x")
                    with col_b:
                        st.metric("Word Diversity", f"{metrics['word_diversity']:.3f}")
                    with col_c:
                        st.metric("Readability Score", f"{metrics['readability']:.2f}")
                        
                except Exception as e:
                    st.error(f"Error during refinement: {str(e)}")
        
        elif refine_button:
            st.warning("Please enter some text to refine.")

def refine_text(input_text, noise_level, num_steps):
    """Refine text using the loaded diffusion model"""
    try:
        # Convert text to embedding
        embedding = st.session_state.text_processor.text_to_embedding(input_text)
        
        # Ensure embedding is on the same device as the model
        device = next(st.session_state.model.parameters()).device
        embedding = embedding.to(device)
        
        # Add initial noise
        noise = torch.randn_like(embedding) * noise_level
        noisy_embedding = embedding + noise
        
        # Perform denoising steps
        with torch.no_grad():
            for step in range(num_steps):
                t = torch.tensor([step], dtype=torch.long, device=device)
                predicted_noise = st.session_state.model(noisy_embedding, t)
                
                # Remove predicted noise
                alpha = 1.0 - (step / num_steps)
                noisy_embedding = noisy_embedding - alpha * predicted_noise
        
        # Convert back to text
        refined_text = st.session_state.text_processor.embedding_to_text(noisy_embedding)
        return refined_text
        
    except Exception as e:
        raise Exception(f"Text refinement failed: {str(e)}")

def model_training_page():
    st.header("🚀 Model Training")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Training Configuration")
        
        # Model parameters
        embedding_dim = st.selectbox("Embedding Dimension", [384, 512, 768], index=0)
        num_steps = st.slider("Diffusion Steps", 50, 200, 100, 10)
        
        # Training parameters
        learning_rate = st.number_input("Learning Rate", 1e-5, 1e-2, 1e-3, format="%.6f")
        batch_size = st.slider("Batch Size", 1, 8, 2, 1)
        num_epochs = st.slider("Number of Epochs", 1, 100, 10, 1)
        
        # Data source
        st.subheader("Training Data")
        data_source = st.selectbox(
            "Data Source",
            ["Sample Financial Texts", "Upload Custom Data"]
        )
        
        if data_source == "Upload Custom Data":
            uploaded_file = st.file_uploader(
                "Upload training data (JSON/TXT)",
                type=['json', 'txt']
            )
        
        # Initialize model button
        if st.button("🔧 Initialize Model"):
            st.session_state.model = DiffusionModel(
                embedding_dim=embedding_dim,
                num_steps=num_steps
            )
            st.session_state.trainer = ModelTrainer(
                model=st.session_state.model,
                text_processor=st.session_state.text_processor
            )
            st.success("Model initialized successfully!")
            st.rerun()
        
        # Training button
        start_training = st.button("🎯 Start Training", type="primary")
    
    with col2:
        st.subheader("Training Progress")
        
        if start_training:
            if st.session_state.model is None:
                st.error("Please initialize the model first.")
            else:
                train_model(learning_rate, batch_size, num_epochs, data_source)
        
        # Display training history
        if st.session_state.training_history:
            display_training_history()

def train_model(learning_rate, batch_size, num_epochs, data_source):
    """Train the diffusion model"""
    try:
        # Prepare training data
        if data_source == "Sample Financial Texts":
            from data.sample_texts import get_sample_financial_texts
            training_texts = get_sample_financial_texts()
        else:
            st.error("Custom data upload not implemented yet.")
            return
        
        # Create progress bars
        epoch_progress = st.progress(0)
        batch_progress = st.progress(0)
        loss_placeholder = st.empty()
        
        # Train the model
        history = st.session_state.trainer.train(
            texts=training_texts,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            progress_callback=lambda epoch, batch, loss: update_training_progress(
                epoch, batch, loss, num_epochs, len(training_texts)//batch_size,
                epoch_progress, batch_progress, loss_placeholder
            )
        )
        
        # Store training history
        st.session_state.training_history.extend(history)
        
        st.success("Training completed successfully!")
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")

def update_training_progress(epoch, batch, loss, total_epochs, total_batches, 
                           epoch_progress, batch_progress, loss_placeholder):
    """Update training progress displays"""
    epoch_progress.progress((epoch + 1) / total_epochs)
    batch_progress.progress((batch + 1) / total_batches)
    loss_placeholder.metric("Current Loss", f"{loss:.6f}")

def display_training_history():
    """Display training loss history"""
    if not st.session_state.training_history:
        return
    
    df = pd.DataFrame(st.session_state.training_history)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df['loss'],
        mode='lines',
        name='Training Loss',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Training Loss Over Time",
        xaxis_title="Training Step",
        yaxis_title="Loss",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def model_evaluation_page():
    st.header("📊 Model Evaluation")
    
    if st.session_state.model is None:
        st.error("Please load or train a model first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evaluation Configuration")
        
        evaluation_type = st.selectbox(
            "Evaluation Type",
            ["Text Quality Metrics", "Comparative Analysis", "Custom Evaluation"]
        )
        
        if evaluation_type == "Text Quality Metrics":
            test_texts = st.text_area(
                "Test Texts (one per line):",
                height=200,
                placeholder="Enter test financial texts, one per line..."
            )
            
            if st.button("📈 Run Evaluation"):
                if test_texts.strip():
                    run_text_quality_evaluation(test_texts)
                else:
                    st.warning("Please enter test texts.")
        
        elif evaluation_type == "Comparative Analysis":
            st.info("Compare model performance across different configurations")
            
            if st.button("🔬 Run Comparative Analysis"):
                run_comparative_analysis()
    
    with col2:
        st.subheader("Evaluation Results")
        
        # Display evaluation results area
        if 'evaluation_results' in st.session_state:
            display_evaluation_results()

def run_text_quality_evaluation(test_texts):
    """Run text quality evaluation"""
    try:
        texts = [text.strip() for text in test_texts.split('\n') if text.strip()]
        
        results = []
        progress_bar = st.progress(0)
        
        for i, text in enumerate(texts):
            # Refine the text
            refined_text = refine_text(text, 0.5, 50)
            
            # Calculate metrics
            metrics = st.session_state.evaluator.evaluate_refinement(text, refined_text)
            results.append({
                'original': text[:50] + "..." if len(text) > 50 else text,
                'refined': refined_text[:50] + "..." if len(refined_text) > 50 else refined_text,
                **metrics
            })
            
            progress_bar.progress((i + 1) / len(texts))
        
        st.session_state.evaluation_results = {
            'type': 'text_quality',
            'results': results,
            'timestamp': datetime.now()
        }
        
        st.success(f"Evaluation completed for {len(texts)} texts!")
        
    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}")

def run_comparative_analysis():
    """Run comparative analysis across different model configurations"""
    st.info("Comparative analysis feature coming soon!")

def display_evaluation_results():
    """Display evaluation results"""
    results = st.session_state.evaluation_results
    
    if results['type'] == 'text_quality':
        df = pd.DataFrame(results['results'])
        
        st.subheader("Quality Metrics Summary")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg BLEU Score", f"{df['bleu_score'].mean():.3f}")
        with col2:
            st.metric("Avg ROUGE Score", f"{df['rouge_score'].mean():.3f}")
        with col3:
            st.metric("Avg Length Ratio", f"{df['length_ratio'].mean():.2f}")
        with col4:
            st.metric("Avg Readability", f"{df['readability'].mean():.2f}")
        
        # Detailed results table
        st.subheader("Detailed Results")
        st.dataframe(df, use_container_width=True)

def model_management_page():
    st.header("⚙️ Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Save Model")
        
        if st.session_state.model is not None:
            model_name = st.text_input("Model Name", "financial_dllm_model")
            
            if st.button("💾 Save Model"):
                save_model(model_name)
        else:
            st.info("No model to save. Train a model first.")
        
        st.subheader("Load Model")
        
        # List available models
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        
        if model_files:
            selected_model = st.selectbox("Select Model", model_files)
            
            if st.button("📂 Load Model"):
                load_model(selected_model)
        else:
            st.info("No saved models found.")
    
    with col2:
        st.subheader("Model Information")
        
        if st.session_state.model is not None:
            st.json({
                "embedding_dim": st.session_state.model.embedding_dim,
                "num_steps": st.session_state.model.num_steps,
                "parameters": sum(p.numel() for p in st.session_state.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in st.session_state.model.parameters() if p.requires_grad)
            })
        
        st.subheader("Training History")
        if st.session_state.training_history:
            st.json({
                "total_steps": len(st.session_state.training_history),
                "final_loss": st.session_state.training_history[-1]['loss'],
                "best_loss": min(h['loss'] for h in st.session_state.training_history)
            })

def save_model(model_name):
    """Save the current model"""
    try:
        model_path = f"{model_name}.pth"
        
        checkpoint = {
            'model_state_dict': st.session_state.model.state_dict(),
            'model_config': {
                'embedding_dim': st.session_state.model.embedding_dim,
                'num_steps': st.session_state.model.num_steps
            },
            'training_history': st.session_state.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, model_path)
        st.success(f"Model saved as {model_path}")
        
    except Exception as e:
        st.error(f"Failed to save model: {str(e)}")

def load_model(model_file):
    """Load a saved model"""
    try:
        checkpoint = torch.load(model_file, map_location='cpu')
        
        # Initialize model with saved config
        config = checkpoint['model_config']
        st.session_state.model = DiffusionModel(
            embedding_dim=config['embedding_dim'],
            num_steps=config['num_steps']
        )
        
        # Load model state
        st.session_state.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history if available
        if 'training_history' in checkpoint:
            st.session_state.training_history = checkpoint['training_history']
        
        # Initialize trainer
        st.session_state.trainer = ModelTrainer(
            model=st.session_state.model,
            text_processor=st.session_state.text_processor
        )
        
        st.success(f"Model loaded from {model_file}")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")

if __name__ == "__main__":
    main()
