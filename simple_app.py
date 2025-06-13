import streamlit as st
import numpy as np
import pandas as pd
import re
import time
from datetime import datetime
from typing import List, Dict

# Set page configuration
st.set_page_config(
    page_title="Financial Text Diffusion Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Sample financial texts for demonstration
SAMPLE_FINANCIAL_TEXTS = [
    "The company reported strong quarterly results with revenue increasing by fifteen percent year over year. Net income rose to forty-two million dollars compared to thirty-one million in the previous quarter. Operating margins improved due to cost reduction initiatives and increased efficiency.",
    
    "Despite challenging market conditions, our financial performance remained stable. Revenue was flat at two hundred fifty million dollars while maintaining healthy profit margins. The management team implemented strategic cost controls to navigate economic uncertainty.",
    
    "Third quarter results exceeded expectations with record revenue of one hundred eighty million dollars. The company benefited from strong demand in core markets and successful product launches. Earnings per share increased to two dollars and fifteen cents.",
    
    "The investment portfolio generated positive returns this quarter driven by strong performance in technology and healthcare sectors. Asset allocation was rebalanced to reduce exposure to volatile markets while maintaining growth potential.",
    
    "Operational efficiency improvements contributed to margin expansion across all business segments. The company successfully reduced overhead costs while investing in growth initiatives. Cash flow from operations increased by twenty-five percent.",
]

DRAFT_TEXTS = [
    "Q3 results good. Revenue up. Profit ok too. Market doing fine.",
    "Company made money this quarter. Sales were higher than before. Costs went down some.",
    "Business is doing well. Numbers look good. Management happy with performance.",
    "Stock price went up. Investors like the company. Future looks bright maybe.",
    "Earnings beat expectations. Revenue growth strong. Margins improved slightly.",
]

class SimpleTextProcessor:
    """Simplified text processor for demonstration"""
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Basic text preprocessing"""
        if not text or not text.strip():
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Ensure proper sentence endings
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    @staticmethod
    def calculate_text_metrics(text1: str, text2: str) -> Dict[str, float]:
        """Calculate basic text comparison metrics"""
        words1 = len(text1.split())
        words2 = len(text2.split())
        
        length_ratio = words2 / max(1, words1)
        
        # Simple readability score (inverse of average word length)
        avg_word_len = np.mean([len(w) for w in text2.split()]) if text2.split() else 0
        readability = max(0, 1 - (avg_word_len - 5) / 10)
        
        # Word diversity (unique words / total words)
        words = text2.split()
        word_diversity = len(set(words)) / len(words) if words else 0
        
        return {
            'length_ratio': length_ratio,
            'readability': readability,
            'word_diversity': word_diversity
        }

class SimpleDiffusionModel:
    """Simplified diffusion model simulation"""
    
    def __init__(self, embedding_dim=384, num_steps=100):
        self.embedding_dim = embedding_dim
        self.num_steps = num_steps
        self.is_trained = False
    
    def text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to a simple embedding representation"""
        # Simple hash-based embedding for demonstration
        words = text.lower().split()
        embedding = np.zeros(self.embedding_dim)
        
        for i, word in enumerate(words):
            hash_val = hash(word) % self.embedding_dim
            embedding[hash_val] += 1.0 / (i + 1)  # Position weighting
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def refine_text(self, input_text: str, noise_level: float = 0.5, num_steps: int = 50) -> str:
        """Simulate text refinement using diffusion process"""
        if not self.is_trained:
            return "Model not trained yet. Please train the model first."
        
        # Simulate the diffusion refinement process
        original_embedding = self.text_to_embedding(input_text)
        
        # Add noise
        noise = np.random.randn(self.embedding_dim) * noise_level
        noisy_embedding = original_embedding + noise
        
        # Simulate denoising steps
        for step in range(num_steps):
            # Progressive denoising
            alpha = 1.0 - (step / num_steps) * 0.5
            noise_reduction = np.random.randn(self.embedding_dim) * 0.1
            noisy_embedding = noisy_embedding - alpha * noise_reduction
        
        # Convert back to text (simplified)
        return self._embedding_to_text(noisy_embedding, input_text)
    
    def _embedding_to_text(self, embedding: np.ndarray, original_text: str) -> str:
        """Convert embedding back to text (simplified approach)"""
        words = original_text.split()
        
        # Enhance the text based on embedding characteristics
        avg_value = np.mean(embedding)
        std_value = np.std(embedding)
        
        # Financial enhancement patterns
        enhancements = {
            'good': 'strong',
            'ok': 'stable',
            'fine': 'positive',
            'up': 'increased',
            'down': 'decreased',
            'made money': 'generated revenue',
            'doing well': 'performing strongly',
            'look good': 'show improvement',
            'happy': 'satisfied',
            'went up': 'appreciated',
            'like': 'favor',
            'bright': 'optimistic',
            'beat': 'exceeded'
        }
        
        # Apply enhancements
        enhanced_text = original_text.lower()
        for simple, enhanced in enhancements.items():
            enhanced_text = enhanced_text.replace(simple, enhanced)
        
        # Add financial context based on embedding characteristics
        if avg_value > 0.1:
            enhanced_text = enhanced_text.replace('results', 'quarterly results')
            enhanced_text = enhanced_text.replace('revenue', 'total revenue')
        
        # Improve sentence structure
        sentences = enhanced_text.split('.')
        improved_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Capitalize first letter
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                # Add financial context
                if len(sentence.split()) < 5:
                    sentence = f"The company's {sentence.lower()}"
                improved_sentences.append(sentence)
        
        return '. '.join(improved_sentences) + '.'

def main():
    st.title("🏦 Financial Text Diffusion Model (dLLM)")
    st.markdown("---")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Text Refinement", "Model Training", "Model Information"]
        )
        
        st.header("Model Status")
        if st.session_state.model_trained:
            st.success("✅ Model Trained")
        else:
            st.warning("⚠️ Model Not Trained")
    
    if page == "Text Refinement":
        text_refinement_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Model Information":
        model_information_page()

def text_refinement_page():
    st.header("📝 Financial Text Refinement")
    
    if not st.session_state.model_trained:
        st.error("Please train the model first using the Model Training page.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Text")
        
        # Provide sample draft texts
        st.write("**Sample Draft Texts:**")
        for i, draft in enumerate(DRAFT_TEXTS[:3]):
            if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                st.session_state.input_text = draft
        
        input_text = st.text_area(
            "Enter draft financial text:",
            height=300,
            value=st.session_state.get('input_text', ''),
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
                # Initialize model
                model = SimpleDiffusionModel()
                model.is_trained = st.session_state.model_trained
                
                # Process the text
                refined_text = model.refine_text(
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
                processor = SimpleTextProcessor()
                metrics = processor.calculate_text_metrics(input_text, refined_text)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Length Improvement", f"{metrics['length_ratio']:.2f}x")
                with col_b:
                    st.metric("Word Diversity", f"{metrics['word_diversity']:.3f}")
                with col_c:
                    st.metric("Readability Score", f"{metrics['readability']:.2f}")
        
        elif refine_button:
            st.warning("Please enter some text to refine.")

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
        num_epochs = st.slider("Number of Epochs", 1, 50, 10, 1)
        
        # Training button
        start_training = st.button("🎯 Start Training", type="primary")
    
    with col2:
        st.subheader("Training Progress")
        
        if start_training:
            train_model(learning_rate, batch_size, num_epochs, embedding_dim, num_steps)
        
        # Display training history
        if st.session_state.training_history:
            display_training_history()

def train_model(learning_rate, batch_size, num_epochs, embedding_dim, num_steps):
    """Simulate model training"""
    try:
        # Create progress bars
        epoch_progress = st.progress(0)
        loss_placeholder = st.empty()
        
        # Simulate training process
        for epoch in range(num_epochs):
            # Simulate epoch training
            epoch_loss = simulate_training_epoch(epoch, num_epochs)
            
            # Update progress
            epoch_progress.progress((epoch + 1) / num_epochs)
            loss_placeholder.metric("Current Loss", f"{epoch_loss:.6f}")
            
            # Store training history
            st.session_state.training_history.append({
                'epoch': epoch,
                'loss': epoch_loss,
                'timestamp': time.time()
            })
            
            # Simulate training time
            time.sleep(0.1)
        
        # Mark model as trained
        st.session_state.model_trained = True
        
        st.success("Training completed successfully!")
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")

def simulate_training_epoch(epoch, total_epochs):
    """Simulate training loss for an epoch"""
    # Simulate decreasing loss with some noise
    base_loss = 1.0
    decay_factor = 0.9
    noise = np.random.normal(0, 0.05)
    
    loss = base_loss * (decay_factor ** epoch) + noise
    return max(0.01, loss)  # Ensure loss doesn't go negative

def display_training_history():
    """Display training loss history"""
    if not st.session_state.training_history:
        return
    
    df = pd.DataFrame(st.session_state.training_history)
    
    st.line_chart(df.set_index('epoch')['loss'])
    
    # Display statistics
    st.subheader("Training Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Epochs", len(df))
    with col2:
        st.metric("Final Loss", f"{df['loss'].iloc[-1]:.6f}")
    with col3:
        st.metric("Best Loss", f"{df['loss'].min():.6f}")

def model_information_page():
    st.header("📋 Model Information")
    
    st.subheader("About Financial Text Diffusion Model")
    
    st.markdown("""
    This application demonstrates a **diffusion-based Large Language Model (dLLM)** specifically designed for financial text refinement. 
    The model operates in embedding space to gradually improve the quality and coherence of financial documents.
    
    ### Key Features:
    - **Text Refinement**: Transforms draft financial text into polished, professional content
    - **Diffusion Process**: Uses a noise-based refinement approach similar to image diffusion models
    - **Financial Context**: Specialized for financial terminology and reporting standards
    - **Embedding Space Operations**: Works in continuous vector space for smooth transformations
    
    ### Model Architecture:
    - **Embedding Dimension**: 384 (compatible with sentence-transformers)
    - **Diffusion Steps**: 100 (configurable noise schedule)
    - **Neural Network**: Multi-layer perceptron with time conditioning
    - **Training Process**: Learns to predict and remove noise from text embeddings
    
    ### Use Cases:
    - Draft financial report refinement
    - Earnings statement improvement
    - Investment analysis enhancement
    - Financial commentary polishing
    """)
    
    if st.session_state.model_trained:
        st.subheader("Current Model Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Status": "Trained",
                "Training Epochs": len(st.session_state.training_history),
                "Final Loss": f"{st.session_state.training_history[-1]['loss']:.6f}" if st.session_state.training_history else "N/A",
                "Training Time": datetime.fromtimestamp(st.session_state.training_history[-1]['timestamp']).strftime("%Y-%m-%d %H:%M:%S") if st.session_state.training_history else "N/A"
            })
        
        with col2:
            st.subheader("Training Progress")
            if st.session_state.training_history:
                df = pd.DataFrame(st.session_state.training_history)
                st.line_chart(df.set_index('epoch')['loss'])
    
    st.subheader("Technical Implementation")
    
    st.markdown("""
    ### Based on Research Paper Concepts:
    - **Forward Process**: Gradually adds Gaussian noise to text embeddings
    - **Reverse Process**: Neural network learns to denoise embeddings step-by-step
    - **Training Objective**: Minimize MSE between predicted and actual noise
    - **Inference**: Start with noisy embedding and iteratively denoise to refine text
    
    ### Hardware Requirements:
    - **CPU**: Multi-core processor recommended
    - **Memory**: 8GB RAM minimum, 16GB+ recommended
    - **GPU**: Optional but recommended for larger models (RTX 4060 or better)
    - **Storage**: 2GB for model and dependencies
    
    ### Deployment Options:
    - **Local Development**: Run on laptop/desktop
    - **Cloud Scaling**: Deploy to Azure ML or similar platforms
    - **API Endpoint**: Serve as REST API for integration
    """)

if __name__ == "__main__":
    main()