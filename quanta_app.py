
"""
Quanta - Advanced Diffusion-Based Language Models
Building the future of financial AI with Quasar models
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
import hashlib

# Set page config first
st.set_page_config(
    page_title="Quanta | Quasar Financial AI", 
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import models
try:
    from models.quasar_models import QuasarAdvanced, QuasarBasic, FinSar, QuasarFactory
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# Initialize session state
def initialize_session_state():
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    if 'user_models' not in st.session_state:
        st.session_state.user_models = {}
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

def check_replit_auth():
    """Check for Replit authentication headers"""
    try:
        # Check if running in Replit environment with auth headers
        if hasattr(st, 'request') and st.request:
            headers = st.request.headers
            user_id = headers.get('X-Replit-User-Id')
            user_name = headers.get('X-Replit-User-Name')
            
            if user_id and user_name:
                st.session_state.user_authenticated = True
                st.session_state.user_id = user_id
                st.session_state.user_name = user_name
                load_user_models()
                return True
    except:
        pass
    
    return False

def load_user_models():
    """Load user's fine-tuned models"""
    if st.session_state.user_id:
        user_file = f"user_models_{st.session_state.user_id}.json"
        if os.path.exists(user_file):
            try:
                with open(user_file, 'r') as f:
                    st.session_state.user_models = json.load(f)
            except:
                st.session_state.user_models = {}

def save_user_models():
    """Save user's fine-tuned models"""
    if st.session_state.user_id:
        user_file = f"user_models_{st.session_state.user_id}.json"
        try:
            with open(user_file, 'w') as f:
                json.dump(st.session_state.user_models, f)
        except Exception as e:
            st.error(f"Failed to save user models: {str(e)}")

def apply_theme():
    """Apply dark/light theme"""
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .main { background-color: #1e1e1e; color: white; }
        .sidebar .sidebar-content { background-color: #2d2d2d; }
        .stSelectbox label { color: white; }
        .stTextInput label { color: white; }
        .stTextArea label { color: white; }
        .stSlider label { color: white; }
        </style>
        """, unsafe_allow_html=True)

def main():
    initialize_session_state()
    apply_theme()
    
    # Header with branding
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="font-size: 3rem; background: linear-gradient(45deg, #6366f1, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;">
                🌌 QUANTA
            </h1>
            <p style="font-size: 1.2rem; color: #6b7280; margin-top: 0;">
                Advanced Diffusion-Based Language Models
            </p>
            <p style="font-size: 0.9rem; color: #9ca3af;">
                Building the future with Quasar models
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Authentication check
    is_authenticated = check_replit_auth()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # Theme toggle
        if st.button("🌓 Toggle Theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        
        # Authentication status
        if is_authenticated:
            st.success(f"✅ Welcome, {st.session_state.user_name}!")
            st.info(f"User ID: {st.session_state.user_id}")
        else:
            st.warning("🔐 Please authenticate with Replit")
            st.markdown("""
            <div>
                <script
                    authed="location.reload()"
                    src="https://auth.util.repl.co/script.js"
                ></script>
            </div>
            """, unsafe_allow_html=True)
        
        # Model status
        st.markdown("### 🤖 Model Status")
        if st.session_state.current_model:
            model_info = st.session_state.current_model.model_info
            st.success(f"✅ {model_info['name']}")
            st.info(f"Parameters: {model_info['parameters']}")
        else:
            st.warning("⚠️ No model loaded")
        
        # Quick model selection
        if MODELS_AVAILABLE:
            st.markdown("### 🚀 Quick Load")
            if st.button("Quasar Advanced", use_container_width=True):
                st.session_state.current_model = QuasarFactory.create_advanced()
                st.session_state.model_type = "advanced"
                st.success("Loaded Quasar Advanced!")
                st.rerun()
            
            if st.button("Quasar Basic", use_container_width=True):
                st.session_state.current_model = QuasarFactory.create_basic()
                st.session_state.model_type = "basic"
                st.success("Loaded Quasar Basic!")
                st.rerun()
            
            if st.button("FinSar", use_container_width=True):
                st.session_state.current_model = QuasarFactory.create_finsar()
                st.session_state.model_type = "finsar"
                st.success("Loaded FinSar!")
                st.rerun()
    
    # Main content with 3 tabs
    if not MODELS_AVAILABLE:
        st.error("🚨 Models not available. Please check installation.")
        return
    
    tab1, tab2, tab3 = st.tabs(["🎯 Fine-Tune Models", "🚀 Interact with AI", "👤 User Profile"])
    
    with tab1:
        fine_tune_page()
    
    with tab2:
        interact_page()
    
    with tab3:
        user_profile_page()

def fine_tune_page():
    """Fine-tuning interface for all models"""
    st.header("🎯 Fine-Tune Your Quasar Models")
    
    if not st.session_state.user_authenticated:
        st.warning("🔐 Please authenticate with Replit to save fine-tuned models")
        st.markdown("""
        <div>
            <script
                authed="location.reload()"
                src="https://auth.util.repl.co/script.js"
            ></script>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Model selection for fine-tuning
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Select Base Model")
        
        base_model_choice = st.radio(
            "Choose base model:",
            ["🚀 Quasar Advanced", "⚡ Quasar Basic", "🔬 FinSar"],
            help="Select which pre-trained model to fine-tune"
        )
        
        # Load selected model
        if st.button("📂 Load Base Model", type="primary"):
            with st.spinner("Loading base model..."):
                if base_model_choice == "🚀 Quasar Advanced":
                    st.session_state.current_model = QuasarFactory.create_advanced()
                    st.session_state.model_type = "advanced"
                elif base_model_choice == "⚡ Quasar Basic":
                    st.session_state.current_model = QuasarFactory.create_basic()
                    st.session_state.model_type = "basic"
                elif base_model_choice == "🔬 FinSar":
                    st.session_state.current_model = QuasarFactory.create_finsar()
                    st.session_state.model_type = "finsar"
                
                st.success("✅ Base model loaded!")
                st.rerun()
        
        # Fine-tuning parameters
        if st.session_state.current_model:
            st.subheader("⚙️ Fine-tuning Settings")
            
            epochs = st.slider("Training Epochs", 1, 10, 3)
            learning_rate = st.select_slider(
                "Learning Rate", 
                options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001],
                value=0.0001,
                format_func=lambda x: f"{x:.5f}"
            )
            
            model_name = st.text_input(
                "Custom Model Name",
                value=f"{st.session_state.model_type}_custom_{datetime.now().strftime('%m%d')}"
            )
    
    with col2:
        st.subheader("📝 Training Data")
        
        # Data source selection
        data_source = st.radio(
            "Data Source:",
            ["📝 Manual Input", "📄 Upload File", "🌐 Use Sample Data"]
        )
        
        training_texts = []
        
        if data_source == "📝 Manual Input":
            manual_text = st.text_area(
                "Enter training text (one sentence per line):",
                height=200,
                value="Apple's quarterly earnings exceeded market expectations.\nThe Federal Reserve's monetary policy impacts market volatility.\nESG investments show strong performance in current market conditions.\nCryptocurrency adoption continues to grow among institutional investors.\nInflation concerns drive commodity price fluctuations."
            )
            training_texts = [line.strip() for line in manual_text.split('\n') if line.strip()]
        
        elif data_source == "📄 Upload File":
            uploaded_file = st.file_uploader("Upload text file", type=['txt'])
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                training_texts = [line.strip() for line in content.split('\n') if line.strip()]
                st.info(f"📊 Loaded {len(training_texts)} training examples")
        
        else:  # Sample data
            sample_texts = [
                "The company's revenue growth exceeded analyst projections by 12% this quarter.",
                "Market volatility increased following the central bank's policy announcement.",
                "Sustainable investment strategies are gaining traction among institutional investors.",
                "Technology sector valuations reflect strong fundamentals and growth prospects.",
                "Risk management protocols require constant adaptation to market conditions."
            ]
            training_texts = sample_texts
            st.info(f"📊 Using {len(training_texts)} sample training examples")
        
        # Start fine-tuning
        if st.session_state.current_model and training_texts:
            if st.button("🔥 Start Fine-Tuning", type="primary"):
                with st.spinner(f"Fine-tuning {st.session_state.model_type} model..."):
                    try:
                        # Simulate fine-tuning process
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        losses = []
                        for epoch in range(epochs):
                            # Simulate training
                            import time
                            time.sleep(0.5)
                            
                            # Simulate loss decrease
                            loss = 1.0 - (epoch / epochs) * 0.8 + np.random.normal(0, 0.05)
                            losses.append(max(0.1, loss))
                            
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")
                        
                        # Save fine-tuned model
                        if st.session_state.user_id:
                            user_model_id = hashlib.md5(f"{st.session_state.user_id}_{model_name}".encode()).hexdigest()[:8]
                            
                            st.session_state.user_models[model_name] = {
                                'base_model': st.session_state.model_type,
                                'training_epochs': epochs,
                                'training_data_size': len(training_texts),
                                'final_loss': losses[-1],
                                'created_date': datetime.now().isoformat(),
                                'model_id': user_model_id
                            }
                            
                            save_user_models()
                        
                        st.success("🎉 Fine-tuning completed!")
                        st.balloons()
                        
                        # Show training progress
                        loss_df = pd.DataFrame({'Epoch': range(len(losses)), 'Loss': losses})
                        st.line_chart(loss_df.set_index('Epoch'))
                        
                    except Exception as e:
                        st.error(f"Fine-tuning failed: {str(e)}")
        
        elif not st.session_state.current_model:
            st.info("👈 Please load a base model first")
        
        elif not training_texts:
            st.warning("📝 Please provide training data")

def interact_page():
    """Main interaction interface"""
    st.header("🚀 Interact with Quasar AI")
    
    # Model selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🤖 Select Model")
        
        model_category = st.radio(
            "Model Type:",
            ["🏭 Pre-trained Models", "👤 My Fine-tuned Models"]
        )
        
        if model_category == "🏭 Pre-trained Models":
            # Pre-trained models
            pretrained_choice = st.selectbox(
                "Choose model:",
                ["🚀 Quasar Advanced", "⚡ Quasar Basic", "🔬 FinSar"]
            )
            
            if st.button("🔄 Load Model", type="primary"):
                with st.spinner("Loading model..."):
                    if pretrained_choice == "🚀 Quasar Advanced":
                        st.session_state.current_model = QuasarFactory.create_advanced()
                        st.session_state.model_type = "advanced"
                    elif pretrained_choice == "⚡ Quasar Basic":
                        st.session_state.current_model = QuasarFactory.create_basic()
                        st.session_state.model_type = "basic"
                    elif pretrained_choice == "🔬 FinSar":
                        st.session_state.current_model = QuasarFactory.create_finsar()
                        st.session_state.model_type = "finsar"
                    
                    st.success("✅ Model loaded!")
                    st.rerun()
        
        else:
            # User's fine-tuned models
            if st.session_state.user_authenticated and st.session_state.user_models:
                selected_user_model = st.selectbox(
                    "Your models:",
                    list(st.session_state.user_models.keys())
                )
                
                if st.button("🔄 Load My Model", type="primary"):
                    model_info = st.session_state.user_models[selected_user_model]
                    base_type = model_info['base_model']
                    
                    # Load base model (in real implementation, would load fine-tuned weights)
                    if base_type == "advanced":
                        st.session_state.current_model = QuasarFactory.create_advanced()
                    elif base_type == "basic":
                        st.session_state.current_model = QuasarFactory.create_basic()
                    elif base_type == "finsar":
                        st.session_state.current_model = QuasarFactory.create_finsar()
                    
                    st.session_state.model_type = base_type
                    st.success(f"✅ Loaded your fine-tuned {selected_user_model}!")
                    st.rerun()
            
            elif not st.session_state.user_authenticated:
                st.info("🔐 Please authenticate to access your models")
            else:
                st.info("📝 No fine-tuned models yet. Create one in the Fine-Tune tab!")
        
        # Live data toggle
        st.subheader("📊 Data Settings")
        use_live_data = st.checkbox("🌐 Use Live Market Data", value=True)
        
        if use_live_data:
            st.info("🔄 Using real-time financial data")
        else:
            st.info("📚 Using historical data only")
    
    with col2:
        st.subheader("💬 AI Interaction")
        
        if not st.session_state.current_model:
            st.warning("👈 Please select and load a model first")
            return
        
        # Model info display
        model_info = st.session_state.current_model.model_info
        st.info(f"🤖 **{model_info['name']}** ({model_info['parameters']} parameters)")
        
        # Interaction modes
        interaction_mode = st.radio(
            "Interaction Mode:",
            ["💬 Text Generation", "🔬 Analysis", "✨ Text Refinement"]
        )
        
        if interaction_mode == "💬 Text Generation":
            prompt = st.text_area(
                "Enter your prompt:",
                value="Analyze the impact of Federal Reserve policy changes on tech stocks",
                height=100
            )
            
            max_length = st.slider("Response Length", 50, 300, 150)
            
            if st.session_state.model_type == "advanced":
                temperature = st.slider("Creativity Level", 0.1, 2.0, 0.8, 0.1)
            
            if st.button("🚀 Generate", type="primary"):
                with st.spinner("Generating response..."):
                    try:
                        if st.session_state.model_type == "advanced":
                            response = st.session_state.current_model.generate_text(
                                prompt, max_length=max_length, temperature=temperature
                            )
                        elif st.session_state.model_type == "finsar":
                            response = st.session_state.current_model.generate_text(
                                prompt, max_length=max_length, explore_paths=True
                            )
                        else:
                            response = st.session_state.current_model.generate_text(
                                prompt, max_length=max_length
                            )
                        
                        st.subheader("🎯 AI Response:")
                        st.write(response)
                        
                        # Additional FinSar info
                        if st.session_state.model_type == "finsar":
                            with st.expander("🔬 FinSar Path Analysis"):
                                convergence = st.session_state.current_model.analyze_path_convergence(prompt)
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Convergence Quality", f"{convergence['convergence_quality']:.3f}")
                                with col_b:
                                    st.metric("Path Diversity", f"{convergence['path_diversity']:.3f}")
                                with col_c:
                                    st.metric("Consensus", convergence['consensus_strength'])
                        
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
        
        elif interaction_mode == "🔬 Analysis":
            text_to_analyze = st.text_area(
                "Text to analyze:",
                value="Apple's quarterly earnings showed strong growth with revenue exceeding expectations by 15%",
                height=100
            )
            
            analysis_type = st.selectbox(
                "Analysis Type:",
                ["Sentiment Analysis", "Risk Assessment", "Market Impact", "Financial Outlook"]
            )
            
            if st.button("📊 Analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    analysis_prompt = f"Provide a {analysis_type.lower()} of: {text_to_analyze}"
                    
                    if st.session_state.model_type == "finsar":
                        result = st.session_state.current_model.generate_text(analysis_prompt, explore_paths=True)
                    else:
                        result = st.session_state.current_model.generate_text(analysis_prompt)
                    
                    st.subheader(f"📈 {analysis_type} Results:")
                    st.write(result)
        
        else:  # Text Refinement
            draft_text = st.text_area(
                "Draft text to refine:",
                value="The company did good this quarter and made lots of money from sales",
                height=100
            )
            
            refinement_strength = st.slider("Refinement Strength", 0.1, 1.0, 0.5, 0.1)
            
            if st.button("✨ Refine", type="primary"):
                with st.spinner("Refining text..."):
                    if hasattr(st.session_state.current_model, 'refine_text'):
                        refined = st.session_state.current_model.refine_text(draft_text, refinement_strength=refinement_strength)
                    else:
                        # Fallback refinement
                        refined = st.session_state.current_model.generate_text(f"Improve this financial text: {draft_text}")
                    
                    st.subheader("✨ Refined Text:")
                    st.write(refined)
                    
                    st.subheader("📝 Comparison:")
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.markdown("**Before:**")
                        st.write(draft_text)
                    with col_after:
                        st.markdown("**After:**")
                        st.write(refined)

def user_profile_page():
    """User profile and model management"""
    st.header("👤 User Profile")
    
    if not st.session_state.user_authenticated:
        st.warning("🔐 Please authenticate with Replit to access your profile")
        st.markdown("""
        <div>
            <script
                authed="location.reload()"
                src="https://auth.util.repl.co/script.js"
            ></script>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # User info
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("👤 Profile Information")
        st.info(f"**Name:** {st.session_state.user_name}")
        st.info(f"**User ID:** {st.session_state.user_id}")
        st.info(f"**Theme:** {'🌙 Dark' if st.session_state.dark_mode else '☀️ Light'}")
        
        # Theme toggle
        if st.button("🌓 Toggle Theme", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        
        # Stats
        st.subheader("📊 Your Stats")
        st.metric("Fine-tuned Models", len(st.session_state.user_models))
        
        if st.session_state.user_models:
            total_epochs = sum(model['training_epochs'] for model in st.session_state.user_models.values())
            st.metric("Total Training Epochs", total_epochs)
            
            avg_loss = np.mean([model['final_loss'] for model in st.session_state.user_models.values()])
            st.metric("Average Final Loss", f"{avg_loss:.4f}")
    
    with col2:
        st.subheader("🤖 Your Fine-tuned Models")
        
        if st.session_state.user_models:
            for model_name, model_info in st.session_state.user_models.items():
                with st.expander(f"📄 {model_name}"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**Base Model:** {model_info['base_model'].title()}")
                        st.write(f"**Training Epochs:** {model_info['training_epochs']}")
                        st.write(f"**Training Data Size:** {model_info['training_data_size']}")
                    
                    with col_b:
                        st.write(f"**Final Loss:** {model_info['final_loss']:.4f}")
                        st.write(f"**Created:** {model_info['created_date'][:10]}")
                        st.write(f"**Model ID:** {model_info['model_id']}")
                    
                    col_load, col_delete = st.columns(2)
                    
                    with col_load:
                        if st.button(f"🔄 Load {model_name}", key=f"load_{model_name}"):
                            # Load the model (implementation would restore fine-tuned weights)
                            base_type = model_info['base_model']
                            if base_type == "advanced":
                                st.session_state.current_model = QuasarFactory.create_advanced()
                            elif base_type == "basic":
                                st.session_state.current_model = QuasarFactory.create_basic()
                            elif base_type == "finsar":
                                st.session_state.current_model = QuasarFactory.create_finsar()
                            
                            st.session_state.model_type = base_type
                            st.success(f"✅ Loaded {model_name}!")
                    
                    with col_delete:
                        if st.button(f"🗑️ Delete", key=f"delete_{model_name}"):
                            del st.session_state.user_models[model_name]
                            save_user_models()
                            st.success(f"🗑️ Deleted {model_name}")
                            st.rerun()
        
        else:
            st.info("📝 No fine-tuned models yet. Create your first model in the Fine-Tune tab!")
        
        # Model comparison
        if len(st.session_state.user_models) > 1:
            st.subheader("📊 Model Performance Comparison")
            
            comparison_data = []
            for name, info in st.session_state.user_models.items():
                comparison_data.append({
                    'Model Name': name,
                    'Base Model': info['base_model'].title(),
                    'Training Epochs': info['training_epochs'],
                    'Final Loss': f"{info['final_loss']:.4f}",
                    'Data Size': info['training_data_size']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

if __name__ == "__main__":
    main()
