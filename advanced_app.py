import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime
import time
import json

# Add project directories to path
sys.path.append('.')
sys.path.append('./models/advanced')
sys.path.append('./data_collection')
sys.path.append('./database')
sys.path.append('./utils/database')

# Import our modules
from models.advanced.financial_diffusion_llm import FinancialDiffusionLLM
from data_collection.financial_data_collector import collect_comprehensive_dataset
from utils.database.data_manager import FinancialDataManager

# Set page configuration
st.set_page_config(
    page_title="Advanced Financial Diffusion LLM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'advanced_model' not in st.session_state:
    st.session_state.advanced_model = None
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def main():
    st.title("🧠 Advanced Financial Diffusion Language Model")
    st.markdown("Pre-trained transformer-based diffusion model for financial text generation and refinement")
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Data Collection", "Model Training", "Text Generation", "Model Management", "Database Analytics"]
        )
        
        st.header("System Status")
        
        # Model status
        if st.session_state.model_trained and st.session_state.advanced_model:
            st.success("✅ Advanced Model Ready")
            model_info = st.session_state.advanced_model.get_model_info()
            st.info(f"Parameters: {model_info['parameters']:,}")
            st.info(f"Vocab Size: {model_info['vocab_size']:,}")
        else:
            st.warning("⚠️ Model Not Trained")
        
        # Database status
        try:
            if st.session_state.data_manager is None:
                st.session_state.data_manager = FinancialDataManager()
            st.success("✅ Database Connected")
        except Exception as e:
            st.error(f"❌ Database Error: {str(e)}")
    
    # Route to selected page
    if page == "Data Collection":
        data_collection_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Text Generation":
        text_generation_page()
    elif page == "Model Management":
        model_management_page()
    elif page == "Database Analytics":
        database_analytics_page()

def data_collection_page():
    st.header("📊 Real-Time Financial Data Collection")
    
    st.markdown("""
    Collect live financial data from multiple sources:
    - Yahoo Finance (company data, stock prices, financial statements)
    - Financial news feeds (RSS from major outlets)
    - SEC EDGAR filings (10-K, 10-Q summaries)
    - Market indicators and indices
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Sources")
        
        # Data collection controls
        collect_companies = st.checkbox("Collect Company Data", value=True)
        collect_news = st.checkbox("Collect Financial News", value=True)
        collect_sec = st.checkbox("Collect SEC Filings", value=True)
        collect_market = st.checkbox("Collect Market Data", value=True)
        
        num_companies = st.slider("Number of Companies", 5, 50, 10)
        
        if st.button("🔄 Start Data Collection", type="primary"):
            collect_financial_data(collect_companies, collect_news, collect_sec, collect_market, num_companies)
    
    with col2:
        st.subheader("Collection Status")
        
        # Show data collection status
        if st.session_state.data_manager:
            try:
                stats = st.session_state.data_manager.get_training_statistics()
                
                if stats:
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Companies", stats.get('total_companies', 0))
                        st.metric("News Articles", stats.get('total_news', 0))
                    
                    with col_b:
                        st.metric("Earnings Records", stats.get('total_earnings', 0))
                        st.metric("SEC Filings", stats.get('total_sec_filings', 0))
                    
                    with col_c:
                        st.metric("Market Indicators", stats.get('total_market_indicators', 0))
                        st.metric("Training Texts", stats.get('total_training_texts', 0))
                
                # Show recent data
                st.subheader("Recent Financial Data")
                if st.button("📈 Show Recent Data"):
                    show_recent_data()
                    
            except Exception as e:
                st.error(f"Error loading statistics: {str(e)}")

def collect_financial_data(collect_companies, collect_news, collect_sec, collect_market, num_companies):
    """Collect financial data from various sources"""
    if not st.session_state.data_manager:
        st.session_state.data_manager = FinancialDataManager()
    
    with st.spinner("Collecting financial data..."):
        try:
            # Collect and store data
            counts = st.session_state.data_manager.collect_and_store_data()
            
            st.success("Data collection completed!")
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Companies Added", counts.get('companies', 0))
            with col2:
                st.metric("News Articles", counts.get('news', 0))
            with col3:
                st.metric("SEC Filings", counts.get('sec_filings', 0))
            with col4:
                st.metric("Market Indicators", counts.get('market_indicators', 0))
            
            # Prepare training data
            with st.spinner("Preparing training texts..."):
                training_texts = st.session_state.data_manager.prepare_training_texts()
                st.session_state.training_data = training_texts
                
                st.info(f"Prepared {len(training_texts)} training texts from collected data")
            
        except Exception as e:
            st.error(f"Data collection failed: {str(e)}")

def show_recent_data():
    """Display recent financial data"""
    if not st.session_state.data_manager:
        return
    
    try:
        recent_data = st.session_state.data_manager.get_recent_financial_data(limit=10)
        
        # Recent companies
        if recent_data['companies']:
            st.subheader("Recent Companies")
            companies_df = pd.DataFrame(recent_data['companies'])
            st.dataframe(companies_df, use_container_width=True)
        
        # Recent news
        if recent_data['news']:
            st.subheader("Recent News")
            news_df = pd.DataFrame(recent_data['news'])
            st.dataframe(news_df, use_container_width=True)
        
        # Market indicators
        if recent_data['market_indicators']:
            st.subheader("Market Indicators")
            market_df = pd.DataFrame(recent_data['market_indicators'])
            st.dataframe(market_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying recent data: {str(e)}")

def model_training_page():
    st.header("🚀 Advanced Model Training")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        
        # Advanced model parameters
        vocab_size = st.selectbox("Vocabulary Size", [5000, 10000, 20000], index=1)
        d_model = st.selectbox("Model Dimension", [256, 512, 768], index=1)
        num_heads = st.selectbox("Attention Heads", [4, 8, 12], index=1)
        num_layers = st.selectbox("Transformer Layers", [4, 6, 8], index=1)
        max_seq_length = st.selectbox("Max Sequence Length", [256, 512, 1024], index=1)
        
        st.subheader("Training Parameters")
        num_epochs = st.slider("Training Epochs", 10, 200, 50)
        
        # Training data source
        st.subheader("Training Data")
        use_database = st.checkbox("Use Database Training Texts", value=True)
        
        if not use_database:
            st.info("Using sample financial texts for training")
        
        # Initialize and train model
        if st.button("🎯 Initialize & Train Model", type="primary"):
            train_advanced_model(vocab_size, d_model, num_heads, num_layers, max_seq_length, num_epochs, use_database)
    
    with col2:
        st.subheader("Training Progress")
        
        # Display training progress and results
        if st.session_state.advanced_model and hasattr(st.session_state.advanced_model, 'training_history'):
            if st.session_state.advanced_model.training_history:
                display_advanced_training_history()

def train_advanced_model(vocab_size, d_model, num_heads, num_layers, max_seq_length, num_epochs, use_database):
    """Train the advanced financial diffusion LLM"""
    try:
        # Initialize model
        st.session_state.advanced_model = FinancialDiffusionLLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
            num_diffusion_steps=1000
        )
        
        # Get training data
        if use_database and st.session_state.data_manager:
            training_texts = st.session_state.data_manager.prepare_training_texts()
        else:
            # Use sample data
            training_texts = [
                "The company reported strong quarterly earnings with revenue growth of 15% year-over-year and improved operating margins.",
                "Market volatility increased following Federal Reserve policy announcements regarding interest rate adjustments and economic outlook.",
                "Quarterly results exceeded analyst expectations with earnings per share of $2.45 compared to consensus estimates of $2.20.",
                "The investment portfolio generated positive returns driven by strategic asset allocation and risk management practices.",
                "Operating cash flow remained robust at $150 million supporting continued business expansion and shareholder returns.",
                "Revenue diversification across multiple segments reduced dependency on cyclical market conditions and enhanced stability.",
                "The company maintained strong balance sheet fundamentals with improved debt-to-equity ratios and liquidity positions.",
                "Strategic acquisitions contributed to market share expansion and operational synergies exceeding initial projections.",
                "Management guidance for the full fiscal year reflects confidence in sustainable growth trajectory and market positioning.",
                "Risk assessment indicates stable credit quality with provisions adequate for current economic environment and portfolio composition."
            ]
        
        if not training_texts:
            st.error("No training texts available. Please collect data first.")
            return
        
        st.info(f"Training model with {len(training_texts)} texts...")
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train model with progress updates
        with st.spinner("Training advanced financial diffusion model..."):
            losses = []
            
            for epoch in range(num_epochs):
                # Simulate training step
                epoch_loss = st.session_state.advanced_model.train_step(training_texts)
                losses.append(epoch_loss)
                
                # Update progress
                progress = (epoch + 1) / num_epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.6f}")
                
                # Update training history
                st.session_state.advanced_model.training_history.append({
                    'epoch': epoch,
                    'loss': epoch_loss,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Small delay for visualization
                time.sleep(0.1)
        
        # Mark as trained
        st.session_state.advanced_model.is_trained = True
        st.session_state.model_trained = True
        
        # Save model checkpoint
        model_path = f"financial_diffusion_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.session_state.advanced_model.save_model(model_path)
        
        # Save to database
        if st.session_state.data_manager:
            try:
                model_info = st.session_state.advanced_model.get_model_info()
                st.session_state.data_manager.save_model_checkpoint(
                    model_name="FinancialDiffusionLLM",
                    model_config=model_info,
                    training_params={"epochs": num_epochs, "texts": len(training_texts)},
                    performance_metrics={"final_loss": losses[-1] if losses else 0.0},
                    epoch=num_epochs,
                    loss=losses[-1] if losses else 0.0,
                    checkpoint_path=model_path,
                    is_best=True
                )
            except Exception as e:
                st.warning(f"Model saved locally but database checkpoint failed: {str(e)}")
        
        st.success(f"Training completed! Model saved as {model_path}")
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")

def display_advanced_training_history():
    """Display advanced training history"""
    if not st.session_state.advanced_model.training_history:
        return
    
    df = pd.DataFrame(st.session_state.advanced_model.training_history)
    
    # Training loss chart
    st.line_chart(df.set_index('epoch')['loss'])
    
    # Training statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Epochs", len(df))
    with col2:
        st.metric("Final Loss", f"{df['loss'].iloc[-1]:.6f}")
    with col3:
        st.metric("Best Loss", f"{df['loss'].min():.6f}")

def text_generation_page():
    st.header("📝 Advanced Text Generation & Refinement")
    
    if not st.session_state.model_trained or not st.session_state.advanced_model:
        st.error("Please train the model first using the Model Training page.")
        return
    
    # Generation modes
    generation_mode = st.selectbox(
        "Generation Mode",
        ["Text Refinement", "Text Generation", "Financial Report Generation"]
    )
    
    if generation_mode == "Text Refinement":
        text_refinement_interface()
    elif generation_mode == "Text Generation":
        text_generation_interface()
    elif generation_mode == "Financial Report Generation":
        report_generation_interface()

def text_refinement_interface():
    """Interface for text refinement"""
    st.subheader("Financial Text Refinement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Input Text**")
        
        # Sample drafts
        sample_drafts = [
            "Company did good this quarter. Revenue up.",
            "Stock price went up. Investors happy. Future looks bright.",
            "Earnings beat expectations. Growth strong.",
            "Market tough but we doing fine. Plan working.",
            "New product selling well. Customers like it."
        ]
        
        selected_draft = st.selectbox("Sample Draft Texts", [""] + sample_drafts)
        
        input_text = st.text_area(
            "Enter draft financial text:",
            value=selected_draft,
            height=200,
            placeholder="Enter your draft financial text here..."
        )
        
        # Refinement parameters
        num_inference_steps = st.slider("Refinement Steps", 10, 100, 50)
        
        refine_button = st.button("🔄 Refine Text", type="primary")
    
    with col2:
        st.write("**Refined Text**")
        
        if refine_button and input_text.strip():
            with st.spinner("Refining text using advanced diffusion model..."):
                try:
                    refined_text = st.session_state.advanced_model.refine_text(
                        input_text, 
                        num_inference_steps
                    )
                    
                    st.text_area(
                        "Refined output:",
                        value=refined_text,
                        height=200,
                        disabled=True
                    )
                    
                    # Show improvement metrics
                    st.subheader("Refinement Analysis")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Original Length", len(input_text.split()))
                        st.metric("Refined Length", len(refined_text.split()))
                    
                    with col_b:
                        improvement_ratio = len(refined_text.split()) / max(1, len(input_text.split()))
                        st.metric("Length Improvement", f"{improvement_ratio:.2f}x")
                        
                        # Simple quality score
                        quality_score = min(1.0, len(refined_text.split()) / 20.0)
                        st.metric("Quality Score", f"{quality_score:.3f}")
                
                except Exception as e:
                    st.error(f"Refinement failed: {str(e)}")
        
        elif refine_button:
            st.warning("Please enter some text to refine.")

def text_generation_interface():
    """Interface for text generation"""
    st.subheader("Financial Text Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Generation Prompt**")
        
        # Sample prompts
        sample_prompts = [
            "The company's quarterly performance",
            "Market analysis indicates",
            "Investment outlook for",
            "Financial results show",
            "Economic conditions suggest"
        ]
        
        selected_prompt = st.selectbox("Sample Prompts", [""] + sample_prompts)
        
        prompt_text = st.text_area(
            "Enter generation prompt:",
            value=selected_prompt,
            height=100,
            placeholder="Enter a prompt to generate financial text..."
        )
        
        # Generation parameters
        max_length = st.slider("Maximum Length", 50, 500, 100)
        num_inference_steps = st.slider("Generation Steps", 20, 100, 50)
        
        generate_button = st.button("🎯 Generate Text", type="primary")
    
    with col2:
        st.write("**Generated Text**")
        
        if generate_button and prompt_text.strip():
            with st.spinner("Generating financial text..."):
                try:
                    generated_text = st.session_state.advanced_model.generate_text(
                        prompt_text,
                        max_length,
                        num_inference_steps
                    )
                    
                    st.text_area(
                        "Generated output:",
                        value=generated_text,
                        height=200,
                        disabled=True
                    )
                    
                    # Generation statistics
                    st.subheader("Generation Statistics")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Prompt Length", len(prompt_text.split()))
                        st.metric("Generated Length", len(generated_text.split()))
                    
                    with col_b:
                        expansion_ratio = len(generated_text.split()) / max(1, len(prompt_text.split()))
                        st.metric("Expansion Ratio", f"{expansion_ratio:.2f}x")
                        st.metric("Steps Used", num_inference_steps)
                
                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")
        
        elif generate_button:
            st.warning("Please enter a prompt to generate text.")

def report_generation_interface():
    """Interface for financial report generation"""
    st.subheader("Financial Report Generation")
    
    # Report types
    report_type = st.selectbox(
        "Report Type",
        ["Quarterly Earnings Summary", "Market Analysis Report", "Investment Outlook", "Risk Assessment"]
    )
    
    # Company context
    company_symbol = st.text_input("Company Symbol (optional)", placeholder="e.g., AAPL")
    
    if st.button("📊 Generate Financial Report", type="primary"):
        with st.spinner("Generating comprehensive financial report..."):
            try:
                # Create context-aware prompt
                if company_symbol:
                    prompt = f"Generate a {report_type.lower()} for {company_symbol}:"
                else:
                    prompt = f"Generate a {report_type.lower()}:"
                
                # Generate report
                report = st.session_state.advanced_model.generate_text(
                    prompt,
                    max_length=300,
                    num_inference_steps=75
                )
                
                st.subheader(f"{report_type}")
                if company_symbol:
                    st.subheader(f"Company: {company_symbol}")
                
                st.write(report)
                
                # Download option
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            
            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")

def model_management_page():
    st.header("⚙️ Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Model")
        
        if st.session_state.advanced_model:
            model_info = st.session_state.advanced_model.get_model_info()
            st.json(model_info)
            
            # Save model
            if st.button("💾 Save Current Model"):
                save_current_model()
        else:
            st.info("No model loaded")
    
    with col2:
        st.subheader("Load Model")
        
        # List available models
        model_files = [f for f in os.listdir('.') if f.endswith('.json') and 'financial_diffusion' in f]
        
        if model_files:
            selected_model = st.selectbox("Available Models", model_files)
            
            if st.button("📂 Load Model"):
                load_saved_model(selected_model)
        else:
            st.info("No saved models found")

def save_current_model():
    """Save current model"""
    if not st.session_state.advanced_model:
        st.error("No model to save")
        return
    
    try:
        filename = f"financial_diffusion_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.session_state.advanced_model.save_model(filename)
        st.success(f"Model saved as {filename}")
    except Exception as e:
        st.error(f"Failed to save model: {str(e)}")

def load_saved_model(filename):
    """Load saved model"""
    try:
        model = FinancialDiffusionLLM()
        model.load_model(filename)
        st.session_state.advanced_model = model
        st.session_state.model_trained = model.is_trained
        st.success(f"Model loaded from {filename}")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")

def database_analytics_page():
    st.header("📈 Database Analytics")
    
    if not st.session_state.data_manager:
        st.error("Database not connected")
        return
    
    try:
        # Get statistics
        stats = st.session_state.data_manager.get_training_statistics()
        
        if stats:
            # Overview metrics
            st.subheader("Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Companies", stats.get('total_companies', 0))
                st.metric("Total News", stats.get('total_news', 0))
            
            with col2:
                st.metric("Earnings Records", stats.get('total_earnings', 0))
                st.metric("SEC Filings", stats.get('total_sec_filings', 0))
            
            with col3:
                st.metric("Market Indicators", stats.get('total_market_indicators', 0))
                st.metric("Training Texts", stats.get('total_training_texts', 0))
            
            with col4:
                st.metric("Model Checkpoints", stats.get('total_model_checkpoints', 0))
                total_data_points = sum([v for k, v in stats.items() if k.startswith('total_')])
                st.metric("Total Data Points", total_data_points)
            
            # Data refresh
            if st.button("🔄 Refresh Analytics"):
                st.rerun()
        
        else:
            st.warning("No data available. Please collect data first.")
    
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")

if __name__ == "__main__":
    main()