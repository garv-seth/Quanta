import streamlit as st

# Set page configuration first
st.set_page_config(
    page_title="Advanced Financial Diffusion LLM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Import data collection functions directly
import yfinance as yf
import requests
import feedparser
from bs4 import BeautifulSoup

# Import Quasar Pre-trained model
try:
    from models.quasar_pretrained import QuasarPretrainedModel, QuasarFactory
    QUASAR_AVAILABLE = True
except ImportError:
    QUASAR_AVAILABLE = False
    QuasarPretrainedModel = None
    QuasarFactory = None

# Import our modules with individual error handling
ADVANCED_MODULES_AVAILABLE = True
FinancialDiffusionLLM = None
SimpleTextProcessor = None

try:
    from models.simple_financial_diffusion import SimpleFinancialDiffusion
    DIFFUSION_MODEL_AVAILABLE = True
except ImportError as e:
    DIFFUSION_MODEL_AVAILABLE = False
    SimpleFinancialDiffusion = None

try:
    from models.advanced.financial_diffusion_llm import FinancialDiffusionLLM
    ADVANCED_MODULES_AVAILABLE = True
except ImportError as e:
    ADVANCED_MODULES_AVAILABLE = False
    FinancialDiffusionLLM = None

try:
    from utils.simple_text_processor import SimpleTextProcessor
    TEXT_PROCESSOR_AVAILABLE = True
except ImportError as e:
    TEXT_PROCESSOR_AVAILABLE = False
    SimpleTextProcessor = None

# Database imports with error handling
DATABASE_AVAILABLE = False
RealFinancialDataManager = None

try:
    from database.real_data_manager import RealFinancialDataManager
    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False

# Initialize session state
def initialize_session_state():
    if 'quasar_model' not in st.session_state:
        st.session_state.quasar_model = None
    if 'advanced_model' not in st.session_state:
        st.session_state.advanced_model = None
    if 'training_data' not in st.session_state:
        st.session_state.training_data = []
    if 'collected_financial_data' not in st.session_state:
        st.session_state.collected_financial_data = {}
    if 'database_connected' not in st.session_state:
        st.session_state.database_connected = False
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = None

# Initialize database connection
def initialize_database():
    if DATABASE_AVAILABLE and not st.session_state.database_connected:
        try:
            st.session_state.data_manager = RealFinancialDataManager()
            st.session_state.database_connected = True
            return True
        except Exception as e:
            st.session_state.database_connected = False
            return False
    return st.session_state.database_connected

# Initialize everything
initialize_session_state()

# Streamlit App
def main():
    st.title("🧠 Advanced Financial Diffusion LLM")
    st.markdown("*Pre-trained models ready for financial analysis and text generation*")

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Model status in sidebar
    st.sidebar.subheader("🤖 Model Status")

    if QUASAR_AVAILABLE:
        st.sidebar.success("✅ Quasar Pre-trained Available")
    else:
        st.sidebar.error("❌ Quasar Pre-trained Unavailable")

    if st.session_state.quasar_model:
        st.sidebar.info("🔥 Quasar Model Loaded")
    elif st.session_state.advanced_model:
        st.sidebar.info("🔧 Custom Model Loaded")
    else:
        st.sidebar.warning("⚠️ No Model Loaded")

    # Database status
    if initialize_database():
        st.sidebar.success("✅ Database Connected")
    else:
        st.sidebar.warning("⚠️ Database Unavailable")

    # Page selection
    page_options = [
        "🏠 Pre-trained Models Hub",
        "🧪 Interactive Playground", 
        "📊 Live Financial Data",
        "⚙️ Model Fine-tuning",
        "📈 Financial Analysis",
        "🔧 Model Management"
    ]

    selected_page = st.sidebar.selectbox("Select Page", page_options)

    # Route to selected page
    if selected_page == "🏠 Pre-trained Models Hub":
        pretrained_models_hub()
    elif selected_page == "🧪 Interactive Playground":
        interactive_playground()
    elif selected_page == "📊 Live Financial Data":
        live_financial_data_page()
    elif selected_page == "⚙️ Model Fine-tuning":
        model_finetuning_page()
    elif selected_page == "📈 Financial Analysis":
        financial_analysis_page()
    elif selected_page == "🔧 Model Management":
        model_management_page()

def pretrained_models_hub():
    """Pre-trained models hub with immediate functionality"""
    st.header("🏠 Pre-trained Models Hub")

    st.markdown("""
    Welcome to the **Quasar Financial AI** models hub! These models come pre-trained on extensive financial data 
    and are ready to use immediately. No training required!
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌟 Quasar Small")
        st.info("**2.1M Parameters**\nFast inference, optimized for real-time analysis")

        if st.button("🚀 Load Quasar Small", type="primary"):
            if QUASAR_AVAILABLE:
                with st.spinner("Loading pre-trained model..."):
                    st.session_state.quasar_model = QuasarFactory.create_small()
                    st.success("✅ Quasar Small loaded successfully!")
                    st.rerun()
            else:
                st.error("Quasar models not available")

    with col2:
        st.subheader("🔥 Quasar Medium")
        st.info("**8.2M Parameters**\nBetter performance, deeper financial understanding")

        if st.button("🚀 Load Quasar Medium"):
            if QUASAR_AVAILABLE:
                with st.spinner("Loading pre-trained model..."):
                    st.session_state.quasar_model = QuasarFactory.create_medium()
                    st.success("✅ Quasar Medium loaded successfully!")
                    st.rerun()
            else:
                st.error("Quasar models not available")

    with col3:
        st.subheader("🛠️ Custom Models")
        st.info("Load your own fine-tuned models")

        model_files = [f for f in os.listdir('.') if f.endswith('.json') and 'quasar' in f.lower()]
        if model_files:
            selected_file = st.selectbox("Saved Models", model_files)
            if st.button("📂 Load Custom"):
                if QUASAR_AVAILABLE:
                    try:
                        st.session_state.quasar_model = QuasarFactory.load_from_file(selected_file)
                        st.success(f"✅ Loaded {selected_file}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load: {str(e)}")
        else:
            st.info("No saved models found")

    # Quick Demo Section
    if st.session_state.quasar_model:
        st.divider()
        st.subheader("🎯 Quick Demo")

        demo_col1, demo_col2 = st.columns(2)

        with demo_col1:
            st.markdown("**Financial Text Generation**")
            demo_prompt = st.text_input("Enter a financial prompt:", 
                                      value="Apple's quarterly earnings show")

            generation_temp = st.slider("Creativity Level", 0.1, 1.5, 0.8, 0.1)
            use_live_data = st.checkbox("Use Live Market Context", value=True)

            if st.button("Generate Text", type="primary"):
                with st.spinner("Generating..."):
                    generated = st.session_state.quasar_model.generate_text(
                        demo_prompt, 
                        max_length=100,
                        temperature=generation_temp,
                        use_market_context=use_live_data
                    )
                    st.write("**Generated Text:**")
                    st.write(generated)

        with demo_col2:
            st.markdown("**Financial Sentiment Analysis**")
            sentiment_text = st.text_area("Text to analyze:", 
                                        value="The company reported strong revenue growth with positive outlook for next quarter")

            if st.button("Analyze Sentiment"):
                sentiment = st.session_state.quasar_model.analyze_financial_sentiment(sentiment_text)

                # Display sentiment scores
                col_pos, col_neg, col_neu = st.columns(3)
                with col_pos:
                    st.metric("Positive", f"{sentiment['positive']:.2%}", delta=None)
                with col_neg:
                    st.metric("Negative", f"{sentiment['negative']:.2%}", delta=None)
                with col_neu:
                    st.metric("Neutral", f"{sentiment['neutral']:.2%}", delta=None)

        # Model Information
        st.divider()
        st.subheader("📋 Current Model Info")
        model_info = st.session_state.quasar_model.get_model_info()

        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Parameters", model_info.get('total_parameters', 'N/A'))
        with info_col2:
            st.metric("Vocab Size", model_info.get('vocab_size', 'N/A'))
        with info_col3:
            st.metric("Training Epochs", len(st.session_state.quasar_model.training_history))

def interactive_playground():
    """Interactive playground for experimenting with the model"""
    st.header("🧪 Interactive Playground")

    if not st.session_state.quasar_model:
        st.warning("⚠️ Please load a model from the Pre-trained Models Hub first")
        return

    st.markdown("Experiment with different model capabilities and parameters!")

    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Generation", "✨ Text Refinement", "📊 Analysis", "🎛️ Advanced Controls"])

    with tab1:
        st.subheader("Financial Text Generation")

        col1, col2 = st.columns([2, 1])

        with col1:
            prompt = st.text_area("Enter your prompt:", 
                                value="The Federal Reserve's recent interest rate decision",
                                height=100)

            if st.button("🚀 Generate", type="primary", key="gen_btn"):
                with st.spinner("Generating financial content..."):
                    result = st.session_state.quasar_model.generate_text(
                        prompt,
                        max_length=st.session_state.get('gen_length', 150),
                        temperature=st.session_state.get('gen_temp', 0.8),
                        use_market_context=st.session_state.get('use_market', True)
                    )
                    st.subheader("Generated Text:")
                    st.write(result)

                    # Option to save
                    if st.button("💾 Save Result"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        with open(f"generated_text_{timestamp}.txt", "w") as f:
                            f.write(f"Prompt: {prompt}\n\nGenerated Text:\n{result}")
                        st.success("Saved to file!")

        with col2:
            st.markdown("**Generation Settings**")
            gen_length = st.slider("Max Length", 50, 300, 150, key="gen_length")
            gen_temp = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1, key="gen_temp")
            use_market = st.checkbox("Live Market Context", True, key="use_market")

            st.markdown("**Quick Prompts**")
            quick_prompts = [
                "Quarterly earnings report summary:",
                "Stock market outlook for",
                "Economic indicators suggest",
                "Investment recommendation for",
                "Risk assessment shows"
            ]

            for quick_prompt in quick_prompts:
                if st.button(f"📌 {quick_prompt}", key=f"quick_{quick_prompt}"):
                    st.session_state.current_prompt = quick_prompt

    with tab2:
        st.subheader("Text Refinement")

        input_text = st.text_area("Text to refine:", 
                                value="The stock did good this quarter and made money",
                                height=100)

        col1, col2 = st.columns(2)

        with col1:
            refinement_strength = st.slider("Refinement Strength", 0.1, 1.0, 0.3, 0.1)
            inference_steps = st.slider("Inference Steps", 10, 50, 25, 5)

        with col2:
            if st.button("✨ Refine Text", type="primary"):
                with st.spinner("Refining text..."):
                    refined = st.session_state.quasar_model.refine_text(
                        input_text,
                        refinement_strength=refinement_strength,
                        num_inference_steps=inference_steps
                    )

                    st.subheader("Refined Text:")
                    st.write(refined)

                    st.subheader("Comparison:")
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.markdown("**Before:**")
                        st.write(input_text)
                    with col_after:
                        st.markdown("**After:**")
                        st.write(refined)

    with tab3:
        st.subheader("Financial Analysis")

        analysis_text = st.text_area("Text to analyze:",
                                   value="Apple reported record revenue this quarter with strong iPhone sales driving growth",
                                   height=100)

        if st.button("📊 Analyze", type="primary"):
            # Sentiment Analysis
            sentiment = st.session_state.quasar_model.analyze_financial_sentiment(analysis_text)

            st.subheader("Sentiment Analysis")

            # Create sentiment chart
            sentiment_data = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                'Score': [sentiment['positive'], sentiment['negative'], sentiment['neutral']]
            })

            st.bar_chart(sentiment_data.set_index('Sentiment'))

            # Detailed metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive Sentiment", f"{sentiment['positive']:.1%}")
            with col2:
                st.metric("Negative Sentiment", f"{sentiment['negative']:.1%}")
            with col3:
                st.metric("Neutral Sentiment", f"{sentiment['neutral']:.1%}")

    with tab4:
        st.subheader("Advanced Model Controls")

        # Model parameters
        st.markdown("**Current Model Parameters**")
        model_info = st.session_state.quasar_model.get_model_info()

        col1, col2 = st.columns(2)
        with col1:
            st.json({
                'Model Type': model_info.get('model_type', 'Unknown'),
                'Vocab Size': model_info.get('vocab_size', 0),
                'Embedding Dim': model_info.get('d_model', 0),
                'Num Layers': model_info.get('num_layers', 0)
            })

        with col2:
            st.markdown("**Training History**")
            if hasattr(st.session_state.quasar_model, 'training_history'):
                history_df = pd.DataFrame(st.session_state.quasar_model.training_history)
                if not history_df.empty and 'loss' in history_df.columns:
                    st.line_chart(history_df.set_index('epoch')['loss'])
                else:
                    st.info("No training history available")

        # Save current model
        st.divider()
        st.markdown("**Save Model**")
        model_name = st.text_input("Model name:", value=f"quasar_custom_{datetime.now().strftime('%Y%m%d')}")

        if st.button("💾 Save Current Model"):
            filename = f"{model_name}.json"
            st.session_state.quasar_model.save_model(filename)
            st.success(f"Model saved as {filename}")

def live_financial_data_page():
    """Live financial data collection and display"""
    st.header("📊 Live Financial Data")

    st.markdown("Real-time financial data collection from Yahoo Finance and other sources.")

    # Data collection controls
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Collect Live Data", type="primary"):
            with st.spinner("Collecting live financial data..."):
                data = collect_real_financial_data()
                st.session_state.collected_financial_data = data
                st.success("Live data collected successfully!")
                st.rerun()

    with col2:
        if st.button("📈 Update Market Context"):
            if st.session_state.quasar_model:
                with st.spinner("Updating market context..."):
                    st.session_state.quasar_model._load_market_context()
                    st.success("Market context updated!")

    with col3:
        if st.button("🗄️ Store in Database"):
            if st.session_state.database_connected and st.session_state.collected_financial_data:
                with st.spinner("Storing data..."):
                    try:
                        results = st.session_state.data_manager.collect_all_live_data()
                        st.success(f"Stored: {results}")
                    except Exception as e:
                        st.error(f"Database error: {str(e)}")

    # Display collected data
    if st.session_state.collected_financial_data:
        display_financial_data(st.session_state.collected_financial_data)
    else:
        st.info("No financial data collected yet. Click 'Collect Live Data' to get started.")

def collect_real_financial_data():
    """Collect real financial data"""
    try:
        # Major stock symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']

        companies = []
        for symbol in symbols[:5]:  # Limit to 5 for demo
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                companies.append({
                    'symbol': symbol,
                    'company_name': info.get('longName', symbol),
                    'sector': info.get('sector', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'current_price': info.get('currentPrice', 0),
                    'pe_ratio': info.get('trailingPE', 0)
                })
            except:
                continue

        # Market indices
        indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']
        market_data = []

        for index in indices:
            try:
                ticker = yf.Ticker(index)
                hist = ticker.history(period='2d')
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else latest
                    change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

                    market_data.append({
                        'symbol': index,
                        'current_value': float(latest['Close']),
                        'change_percent': float(change),
                        'volume': int(latest['Volume']) if 'Volume' in latest.index else 0
                    })
            except:
                continue

        return {
            'companies': companies,
            'market_indicators': market_data,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        st.error(f"Error collecting data: {str(e)}")
        return {}

def display_financial_data(data):
    """Display collected financial data"""

    # Companies data
    if data.get('companies'):
        st.subheader("📊 Company Data")
        companies_df = pd.DataFrame(data['companies'])

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_market_cap = companies_df['market_cap'].mean() if 'market_cap' in companies_df.columns else 0
            st.metric("Avg Market Cap", f"${avg_market_cap/1e9:.1f}B")
        with col2:
            st.metric("Companies", len(companies_df))
        with col3:
            sectors = companies_df['sector'].nunique() if 'sector' in companies_df.columns else 0
            st.metric("Sectors", sectors)
        with col4:
            avg_pe = companies_df['pe_ratio'].mean() if 'pe_ratio' in companies_df.columns else 0
            st.metric("Avg P/E", f"{avg_pe:.1f}")

        st.dataframe(companies_df, use_container_width=True)

    # Market indicators
    if data.get('market_indicators'):
        st.subheader("📈 Market Indicators")
        market_df = pd.DataFrame(data['market_indicators'])

        for _, indicator in market_df.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                color = "🟢" if indicator['change_percent'] > 0 else "🔴" if indicator['change_percent'] < 0 else "🟡"
                st.write(f"{color} **{indicator['symbol']}**")
            with col2:
                st.metric(
                    f"{indicator['current_value']:.2f}",
                    f"{indicator['change_percent']:+.2f}%"
                )

def model_finetuning_page():
    """Model fine-tuning interface"""
    st.header("⚙️ Model Fine-tuning")

    if not st.session_state.quasar_model:
        st.warning("⚠️ Please load a model first")
        return

    st.markdown("Fine-tune your pre-trained model on specific financial data for enhanced performance.")

    # Fine-tuning configuration
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Training Data")

        data_source = st.radio("Data Source:", 
                              ["Use Live Financial Data", "Upload Custom Text", "Manual Input"])

        training_texts = []

        if data_source == "Use Live Financial Data":
            if st.session_state.collected_financial_data:
                # Extract training texts from collected data
                for company in st.session_state.collected_financial_data.get('companies', []):
                    training_texts.append(f"{company['company_name']} operates in the {company['sector']} sector with a market cap of ${company.get('market_cap', 0):,}")

                st.info(f"Prepared {len(training_texts)} training texts from live data")
            else:
                st.warning("No live data available. Please collect data first.")

        elif data_source == "Upload Custom Text":
            uploaded_file = st.file_uploader("Upload text file", type=['txt'])
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                training_texts = [line.strip() for line in content.split('\n') if line.strip()]
                st.info(f"Loaded {len(training_texts)} lines from file")

        else:  # Manual Input
            manual_text = st.text_area("Enter training text (one sentence per line):", 
                                     height=150,
                                     value="Apple reported strong quarterly earnings.\nThe tech sector shows continued growth.\nMarket volatility remains a concern for investors.")
            training_texts = [line.strip() for line in manual_text.split('\n') if line.strip()]

    with col2:
        st.subheader("Fine-tuning Settings")

        epochs = st.slider("Training Epochs", 1, 10, 3)
        learning_rate = st.select_slider("Learning Rate", 
                                       options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001],
                                       value=0.0001,
                                       format_func=lambda x: f"{x:.5f}")

        st.info(f"Pre-trained models typically need only {epochs} epochs for effective fine-tuning")

        # Start fine-tuning
        if st.button("🔥 Start Fine-tuning", type="primary"):
            if training_texts:
                with st.spinner(f"Fine-tuning model on {len(training_texts)} texts..."):
                    losses = st.session_state.quasar_model.fine_tune(
                        training_texts, 
                        epochs=epochs, 
                        learning_rate=learning_rate
                    )

                    st.success("Fine-tuning completed!")

                    # Display training progress
                    st.subheader("Training Progress")
                    loss_df = pd.DataFrame({'Epoch': range(len(losses)), 'Loss': losses})
                    st.line_chart(loss_df.set_index('Epoch'))

                    # Save fine-tuned model
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"quasar_finetuned_{timestamp}.json"
                    st.session_state.quasar_model.save_model(filename)
                    st.info(f"Fine-tuned model saved as {filename}")
            else:
                st.error("No training texts available")

def financial_analysis_page():
    """Advanced financial analysis features"""
    st.header("📈 Financial Analysis")

    if not st.session_state.quasar_model:
        st.warning("⚠️ Please load a model first")
        return

    st.markdown("Perform advanced financial analysis using your diffusion model.")

    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Market Analysis", "📄 Report Generation", "🎯 Sentiment Analysis"])

    with tab1:
        st.subheader("Market Analysis")

        if hasattr(st.session_state.quasar_model, 'market_context'):
            market_context = st.session_state.quasar_model.market_context

            st.markdown("**Current Market Overview**")

            for index, data in market_context.items():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{index}", f"{data.get('current', 0):.2f}")
                with col2:
                    st.metric("Change", f"{data.get('change', 0):+.2f}%")
                with col3:
                    trend_color = "🟢" if data.get('trend') == 'positive' else "🔴" if data.get('trend') == 'negative' else "🟡"
                    st.write(f"{trend_color} {data.get('trend', 'neutral').title()}")

        # Generate market analysis
        if st.button("📊 Generate Market Analysis", type="primary"):
            with st.spinner("Analyzing market conditions..."):
                analysis_prompt = "Based on current market conditions, provide a comprehensive market analysis including key trends and outlook"
                analysis = st.session_state.quasar_model.generate_text(
                    analysis_prompt,
                    max_length=200,
                    use_market_context=True
                )

                st.subheader("Market Analysis Report")
                st.write(analysis)

    with tab2:
        st.subheader("Financial Report Generation")

        report_type = st.selectbox("Report Type", 
                                 ["Quarterly Earnings Summary", "Market Outlook", "Investment Analysis", "Risk Assessment"])

        company_input = st.text_input("Company/Symbol (optional):", placeholder="e.g., AAPL")

        if st.button("📄 Generate Report", type="primary"):
            with st.spinner("Generating financial report..."):
                if company_input:
                    prompt = f"Generate a detailed {report_type.lower()} for {company_input}"
                else:
                    prompt = f"Generate a comprehensive {report_type.lower()}"

                report = st.session_state.quasar_model.generate_text(
                    prompt,
                    max_length=250,
                    temperature=0.7,
                    use_market_context=True
                )

                st.subheader(f"{report_type}")
                if company_input:
                    st.markdown(f"**Subject:** {company_input}")

                st.write(report)

                # Download option
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

    with tab3:
        st.subheader("Batch Sentiment Analysis")

        # Input methods
        input_method = st.radio("Input Method:", ["Text Area", "File Upload"])

        texts_to_analyze = []

        if input_method == "Text Area":
            batch_text = st.text_area("Enter texts to analyze (one per line):", 
                                    height=150,
                                    value="Apple's revenue exceeded expectations this quarter\nMarket volatility creates uncertainty for investors\nStrong employment data supports economic growth")
            texts_to_analyze = [line.strip() for line in batch_text.split('\n') if line.strip()]

        else:
            uploaded_file = st.file_uploader("Upload text file", type=['txt'])
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                texts_to_analyze = [line.strip() for line in content.split('\n') if line.strip()]

        if texts_to_analyze and st.button("🎯 Analyze Sentiment", type="primary"):
            with st.spinner(f"Analyzing sentiment for {len(texts_to_analyze)} texts..."):
                results = []

                for i, text in enumerate(texts_to_analyze):
                    sentiment = st.session_state.quasar_model.analyze_financial_sentiment(text)
                    results.append({
                        'Text': text[:50] + "..." if len(text) > 50 else text,
                        'Positive': sentiment['positive'],
                        'Negative': sentiment['negative'],
                        'Neutral': sentiment['neutral'],
                        'Dominant': max(sentiment.keys(), key=sentiment.get)
                    })

                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                # Summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_positive = results_df['Positive'].mean()
                    st.metric("Avg Positive", f"{avg_positive:.2%}")
                with col2:
                    avg_negative = results_df['Negative'].mean()
                    st.metric("Avg Negative", f"{avg_negative:.2%}")
                with col3:
                    avg_neutral = results_df['Neutral'].mean()
                    st.metric("Avg Neutral", f"{avg_neutral:.2%}")

def model_management_page():
    """Model management interface"""
    st.header("🔧 Model Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Model")

        if st.session_state.quasar_model:
            model_info = st.session_state.quasar_model.get_model_info()
            st.json(model_info)

            # Model operations
            st.subheader("Model Operations")

            model_name = st.text_input("Save as:", value=f"quasar_{datetime.now().strftime('%Y%m%d')}")

            if st.button("💾 Save Model"):
                filename = f"{model_name}.json"
                st.session_state.quasar_model.save_model(filename)
                st.success(f"Model saved as {filename}")

            if st.button("🔄 Reset Model"):
                if st.confirm("Are you sure you want to reset the model?"):
                    st.session_state.quasar_model = None
                    st.success("Model reset successfully")
                    st.rerun()
        else:
            st.info("No model loaded")

    with col2:
        st.subheader("Available Models")

        # List saved models
        model_files = [f for f in os.listdir('.') if f.endswith('.json') and ('quasar' in f.lower() or 'financial' in f.lower())]

        if model_files:
            for model_file in model_files:
                with st.expander(f"📄 {model_file}"):
                    # File info
                    file_size = os.path.getsize(model_file) / 1024  # KB
                    st.write(f"**Size:** {file_size:.1f} KB")

                    col_load, col_delete = st.columns(2)
                    with col_load:
                        if st.button(f"📂 Load", key=f"load_{model_file}"):
                            if QUASAR_AVAILABLE:
                                try:
                                    st.session_state.quasar_model = QuasarFactory.load_from_file(model_file)
                                    st.success(f"Loaded {model_file}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to load: {str(e)}")

                    with col_delete:
                        if st.button(f"🗑️ Delete", key=f"delete_{model_file}"):
                            if st.confirm(f"Delete {model_file}?"):
                                os.remove(model_file)
                                st.success(f"Deleted {model_file}")
                                st.rerun()
        else:
            st.info("No saved models found")

        # Model comparison
        if len(model_files) > 1:
            st.subheader("Model Comparison")
            selected_models = st.multiselect("Select models to compare:", model_files)

            if len(selected_models) >= 2 and st.button("📊 Compare Models"):
                comparison_data = []
                for model_file in selected_models:
                    try:
                        # Load model info without fully loading the model
                        with open(model_file, 'r') as f:
                            model_data = json.load(f)

                        comparison_data.append({
                            'Model': model_file,
                            'Vocab Size': model_data.get('config', {}).get('vocab_size', 'N/A'),
                            'Embedding Dim': model_data.get('config', {}).get('d_model', 'N/A'),
                            'Layers': model_data.get('config', {}).get('num_layers', 'N/A'),
                            'Training Epochs': len(model_data.get('training_history', []))
                        })
                    except:
                        continue

                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

if __name__ == "__main__":
    main()