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
except ImportError as e:
    SimpleTextProcessor = None

try:
    from database.real_data_manager import RealFinancialDataManager
    from database.init_db import setup_database
    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False
    RealFinancialDataManager = None

# Simplified data manager for database operations
class SimpleDataManager:
    def __init__(self):
        self.db_connected = False
        try:
            import os
            from database.schema import DatabaseManager
            self.db_manager = DatabaseManager()
            self.db_manager.create_tables()
            self.db_connected = True
        except Exception as e:
            st.warning(f"Database connection failed: {e}")
    
    def collect_and_store_sample_data(self):
        """Generate sample financial data for training"""
        return {
            'companies': 10,
            'news': 25,
            'sec_filings': 5,
            'market_indicators': 4
        }
    
    def prepare_sample_training_texts(self):
        """Return sample financial training texts"""
        return [
            "The company reported strong quarterly earnings with revenue growth of 15% year-over-year and improved operating margins across all business segments.",
            "Market volatility increased following Federal Reserve policy announcements regarding interest rate adjustments and inflation concerns.",
            "Investment portfolio performance exceeded expectations with returns of 12% driven by strategic asset allocation and risk management practices.",
            "Operating cash flow remained robust at $250 million supporting continued business expansion and shareholder return initiatives.",
            "The merger and acquisition strategy strengthened market position while achieving cost synergies and operational efficiencies.",
            "Financial results demonstrated resilience despite challenging economic conditions with stable revenue and improved profit margins.",
            "Strategic investments in technology and innovation delivered measurable returns through automation and operational improvements.",
            "Balance sheet fundamentals remained strong with improved debt-to-equity ratios and enhanced liquidity positions.",
            "Risk management protocols effectively mitigated market exposure while maintaining growth opportunities and competitive advantages.",
            "Quarterly guidance reflects management confidence in sustainable business model and long-term value creation strategies."
        ]
    
    def get_training_statistics(self):
        """Return sample statistics"""
        return {
            'total_companies': 50,
            'total_news': 150,
            'total_earnings': 25,
            'total_sec_filings': 15,
            'total_market_indicators': 20,
            'total_training_texts': 100,
            'total_model_checkpoints': 3
        }

# Real financial data collection functions
def collect_real_financial_data():
    """Collect real financial data using Yahoo Finance and news APIs"""
    collected_data = {
        'companies': [],
        'news': [],
        'market_indicators': []
    }
    
    # Major S&P 500 companies for data collection
    major_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
    
    # Collect company data
    for symbol in major_symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            company_data = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'business_summary': info.get('longBusinessSummary', ''),
                'market_cap': info.get('marketCap', 0),
                'revenue': info.get('totalRevenue', 0),
                'collected_at': datetime.now().isoformat()
            }
            collected_data['companies'].append(company_data)
            
        except Exception as e:
            st.warning(f"Error collecting data for {symbol}: {str(e)}")
            continue
    
    # Collect market indicators
    indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']
    
    for index in indices:
        try:
            ticker = yf.Ticker(index)
            hist = ticker.history(period='2d')
            
            if not hist.empty:
                latest = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else latest
                
                change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
                
                indicator_data = {
                    'indicator': index,
                    'current_value': float(latest['Close']),
                    'previous_value': float(prev['Close']),
                    'change_percent': float(change),
                    'volume': int(latest['Volume']) if 'Volume' in latest.index else 0,
                    'analysis_date': str(hist.index[-1]),
                    'collected_at': datetime.now().isoformat()
                }
                collected_data['market_indicators'].append(indicator_data)
                
        except Exception as e:
            st.warning(f"Error collecting market data for {index}: {str(e)}")
            continue
    
    # Collect financial news
    try:
        # Try Yahoo Finance RSS feed
        feed = feedparser.parse('https://feeds.finance.yahoo.com/rss/2.0/headline')
        
        for entry in feed.entries[:10]:
            news_item = {
                'title': entry.get('title', ''),
                'summary': entry.get('summary', ''),
                'link': entry.get('link', ''),
                'published': entry.get('published', ''),
                'source': 'Yahoo Finance',
                'collected_at': datetime.now().isoformat()
            }
            collected_data['news'].append(news_item)
            
    except Exception as e:
        st.warning(f"Error collecting news: {str(e)}")
    
    return collected_data

def extract_training_texts_from_data(data):
    """Extract training texts from collected financial data"""
    training_texts = []
    
    # Extract from company business summaries
    for company in data.get('companies', []):
        summary = company.get('business_summary', '')
        if summary and len(summary.split()) > 10:
            training_texts.append(summary)
            
            # Create additional training text from company data
            market_cap = company.get('market_cap', 0)
            revenue = company.get('revenue', 0)
            
            if market_cap and revenue:
                performance_text = f"{company.get('company_name', '')} operates in the {company.get('sector', '')} sector with a market capitalization of ${market_cap:,.0f} and annual revenue of ${revenue:,.0f}."
                training_texts.append(performance_text)
    
    # Extract from news
    for news in data.get('news', []):
        title = news.get('title', '')
        summary = news.get('summary', '')
        
        if title and len(title.split()) > 5:
            training_texts.append(title)
        if summary and len(summary.split()) > 10:
            training_texts.append(summary)
    
    # Create market analysis texts
    for indicator in data.get('market_indicators', []):
        change_pct = indicator.get('change_percent', 0)
        indicator_name = indicator.get('indicator', '')
        current_value = indicator.get('current_value', 0)
        
        if change_pct > 0:
            market_text = f"The {indicator_name} index gained {change_pct:.2f}% to close at {current_value:.2f}, reflecting positive market sentiment and investor confidence."
        elif change_pct < 0:
            market_text = f"The {indicator_name} index declined {abs(change_pct):.2f}% to {current_value:.2f}, indicating market volatility and cautious investor behavior."
        else:
            market_text = f"The {indicator_name} index remained stable at {current_value:.2f}, showing balanced market conditions."
        
        training_texts.append(market_text)
    
    return [text for text in training_texts if text and len(text.split()) > 5]

# Page configuration is already set at the top of the file

# Initialize session state
if 'advanced_model' not in st.session_state:
    st.session_state.advanced_model = None
if 'data_manager' not in st.session_state:
    # Try to use real database manager first
    if DATABASE_AVAILABLE and RealFinancialDataManager:
        try:
            st.session_state.data_manager = RealFinancialDataManager()
            st.session_state.database_connected = True
        except Exception as e:
            st.session_state.data_manager = SimpleDataManager()
            st.session_state.database_connected = False
    else:
        st.session_state.data_manager = SimpleDataManager()
        st.session_state.database_connected = False
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
        if st.session_state.data_manager and hasattr(st.session_state.data_manager, 'db_connected') and st.session_state.data_manager.db_connected:
            st.success("✅ Database Connected")
        else:
            st.warning("⚠️ Using Sample Data")
    
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
    with st.spinner("Collecting live financial data from Yahoo Finance and storing in PostgreSQL..."):
        try:
            if st.session_state.database_connected and hasattr(st.session_state.data_manager, 'collect_all_live_data'):
                # Use real database collection
                counts = st.session_state.data_manager.collect_all_live_data()
                
                # Get training texts from database
                training_texts = st.session_state.data_manager.prepare_training_texts_from_db()
                st.session_state.training_data = training_texts
                
                # Get recent data for display
                recent_data = st.session_state.data_manager.get_recent_financial_data(limit=50)
                st.session_state.collected_financial_data = recent_data
                
                st.success("Live financial data collected and stored in PostgreSQL database!")
                
            else:
                # Fallback to in-memory collection
                dataset = collect_real_financial_data()
                
                counts = {
                    'companies': len(dataset.get('companies', [])),
                    'news': len(dataset.get('news', [])),
                    'sec_filings': 0,
                    'market_indicators': len(dataset.get('market_indicators', []))
                }
                
                training_texts = extract_training_texts_from_data(dataset)
                st.session_state.training_data = training_texts
                st.session_state.collected_financial_data = dataset
                
                st.success("Live financial data collected!")
                st.info("Database not connected - using in-memory storage")
            
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
            
            st.info(f"Prepared {len(st.session_state.training_data)} training texts from live data")
            
        except Exception as e:
            st.error(f"Data collection failed: {str(e)}")
            # Final fallback to sample data
            counts = st.session_state.data_manager.collect_and_store_sample_data()
            training_texts = st.session_state.data_manager.prepare_sample_training_texts()
            st.session_state.training_data = training_texts
            st.warning("Using sample data due to collection error")

def show_recent_data():
    """Display recent financial data"""
    if hasattr(st.session_state, 'collected_financial_data') and st.session_state.collected_financial_data:
        data = st.session_state.collected_financial_data
        
        # Display companies
        if data.get('companies'):
            st.subheader("📊 Live Company Data from Yahoo Finance")
            companies_df = pd.DataFrame(data['companies'])
            
            # Show key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_market_cap = companies_df['market_cap'].mean() if 'market_cap' in companies_df.columns and len(companies_df) > 0 else 0
                st.metric("Avg Market Cap", f"${int(avg_market_cap):,}")
            with col2:
                total_companies = len(companies_df)
                st.metric("Companies", total_companies)
            with col3:
                sectors = companies_df['sector'].nunique() if 'sector' in companies_df.columns and len(companies_df) > 0 else 0
                st.metric("Sectors", sectors)
            
            st.dataframe(companies_df[['symbol', 'company_name', 'sector', 'market_cap']], use_container_width=True)
        
        # Display market indicators
        if data.get('market_indicators'):
            st.subheader("📈 Live Market Indicators")
            market_df = pd.DataFrame(data['market_indicators'])
            
            for _, indicator in market_df.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    indicator_name = str(indicator['indicator'])
                    current_val = float(indicator['current_value'])
                    change_pct = float(indicator['change_percent'])
                    st.metric(
                        indicator_name,
                        f"{current_val:.2f}",
                        f"{change_pct:.2f}%"
                    )
                with col2:
                    if change_pct > 0:
                        st.success("📈")
                    else:
                        st.error("📉")
        
        # Display news
        if data.get('news'):
            st.subheader("📰 Latest Financial News")
            news_df = pd.DataFrame(data['news'])
            
            for _, news in news_df.iterrows():
                title = str(news['title'])[:100] if news['title'] else "No title"
                with st.expander(f"📰 {title}..."):
                    st.write(f"**Source:** {str(news['source'])}")
                    st.write(f"**Published:** {str(news['published'])}")
                    if news['summary']:
                        st.write(f"**Summary:** {str(news['summary'])}")
                    if news['link']:
                        st.write(f"[Read full article]({str(news['link'])})")
        
        # Display training data preview
        if hasattr(st.session_state, 'training_data') and st.session_state.training_data:
            st.subheader("🧠 Training Data Preview")
            st.info(f"Total training texts prepared: {len(st.session_state.training_data)}")
            
            # Show first few training texts
            with st.expander("View Sample Training Texts"):
                for i, text in enumerate(st.session_state.training_data[:5]):
                    st.write(f"**Text {i+1}:** {text[:200]}...")
    else:
        st.warning("No financial data collected yet. Please collect data first from the Data Collection page.")
        if st.button("Collect Live Data Now"):
            # Collect real data
            dataset = collect_real_financial_data()
            st.session_state.collected_financial_data = dataset
            training_texts = extract_training_texts_from_data(dataset)
            st.session_state.training_data = training_texts
            st.rerun()

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
    """Train the financial diffusion model"""
    try:
        # Use simplified diffusion model if available
        if DIFFUSION_MODEL_AVAILABLE and SimpleFinancialDiffusion:
            st.session_state.advanced_model = SimpleFinancialDiffusion(
                vocab_size=vocab_size,
                embedding_dim=d_model,
                num_steps=50
            )
        elif ADVANCED_MODULES_AVAILABLE and FinancialDiffusionLLM:
            st.session_state.advanced_model = FinancialDiffusionLLM(
                vocab_size=vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_length=max_seq_length,
                num_diffusion_steps=1000
            )
        else:
            st.error("No diffusion model available for training")
            return
        
        # Get real financial training data from PostgreSQL database
        training_texts = []
        
        if use_database and st.session_state.database_connected:
            try:
                # First, ensure we have fresh data
                if hasattr(st.session_state.data_manager, 'collect_all_live_data'):
                    st.info("Collecting fresh Yahoo Finance data...")
                    results = st.session_state.data_manager.collect_all_live_data()
                    st.success(f"Collected: {results}")
                
                # Get training texts from real financial data
                if hasattr(st.session_state.data_manager, 'prepare_training_texts_from_db'):
                    training_texts = st.session_state.data_manager.prepare_training_texts_from_db()
                    st.success(f"Using {len(training_texts)} real financial texts from database")
                    
            except Exception as e:
                st.error(f"Database error: {str(e)}")
                return
        
        # Only use fallback if no database connection
        if not training_texts:
            if st.session_state.training_data:
                training_texts = st.session_state.training_data
                st.warning("Using previously collected data")
            else:
                st.error("No real financial data available. Please initialize database connection.")
                return
        
        if not training_texts:
            st.error("No training texts available. Please collect data first.")
            return
        
        st.info(f"Training model with {len(training_texts)} texts...")
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train model with progress updates
        with st.spinner("Training financial diffusion model..."):
            progress_bar.progress(0.1)
            status_text.text("Starting training...")
            
            # Train the model
            losses = st.session_state.advanced_model.train(training_texts, epochs=num_epochs)
            
            progress_bar.progress(1.0)
            status_text.text(f"Training completed! Final loss: {losses[-1]:.6f}" if losses else "Training completed!")
        
        # Mark as trained
        st.session_state.advanced_model.is_trained = True
        st.session_state.model_trained = True
        
        # Save model checkpoint to database
        checkpoint_name = f"financial_diffusion_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save to database if available
        if st.session_state.database_connected and hasattr(st.session_state.data_manager, 'save_model_checkpoint'):
            try:
                model_info = st.session_state.advanced_model.get_model_info() if hasattr(st.session_state.advanced_model, 'get_model_info') else {"type": "SimpleFinancialDiffusion"}
                checkpoint_id = st.session_state.data_manager.save_model_checkpoint(
                    model_name="FinancialDiffusionLLM",
                    model_config=model_info,
                    training_params={"epochs": num_epochs, "texts": len(training_texts), "use_database": use_database},
                    performance_metrics={"final_loss": losses[-1] if losses else 0.0, "training_texts_count": len(training_texts)},
                    epoch=num_epochs,
                    loss=losses[-1] if losses else 0.0,
                    checkpoint_path=f"database_checkpoint_{checkpoint_id}",
                    is_best=True
                )
                st.success(f"Training completed! Model saved to PostgreSQL database (ID: {checkpoint_id})")
            except Exception as e:
                st.warning(f"Training completed but database save failed: {str(e)}")
                st.success("Training completed successfully!")
        else:
            st.success("Training completed successfully!")
            st.info("Model checkpoints will be saved to database when connection is available")
        
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
                        num_steps=num_inference_steps
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
                        max_length=max_length
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
        if not ADVANCED_MODULES_AVAILABLE or not FinancialDiffusionLLM:
            st.error("Advanced model not available for loading")
            return
            
        model = FinancialDiffusionLLM()
        model.load_model(filename)
        st.session_state.advanced_model = model
        st.session_state.model_trained = model.is_trained
        st.success(f"Model loaded from {filename}")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")

def database_analytics_page():
    st.header("📊 Database Analytics & Management")
    
    # Database connection status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if getattr(st.session_state, 'database_connected', False):
            st.success("PostgreSQL Connected")
        else:
            st.warning("Using In-Memory Storage")
    
    with col2:
        if st.button("🔄 Refresh Connection"):
            st.session_state.data_manager = None
            st.rerun()
    
    with col3:
        if st.button("🗄️ Initialize Database"):
            try:
                if DATABASE_AVAILABLE:
                    from database.init_db import setup_database
                    success = setup_database()
                    if success:
                        st.success("Database initialized successfully!")
                    else:
                        st.error("Database initialization failed")
                else:
                    st.error("Database modules not available")
            except Exception as e:
                st.error(f"Database initialization error: {str(e)}")
    
    # Database statistics
    if hasattr(st.session_state.data_manager, 'get_training_statistics'):
        try:
            stats = st.session_state.data_manager.get_training_statistics()
            
            st.subheader("📈 Database Statistics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Companies", stats.get('total_companies', 0))
            with col2:
                st.metric("News Articles", stats.get('total_news', 0))
            with col3:
                st.metric("Market Indicators", stats.get('total_market_indicators', 0))
            with col4:
                st.metric("Training Texts", stats.get('total_training_texts', 0))
            with col5:
                st.metric("Model Checkpoints", stats.get('total_model_checkpoints', 0))
                
        except Exception as e:
            st.error(f"Error loading database statistics: {str(e)}")
    
    # Recent data overview
    if hasattr(st.session_state.data_manager, 'get_recent_financial_data'):
        try:
            st.subheader("📊 Recent Financial Data")
            
            recent_data = st.session_state.data_manager.get_recent_financial_data(limit=10)
            
            # Recent companies
            if recent_data.get('companies'):
                st.write("**Recent Companies:**")
                companies_df = pd.DataFrame(recent_data['companies'])
                st.dataframe(companies_df, use_container_width=True)
            
            # Recent market indicators
            if recent_data.get('market_indicators'):
                st.write("**Market Indicators:**")
                indicators_df = pd.DataFrame(recent_data['market_indicators'])
                st.dataframe(indicators_df, use_container_width=True)
            
            # Recent news
            if recent_data.get('news'):
                st.write("**Recent News Headlines:**")
                for news in recent_data['news'][:5]:
                    st.write(f"• {news['title'][:100]}...")
                    
        except Exception as e:
            st.error(f"Error loading recent data: {str(e)}")
    
    # Database management actions
    st.subheader("🛠️ Database Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Collect Live Data"):
            if hasattr(st.session_state.data_manager, 'collect_all_live_data'):
                try:
                    with st.spinner("Collecting live financial data..."):
                        results = st.session_state.data_manager.collect_all_live_data()
                        st.success(f"Data collected: {results}")
                except Exception as e:
                    st.error(f"Data collection failed: {str(e)}")
            else:
                st.warning("Live data collection not available")
    
    with col2:
        if st.button("🧠 Prepare Training Data"):
            if hasattr(st.session_state.data_manager, 'prepare_training_texts_from_db'):
                try:
                    with st.spinner("Preparing training texts..."):
                        texts = st.session_state.data_manager.prepare_training_texts_from_db()
                        st.session_state.training_data = texts
                        st.success(f"Prepared {len(texts)} training texts")
                except Exception as e:
                    st.error(f"Training data preparation failed: {str(e)}")
            else:
                st.warning("Database training data preparation not available")
    
    with col3:
        if st.button("🗂️ Export Data"):
            st.info("Data export functionality coming soon")
    
    # Connection details
    if getattr(st.session_state, 'database_connected', False):
        st.subheader("🔗 Database Connection Info")
        db_url = os.getenv('DATABASE_URL', 'Not available')
        if db_url != 'Not available':
            try:
                import urllib.parse as urlparse
                url = urlparse.urlparse(db_url)
                st.write(f"**Host:** {url.hostname}")
                st.write(f"**Port:** {url.port}")
                st.write(f"**Database:** {url.path[1:] if url.path else 'Unknown'}")
                st.write(f"**User:** {url.username}")
            except:
                st.write("Connection details not available")

if __name__ == "__main__":
    main()