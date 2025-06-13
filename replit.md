# Quanta Quasar - Financial Diffusion Language Model

## Overview

Quanta Quasar is a production-ready Diffusion-based Large Language Model (dLLM) specifically designed for financial text processing. The system automatically detects available hardware (62GB RAM Replit environment or local RTX 4060) and optimizes training parameters accordingly. Features comprehensive financial data collection, proper diffusion model architecture with cosine noise scheduling, and extended training duration (45+ minutes) for real model convergence.

## System Architecture

### Frontend Architecture
- **Framework**: Single Streamlit application (`quasar_financial_diffusion.py`)
- **Interface**: Adaptive UI that shows detected hardware specifications
- **Tabs**: Training, Monitoring, Management for complete workflow
- **Configuration**: Automatic hardware detection and optimization
- **Deployment**: Streamlit server on port 5000

### Backend Architecture
- **Core Model**: `FinancialDiffusionModel` - Production transformer-based diffusion model
- **Environment Detection**: `EnvironmentDetector` - Automatic hardware optimization
- **Data Collection**: `FinancialDataCollector` - Real financial text generation
- **Tokenization**: `FinancialTokenizer` - Domain-specific vocabulary building
- **Training**: `QuasarTrainer` - Production training pipeline with proper duration

### Text Processing Pipeline
- **Embedding Generation**: Sentence Transformers (all-MiniLM-L6-v2) for 384-dimensional embeddings
- **Preprocessing**: Financial text normalization, abbreviation expansion, domain-specific term handling
- **Diffusion Process**: Forward noise addition and reverse denoising in embedding space
- **Post-processing**: Embedding-to-text conversion with quality enhancement

## Key Components

### 1. Diffusion Engine
- **Noise Schedule**: Cosine schedule for smooth transitions with financial volatility-aware patterns
- **Denoising Network**: Multi-layer transformer with time conditioning
- **Embedding Space**: 384-dimensional semantic vector space for financial text
- **Process Flow**: Input Text → Embedding Space → Add Noise → Denoising Network → Refined Output

### 2. Model Suite
- **Quasar Advanced**: Full production model with 15.2M parameters, 8-head attention, 6 layers
- **Quasar Basic**: Lightweight model for fast inference
- **FinSar**: Breakthrough model implementing quantum-inspired path integral formulation
- **Simple Models**: Demonstration models with reduced dependencies

### 3. Training Infrastructure
- **ModelTrainer**: Handles training loops, optimization strategies, and monitoring
- **FinancialTextDataset**: Custom dataset class for financial text embeddings
- **Training History**: Comprehensive logging of training metrics and model performance

### 4. Evaluation System
- **ModelEvaluator**: Comprehensive metrics including BLEU, ROUGE, semantic similarity
- **Financial Domain Metrics**: Specialized evaluation for financial text quality
- **Performance Tracking**: Real-time monitoring of model performance and accuracy

## Data Flow

1. **Input Processing**: Raw financial text → TextProcessor → Embeddings
2. **Diffusion Forward**: Clean embeddings → Noise addition → Noisy embeddings
3. **Diffusion Reverse**: Noisy embeddings → Denoising network → Refined embeddings
4. **Output Generation**: Refined embeddings → Text decoder → Enhanced financial text
5. **Quality Assessment**: Generated text → ModelEvaluator → Quality metrics

## External Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework for model implementation
- **Streamlit**: Web application framework for user interface
- **Sentence Transformers**: Pre-trained embeddings for text processing
- **NumPy/Pandas**: Data manipulation and numerical computing
- **Plotly**: Interactive visualization for model metrics

### Financial Data Sources
- **Yahoo Finance (yfinance)**: Real-time financial data and company information
- **SEC EDGAR**: Public financial reports and filings
- **Financial News APIs**: RSS feeds and financial news sources
- **Web Scraping**: BeautifulSoup and requests for data collection

### Database Integration
- **PostgreSQL**: Primary database for storing model data and financial information
- **SQLAlchemy**: ORM for database operations
- **Replit Database**: Alternative storage for model weights and configurations

### Optional Dependencies
- **NLTK**: Natural language processing utilities
- **ROUGE Score**: Text evaluation metrics
- **Scikit-learn**: Machine learning utilities for evaluation

## Deployment Strategy

### Development Environment
- **Platform**: Replit with Python 3.11 runtime
- **Package Manager**: UV for dependency management
- **Development Server**: Streamlit development server with hot reload

### Production Deployment
- **Target**: Autoscale deployment on Replit
- **Port Configuration**: Server runs on port 5000
- **Workflow Management**: Multiple workflow configurations for different models
- **Health Monitoring**: Built-in health checks and performance monitoring

### Model Management
- **Checkpointing**: Automatic model saving with versioning
- **Weight Storage**: Database storage for model weights and configurations
- **Model Loading**: Dynamic model loading based on user selection
- **Performance Caching**: Session state management for model persistence

## Changelog

```
Changelog:
- June 13, 2025. Initial setup
- June 13, 2025. Implemented production training system optimized for RTX 4060
- June 13, 2025. Created FinSar quantum-inspired diffusion model with path exploration
- June 13, 2025. Added real financial data collection from Yahoo Finance and SEC sources
- June 13, 2025. Built hardware-optimized training pipeline with automatic mixed precision
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```