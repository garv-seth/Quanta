"""
Database schema for financial text diffusion model
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

Base = declarative_base()

class FinancialCompany(Base):
    __tablename__ = 'financial_companies'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False)
    company_name = Column(String(255))
    sector = Column(String(100))
    industry = Column(String(100))
    business_summary = Column(Text)
    market_cap = Column(Float)
    revenue = Column(Float)
    financials = Column(JSON)
    balance_sheet = Column(JSON)
    cashflow = Column(JSON)
    collected_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FinancialNews(Base):
    __tablename__ = 'financial_news'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    summary = Column(Text)
    content = Column(Text)
    link = Column(String(1000))
    published = Column(DateTime)
    source = Column(String(200))
    symbol = Column(String(10))
    collected_at = Column(DateTime, default=datetime.utcnow)

class EarningsTranscript(Base):
    __tablename__ = 'earnings_transcripts'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    earnings_date = Column(DateTime)
    transcript_text = Column(Text)
    transcript_summary = Column(Text)
    quarter = Column(String(10))
    year = Column(Integer)
    collected_at = Column(DateTime, default=datetime.utcnow)

class SECFiling(Base):
    __tablename__ = 'sec_filings'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    cik = Column(String(20))
    form_type = Column(String(10))
    filing_date = Column(DateTime)
    accession_number = Column(String(50))
    description = Column(Text)
    filing_url = Column(String(1000))
    extracted_text = Column(Text)
    collected_at = Column(DateTime, default=datetime.utcnow)

class MarketIndicator(Base):
    __tablename__ = 'market_indicators'
    
    id = Column(Integer, primary_key=True)
    indicator = Column(String(20), nullable=False)
    current_value = Column(Float)
    previous_value = Column(Float)
    change_percent = Column(Float)
    volume = Column(Float)
    analysis_date = Column(DateTime)
    collected_at = Column(DateTime, default=datetime.utcnow)

class TrainingText(Base):
    __tablename__ = 'training_texts'
    
    id = Column(Integer, primary_key=True)
    original_text = Column(Text, nullable=False)
    processed_text = Column(Text)
    source_type = Column(String(50))  # 'company_info', 'news', 'earnings', 'sec_filing'
    source_id = Column(Integer)
    symbol = Column(String(10))
    text_length = Column(Integer)
    quality_score = Column(Float)
    is_validated = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelCheckpoint(Base):
    __tablename__ = 'model_checkpoints'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    checkpoint_path = Column(String(500))
    model_config = Column(JSON)
    training_params = Column(JSON)
    performance_metrics = Column(JSON)
    epoch = Column(Integer)
    loss = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_best = Column(Boolean, default=False)

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close database session"""
        session.close()