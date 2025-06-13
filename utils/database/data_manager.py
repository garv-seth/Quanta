"""
Database Data Manager for Financial Text Diffusion Model

Handles storing and retrieving financial data from PostgreSQL database
"""

from sqlalchemy.orm import Session
from database.schema import (
    DatabaseManager, FinancialCompany, FinancialNews, EarningsTranscript,
    SECFiling, MarketIndicator, TrainingText, ModelCheckpoint
)
from data_collection.financial_data_collector import FinancialDataCollector, collect_comprehensive_dataset
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FinancialDataManager:
    """Manages financial data storage and retrieval"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.db_manager.create_tables()
        self.data_collector = FinancialDataCollector()
    
    def collect_and_store_data(self) -> Dict[str, int]:
        """Collect fresh financial data and store in database"""
        logger.info("Starting data collection and storage process")
        
        # Collect comprehensive dataset
        dataset = collect_comprehensive_dataset()
        
        session = self.db_manager.get_session()
        counts = {'companies': 0, 'news': 0, 'earnings': 0, 'sec_filings': 0, 'market_indicators': 0}
        
        try:
            # Store company data
            for company_data in dataset['companies']:
                existing = session.query(FinancialCompany).filter_by(symbol=company_data['symbol']).first()
                
                if existing:
                    # Update existing record
                    for key, value in company_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new record
                    company = FinancialCompany(**company_data)
                    session.add(company)
                    counts['companies'] += 1
            
            # Store news data
            for news_item in dataset['news']:
                # Check if news already exists
                existing = session.query(FinancialNews).filter_by(link=news_item['link']).first()
                
                if not existing:
                    news = FinancialNews(**news_item)
                    session.add(news)
                    counts['news'] += 1
            
            # Store earnings data
            for earnings_data in dataset['earnings']:
                earnings = EarningsTranscript(**earnings_data)
                session.add(earnings)
                counts['earnings'] += 1
            
            # Store SEC filings
            for filing_data in dataset['sec_filings']:
                existing = session.query(SECFiling).filter_by(
                    accession_number=filing_data['accession_number']
                ).first()
                
                if not existing:
                    filing = SECFiling(**filing_data)
                    session.add(filing)
                    counts['sec_filings'] += 1
            
            # Store market indicators
            for indicator_data in dataset['market_analysis']:
                indicator = MarketIndicator(**indicator_data)
                session.add(indicator)
                counts['market_indicators'] += 1
            
            session.commit()
            logger.info(f"Data stored successfully: {counts}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing data: {str(e)}")
            raise
        finally:
            self.db_manager.close_session(session)
        
        return counts
    
    def prepare_training_texts(self) -> List[str]:
        """Prepare training texts from stored financial data"""
        session = self.db_manager.get_session()
        training_texts = []
        
        try:
            # Get company business summaries
            companies = session.query(FinancialCompany).all()
            for company in companies:
                if company.business_summary and len(company.business_summary.split()) > 10:
                    training_texts.append(company.business_summary)
                    
                    # Store as training text
                    training_text = TrainingText(
                        original_text=company.business_summary,
                        source_type='company_info',
                        source_id=company.id,
                        symbol=company.symbol,
                        text_length=len(company.business_summary.split())
                    )
                    session.add(training_text)
            
            # Get news articles
            news_items = session.query(FinancialNews).all()
            for news in news_items:
                if news.title and len(news.title.split()) > 5:
                    training_texts.append(news.title)
                if news.summary and len(news.summary.split()) > 10:
                    training_texts.append(news.summary)
                    
                    # Store as training text
                    training_text = TrainingText(
                        original_text=news.summary,
                        source_type='news',
                        source_id=news.id,
                        symbol=news.symbol,
                        text_length=len(news.summary.split())
                    )
                    session.add(training_text)
            
            # Get earnings transcripts
            earnings = session.query(EarningsTranscript).all()
            for earning in earnings:
                if earning.transcript_summary and len(earning.transcript_summary.split()) > 10:
                    training_texts.append(earning.transcript_summary)
                    
                    # Store as training text
                    training_text = TrainingText(
                        original_text=earning.transcript_summary,
                        source_type='earnings',
                        source_id=earning.id,
                        symbol=earning.symbol,
                        text_length=len(earning.transcript_summary.split())
                    )
                    session.add(training_text)
            
            session.commit()
            logger.info(f"Prepared {len(training_texts)} training texts")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error preparing training texts: {str(e)}")
            raise
        finally:
            self.db_manager.close_session(session)
        
        return training_texts
    
    def save_model_checkpoint(self, model_name: str, model_config: Dict, 
                            training_params: Dict, performance_metrics: Dict,
                            epoch: int, loss: float, checkpoint_path: str,
                            is_best: bool = False) -> int:
        """Save model checkpoint to database"""
        session = self.db_manager.get_session()
        
        try:
            checkpoint = ModelCheckpoint(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                model_config=model_config,
                training_params=training_params,
                performance_metrics=performance_metrics,
                epoch=epoch,
                loss=loss,
                is_best=is_best
            )
            
            session.add(checkpoint)
            session.commit()
            
            checkpoint_id = checkpoint.id
            logger.info(f"Model checkpoint saved with ID: {checkpoint_id}")
            
            return checkpoint_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving model checkpoint: {str(e)}")
            raise
        finally:
            self.db_manager.close_session(session)
    
    def get_best_model_checkpoint(self, model_name: str) -> Optional[ModelCheckpoint]:
        """Get the best model checkpoint for a given model name"""
        session = self.db_manager.get_session()
        
        try:
            checkpoint = session.query(ModelCheckpoint).filter_by(
                model_name=model_name,
                is_best=True
            ).first()
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error retrieving best model checkpoint: {str(e)}")
            return None
        finally:
            self.db_manager.close_session(session)
    
    def get_training_statistics(self) -> Dict:
        """Get training data statistics"""
        session = self.db_manager.get_session()
        
        try:
            stats = {
                'total_companies': session.query(FinancialCompany).count(),
                'total_news': session.query(FinancialNews).count(),
                'total_earnings': session.query(EarningsTranscript).count(),
                'total_sec_filings': session.query(SECFiling).count(),
                'total_market_indicators': session.query(MarketIndicator).count(),
                'total_training_texts': session.query(TrainingText).count(),
                'total_model_checkpoints': session.query(ModelCheckpoint).count()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting training statistics: {str(e)}")
            return {}
        finally:
            self.db_manager.close_session(session)
    
    def get_recent_financial_data(self, limit: int = 100) -> Dict[str, List]:
        """Get recent financial data for display"""
        session = self.db_manager.get_session()
        
        try:
            # Get recent companies
            recent_companies = session.query(FinancialCompany).order_by(
                FinancialCompany.collected_at.desc()
            ).limit(limit).all()
            
            # Get recent news
            recent_news = session.query(FinancialNews).order_by(
                FinancialNews.collected_at.desc()
            ).limit(limit).all()
            
            # Get recent market indicators
            recent_indicators = session.query(MarketIndicator).order_by(
                MarketIndicator.collected_at.desc()
            ).limit(limit).all()
            
            return {
                'companies': [
                    {
                        'symbol': company.symbol,
                        'name': company.company_name,
                        'sector': company.sector,
                        'market_cap': company.market_cap,
                        'revenue': company.revenue,
                        'collected_at': company.collected_at.isoformat() if company.collected_at else None
                    }
                    for company in recent_companies
                ],
                'news': [
                    {
                        'title': news.title,
                        'summary': news.summary[:200] + '...' if news.summary and len(news.summary) > 200 else news.summary,
                        'source': news.source,
                        'published': news.published,
                        'collected_at': news.collected_at.isoformat() if news.collected_at else None
                    }
                    for news in recent_news
                ],
                'market_indicators': [
                    {
                        'indicator': indicator.indicator,
                        'current_value': indicator.current_value,
                        'change_percent': indicator.change_percent,
                        'analysis_date': indicator.analysis_date.isoformat() if indicator.analysis_date else None
                    }
                    for indicator in recent_indicators
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting recent financial data: {str(e)}")
            return {'companies': [], 'news': [], 'market_indicators': []}
        finally:
            self.db_manager.close_session(session)