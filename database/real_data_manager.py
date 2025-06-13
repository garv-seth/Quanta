"""
Real Financial Data Manager with PostgreSQL Integration

Handles storing and retrieving live financial data from Yahoo Finance
and other sources into PostgreSQL database.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from database.schema import (
    FinancialCompany, FinancialNews, MarketIndicator, 
    TrainingText, ModelCheckpoint, Base
)
import yfinance as yf
import feedparser
import pandas as pd

class RealFinancialDataManager:
    """Manages real financial data collection and storage"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Ensure tables exist
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def collect_live_company_data(self, symbols: List[str] = None) -> Dict[str, int]:
        """Collect live company data from Yahoo Finance"""
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
        
        session = self.get_session()
        companies_added = 0
        
        try:
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Check if company already exists
                    existing = session.query(FinancialCompany).filter_by(symbol=symbol).first()
                    
                    if existing:
                        # Update existing record
                        existing.market_cap = info.get('marketCap', 0)
                        existing.revenue = info.get('totalRevenue', 0)
                        existing.updated_at = datetime.utcnow()
                        existing.financials = json.dumps({
                            'pe_ratio': info.get('trailingPE'),
                            'price_to_book': info.get('priceToBook'),
                            'dividend_yield': info.get('dividendYield'),
                            'current_price': info.get('currentPrice')
                        })
                    else:
                        # Create new record
                        company = FinancialCompany(
                            symbol=symbol,
                            company_name=info.get('longName', symbol),
                            sector=info.get('sector', 'Unknown'),
                            industry=info.get('industry', 'Unknown'),
                            business_summary=info.get('longBusinessSummary', ''),
                            market_cap=info.get('marketCap', 0),
                            revenue=info.get('totalRevenue', 0),
                            financials=json.dumps({
                                'pe_ratio': info.get('trailingPE'),
                                'price_to_book': info.get('priceToBook'),
                                'dividend_yield': info.get('dividendYield'),
                                'current_price': info.get('currentPrice')
                            })
                        )
                        session.add(company)
                        companies_added += 1
                
                except Exception as e:
                    print(f"Error collecting data for {symbol}: {e}")
                    continue
            
            session.commit()
            return {'companies': companies_added}
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def collect_live_market_data(self) -> Dict[str, int]:
        """Collect live market indicator data"""
        indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']
        session = self.get_session()
        indicators_added = 0
        
        try:
            for index in indices:
                try:
                    ticker = yf.Ticker(index)
                    hist = ticker.history(period='2d')
                    
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        prev = hist.iloc[-2] if len(hist) > 1 else latest
                        
                        change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
                        
                        # Delete old records for this indicator (keep only latest)
                        session.query(MarketIndicator).filter_by(indicator=index).delete()
                        
                        indicator = MarketIndicator(
                            indicator=index,
                            current_value=float(latest['Close']),
                            previous_value=float(prev['Close']),
                            change_percent=float(change),
                            volume=int(latest['Volume']) if 'Volume' in latest.index else 0,
                            analysis_date=hist.index[-1]
                        )
                        session.add(indicator)
                        indicators_added += 1
                
                except Exception as e:
                    print(f"Error collecting market data for {index}: {e}")
                    continue
            
            session.commit()
            return {'market_indicators': indicators_added}
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def collect_live_news_data(self) -> Dict[str, int]:
        """Collect live financial news data"""
        session = self.get_session()
        news_added = 0
        
        try:
            # Yahoo Finance RSS feed
            feed = feedparser.parse('https://feeds.finance.yahoo.com/rss/2.0/headline')
            
            for entry in feed.entries[:20]:  # Limit to 20 recent articles
                try:
                    # Check if news already exists
                    existing = session.query(FinancialNews).filter_by(
                        title=entry.get('title', ''),
                        link=entry.get('link', '')
                    ).first()
                    
                    if not existing:
                        news = FinancialNews(
                            title=entry.get('title', ''),
                            summary=entry.get('summary', ''),
                            content=entry.get('content', ''),
                            link=entry.get('link', ''),
                            published=datetime.now(),  # RSS doesn't always have good dates
                            source='Yahoo Finance'
                        )
                        session.add(news)
                        news_added += 1
                
                except Exception as e:
                    print(f"Error processing news entry: {e}")
                    continue
            
            session.commit()
            return {'news': news_added}
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def collect_all_live_data(self) -> Dict[str, int]:
        """Collect all types of live financial data"""
        results = {}
        
        # Collect company data
        company_results = self.collect_live_company_data()
        results.update(company_results)
        
        # Collect market data
        market_results = self.collect_live_market_data()
        results.update(market_results)
        
        # Collect news data
        news_results = self.collect_live_news_data()
        results.update(news_results)
        
        return results
    
    def prepare_training_texts_from_db(self) -> List[str]:
        """Extract training texts from stored financial data"""
        session = self.get_session()
        training_texts = []
        
        try:
            # Get company business summaries
            companies = session.query(FinancialCompany).all()
            for company in companies:
                if company.business_summary and len(company.business_summary.split()) > 10:
                    training_texts.append(company.business_summary)
                
                # Create performance text
                if company.market_cap and company.revenue:
                    performance_text = (
                        f"{company.company_name} operates in the {company.sector} sector "
                        f"with a market capitalization of ${company.market_cap:,.0f} "
                        f"and annual revenue of ${company.revenue:,.0f}."
                    )
                    training_texts.append(performance_text)
            
            # Get news content
            news_items = session.query(FinancialNews).order_by(FinancialNews.collected_at.desc()).limit(50).all()
            for news in news_items:
                if news.title and len(news.title.split()) > 5:
                    training_texts.append(news.title)
                if news.summary and len(news.summary.split()) > 10:
                    training_texts.append(news.summary)
            
            # Get market analysis texts
            indicators = session.query(MarketIndicator).all()
            for indicator in indicators:
                change_pct = indicator.change_percent
                indicator_name = indicator.indicator
                current_value = indicator.current_value
                
                if change_pct > 0:
                    market_text = (
                        f"The {indicator_name} index gained {change_pct:.2f}% to close at "
                        f"{current_value:.2f}, reflecting positive market sentiment and investor confidence."
                    )
                elif change_pct < 0:
                    market_text = (
                        f"The {indicator_name} index declined {abs(change_pct):.2f}% to {current_value:.2f}, "
                        f"indicating market volatility and cautious investor behavior."
                    )
                else:
                    market_text = (
                        f"The {indicator_name} index remained stable at {current_value:.2f}, "
                        f"showing balanced market conditions."
                    )
                
                training_texts.append(market_text)
            
            # Store processed training texts in database
            for text in training_texts:
                if len(text.split()) > 5:  # Only store meaningful texts
                    training_text = TrainingText(
                        original_text=text,
                        processed_text=text,
                        source_type='live_data',
                        text_length=len(text.split()),
                        quality_score=min(1.0, len(text.split()) / 50.0)
                    )
                    session.add(training_text)
            
            session.commit()
            return [text for text in training_texts if text and len(text.split()) > 5]
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def save_model_checkpoint(self, model_name: str, model_config: Dict, 
                            training_params: Dict, performance_metrics: Dict,
                            epoch: int, loss: float, checkpoint_path: str,
                            is_best: bool = False) -> int:
        """Save model checkpoint to database"""
        session = self.get_session()
        
        try:
            checkpoint = ModelCheckpoint(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                model_config=json.dumps(model_config),
                training_params=json.dumps(training_params),
                performance_metrics=json.dumps(performance_metrics),
                epoch=epoch,
                loss=loss,
                is_best=is_best
            )
            
            session.add(checkpoint)
            session.commit()
            
            return checkpoint.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_recent_financial_data(self, limit: int = 10) -> Dict[str, List]:
        """Get recent financial data for display"""
        session = self.get_session()
        
        try:
            # Recent companies
            companies = session.query(FinancialCompany).order_by(
                FinancialCompany.updated_at.desc()
            ).limit(limit).all()
            
            # Recent news
            news = session.query(FinancialNews).order_by(
                FinancialNews.collected_at.desc()
            ).limit(limit).all()
            
            # Recent market indicators
            indicators = session.query(MarketIndicator).order_by(
                MarketIndicator.collected_at.desc()
            ).limit(limit).all()
            
            return {
                'companies': [
                    {
                        'symbol': c.symbol,
                        'company_name': c.company_name,
                        'sector': c.sector,
                        'market_cap': c.market_cap,
                        'revenue': c.revenue,
                        'updated_at': c.updated_at.isoformat() if c.updated_at else None
                    } for c in companies
                ],
                'news': [
                    {
                        'title': n.title,
                        'summary': n.summary,
                        'source': n.source,
                        'link': n.link,
                        'published': n.published.isoformat() if n.published else None
                    } for n in news
                ],
                'market_indicators': [
                    {
                        'indicator': i.indicator,
                        'current_value': i.current_value,
                        'change_percent': i.change_percent,
                        'volume': i.volume,
                        'analysis_date': i.analysis_date.isoformat() if i.analysis_date else None
                    } for i in indicators
                ]
            }
            
        except Exception as e:
            raise e
        finally:
            session.close()
    
    def get_training_statistics(self) -> Dict:
        """Get training data statistics"""
        session = self.get_session()
        
        try:
            companies_count = session.query(FinancialCompany).count()
            news_count = session.query(FinancialNews).count()
            indicators_count = session.query(MarketIndicator).count()
            training_texts_count = session.query(TrainingText).count()
            model_checkpoints_count = session.query(ModelCheckpoint).count()
            
            return {
                'total_companies': companies_count,
                'total_news': news_count,
                'total_market_indicators': indicators_count,
                'total_training_texts': training_texts_count,
                'total_model_checkpoints': model_checkpoints_count
            }
            
        except Exception as e:
            raise e
        finally:
            session.close()