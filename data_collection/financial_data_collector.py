"""
Financial Data Collection Module

Collects real financial data from Yahoo Finance, SEC filings, and financial news APIs
to create a comprehensive training dataset for the diffusion model.
"""

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import feedparser
import time
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataCollector:
    """Collects financial data from various free APIs and sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def collect_company_info(self, symbols: List[str]) -> List[Dict]:
        """Collect company information and recent news from Yahoo Finance"""
        company_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get recent news
                news = ticker.news[:5] if ticker.news else []
                
                # Get financial statements
                financials = ticker.financials
                balance_sheet = ticker.balance_sheet
                cashflow = ticker.cashflow
                
                company_data.append({
                    'symbol': symbol,
                    'company_name': info.get('longName', symbol),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'business_summary': info.get('longBusinessSummary', ''),
                    'market_cap': info.get('marketCap', 0),
                    'revenue': info.get('totalRevenue', 0),
                    'news': news,
                    'financials': financials.to_dict() if not financials.empty else {},
                    'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                    'cashflow': cashflow.to_dict() if not cashflow.empty else {},
                    'collected_at': datetime.now().isoformat()
                })
                
                logger.info(f"Collected data for {symbol}")
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {str(e)}")
                continue
        
        return company_data
    
    def collect_earnings_transcripts(self, symbols: List[str]) -> List[Dict]:
        """Collect earnings call transcripts and summaries"""
        transcripts = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get earnings dates and calendar
                earnings_calendar = ticker.calendar
                if earnings_calendar is not None and not earnings_calendar.empty:
                    for date in earnings_calendar.index[:3]:  # Last 3 earnings
                        transcript_data = {
                            'symbol': symbol,
                            'earnings_date': date.isoformat(),
                            'transcript_summary': f"Earnings call for {symbol} on {date.strftime('%Y-%m-%d')}",
                            'collected_at': datetime.now().isoformat()
                        }
                        transcripts.append(transcript_data)
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error collecting earnings data for {symbol}: {str(e)}")
                continue
        
        return transcripts
    
    def collect_financial_news(self) -> List[Dict]:
        """Collect financial news from RSS feeds"""
        news_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.marketwatch.com/rss/topstories',
            'https://feeds.bloomberg.com/markets/news.rss'
        ]
        
        all_news = []
        
        for source_url in news_sources:
            try:
                feed = feedparser.parse(source_url)
                
                for entry in feed.entries[:10]:  # Get 10 latest articles
                    news_item = {
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': source_url,
                        'collected_at': datetime.now().isoformat()
                    }
                    all_news.append(news_item)
                
                logger.info(f"Collected {len(feed.entries[:10])} articles from {source_url}")
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting news from {source_url}: {str(e)}")
                continue
        
        return all_news
    
    def collect_sec_filings_summary(self, symbols: List[str]) -> List[Dict]:
        """Collect SEC filing summaries (10-K, 10-Q summaries)"""
        filings = []
        
        for symbol in symbols:
            try:
                # Use SEC EDGAR API (free but rate limited)
                cik_url = f"https://www.sec.gov/files/company_tickers.json"
                response = self.session.get(cik_url)
                
                if response.status_code == 200:
                    companies = response.json()
                    cik = None
                    
                    # Find CIK for symbol
                    for company in companies.values():
                        if company.get('ticker') == symbol:
                            cik = str(company.get('cik_str')).zfill(10)
                            break
                    
                    if cik:
                        # Get recent filings
                        filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
                        filings_response = self.session.get(filings_url)
                        
                        if filings_response.status_code == 200:
                            filing_data = filings_response.json()
                            recent_filings = filing_data.get('filings', {}).get('recent', {})
                            
                            for i in range(min(5, len(recent_filings.get('form', [])))):
                                form_type = recent_filings['form'][i]
                                if form_type in ['10-K', '10-Q', '8-K']:
                                    filing_summary = {
                                        'symbol': symbol,
                                        'cik': cik,
                                        'form_type': form_type,
                                        'filing_date': recent_filings['filingDate'][i],
                                        'accession_number': recent_filings['accessionNumber'][i],
                                        'description': recent_filings.get('primaryDocDescription', [''])[i],
                                        'collected_at': datetime.now().isoformat()
                                    }
                                    filings.append(filing_summary)
                
                time.sleep(0.2)  # SEC rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting SEC data for {symbol}: {str(e)}")
                continue
        
        return filings
    
    def collect_market_analysis(self) -> List[Dict]:
        """Collect market analysis and economic indicators"""
        analysis_data = []
        
        try:
            # Market indices
            indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']
            
            for index in indices:
                ticker = yf.Ticker(index)
                hist = ticker.history(period='5d')
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else latest
                    
                    change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
                    
                    analysis_data.append({
                        'indicator': index,
                        'current_value': float(latest['Close']),
                        'previous_value': float(prev['Close']),
                        'change_percent': float(change),
                        'volume': int(latest['Volume']) if 'Volume' in latest.index else 0,
                        'analysis_date': hist.index[-1].isoformat(),
                        'collected_at': datetime.now().isoformat()
                    })
            
            logger.info(f"Collected market data for {len(indices)} indices")
            
        except Exception as e:
            logger.error(f"Error collecting market analysis: {str(e)}")
        
        return analysis_data
    
    def extract_financial_text(self, data: Dict) -> List[str]:
        """Extract meaningful financial text from collected data"""
        texts = []
        
        # Extract from business summary
        if data.get('business_summary'):
            texts.append(data['business_summary'])
        
        # Extract from news
        for news_item in data.get('news', []):
            if news_item.get('title'):
                texts.append(news_item['title'])
            if news_item.get('summary'):
                texts.append(news_item['summary'])
        
        # Create financial performance summaries
        if data.get('market_cap') and data.get('revenue'):
            performance_text = f"The company has a market capitalization of ${data['market_cap']:,.0f} with total revenue of ${data['revenue']:,.0f}."
            texts.append(performance_text)
        
        return [text for text in texts if text and len(text.split()) > 5]

def get_sp500_symbols() -> List[str]:
    """Get S&P 500 symbols for data collection"""
    try:
        # Get S&P 500 list from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        table = soup.find('table', {'id': 'constituents'})
        symbols = []
        
        for row in table.find_all('tr')[1:51]:  # Get first 50 companies
            cells = row.find_all('td')
            if cells:
                symbol = cells[0].text.strip()
                symbols.append(symbol)
        
        return symbols
        
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {str(e)}")
        # Fallback to major companies
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JNJ', 'V']

def collect_comprehensive_dataset() -> Dict[str, List]:
    """Collect comprehensive financial dataset"""
    collector = FinancialDataCollector()
    
    # Get company symbols
    symbols = get_sp500_symbols()
    logger.info(f"Collecting data for {len(symbols)} companies")
    
    # Collect all data types
    dataset = {
        'companies': collector.collect_company_info(symbols[:10]),  # Start with 10 companies
        'earnings': collector.collect_earnings_transcripts(symbols[:10]),
        'news': collector.collect_financial_news(),
        'sec_filings': collector.collect_sec_filings_summary(symbols[:5]),
        'market_analysis': collector.collect_market_analysis()
    }
    
    logger.info("Data collection completed")
    return dataset

if __name__ == "__main__":
    dataset = collect_comprehensive_dataset()
    print(f"Collected {len(dataset['companies'])} company records")
    print(f"Collected {len(dataset['news'])} news articles")
    print(f"Collected {len(dataset['market_analysis'])} market indicators")