"""
Sample financial texts for training and testing the diffusion model.

This module provides curated financial text samples that can be used
for training, validation, and demonstration purposes.
"""

from typing import List, Dict


def get_sample_financial_texts() -> List[str]:
    """
    Get a collection of sample financial texts for training.
    
    Returns:
        List of financial text samples
    """
    
    # Sample financial report excerpts and statements
    financial_texts = [
        # Quarterly earnings reports
        "The company reported strong quarterly results with revenue increasing by fifteen percent year over year. Net income rose to forty-two million dollars compared to thirty-one million in the previous quarter. Operating margins improved due to cost reduction initiatives and increased efficiency.",
        
        "Despite challenging market conditions, our financial performance remained stable. Revenue was flat at two hundred fifty million dollars while maintaining healthy profit margins. The management team implemented strategic cost controls to navigate economic uncertainty.",
        
        "Third quarter results exceeded expectations with record revenue of one hundred eighty million dollars. The company benefited from strong demand in core markets and successful product launches. Earnings per share increased to two dollars and fifteen cents.",
        
        # Investment analysis
        "The investment portfolio generated positive returns this quarter driven by strong performance in technology and healthcare sectors. Asset allocation was rebalanced to reduce exposure to volatile markets while maintaining growth potential.",
        
        "Market volatility created both challenges and opportunities for our investment strategy. We maintained a conservative approach focusing on high-quality assets with stable dividend yields. The portfolio outperformed benchmark indices by two percentage points.",
        
        # Company performance updates
        "Operational efficiency improvements contributed to margin expansion across all business segments. The company successfully reduced overhead costs while investing in growth initiatives. Cash flow from operations increased by twenty-five percent.",
        
        "The merger and acquisition activity strengthened our market position and expanded geographic reach. Integration costs were within expected ranges and synergies are beginning to materialize. Revenue diversification reduced dependency on single market segments.",
        
        # Financial guidance and forecasts
        "Management raised full-year guidance based on strong first-half performance and positive market trends. Revenue is expected to grow between eight and twelve percent with improved profitability. Capital expenditure plans remain on track.",
        
        "The economic outlook presents mixed signals with both opportunities and risks ahead. We maintain cautious optimism while preparing for various scenarios. Our strong balance sheet provides flexibility during uncertain times.",
        
        # Risk assessments
        "Credit risk analysis indicates stable trends across the loan portfolio with default rates remaining below historical averages. Provisions for credit losses were adjusted based on updated economic forecasts and portfolio composition.",
        
        "Interest rate sensitivity analysis shows manageable exposure to rate fluctuations. The company maintains hedging strategies to mitigate potential impacts on net interest income. Asset-liability management remains a key focus area.",
        
        # Market analysis
        "Consumer spending patterns shifted during the quarter reflecting changing preferences and economic conditions. Retail sales data showed resilience in essential categories while discretionary spending remained subdued.",
        
        "Industry consolidation accelerated as companies seek scale advantages and operational synergies. Market leaders are gaining share through strategic investments and competitive positioning. Regulatory changes may impact future industry dynamics.",
        
        # Annual report excerpts
        "Fiscal year results demonstrated the resilience of our business model and strategic positioning. Total revenue reached one point two billion dollars with net income of one hundred fifty million. Return on equity improved to fourteen percent.",
        
        "Our commitment to sustainable growth and responsible business practices drove value creation for all stakeholders. Environmental, social, and governance initiatives gained momentum while delivering measurable results.",
        
        # Cash flow statements
        "Operating cash flow remained strong at seventy-five million dollars supporting business operations and growth investments. Free cash flow generation enabled debt reduction and shareholder returns through dividends and share repurchases.",
        
        "Working capital management improved through enhanced inventory turnover and collections processes. Days sales outstanding decreased while maintaining strong customer relationships. Cash conversion cycle optimization continues.",
        
        # Balance sheet analysis
        "The balance sheet strengthened with improved debt-to-equity ratios and increased cash reserves. Liquidity position provides adequate resources for operational needs and strategic opportunities. Credit ratings were affirmed by major agencies.",
        
        "Asset quality metrics showed improvement across key indicators with non-performing assets declining. Loan loss provisions were adequate based on historical experience and forward-looking assessments.",
        
        # Strategic initiatives
        "Digital transformation initiatives accelerated operational efficiency and customer experience improvements. Technology investments generated measurable returns through automation and data analytics capabilities.",
        
        "Geographic expansion into emerging markets created new growth opportunities while diversifying revenue streams. Local partnerships and market knowledge facilitated successful market entry strategies.",
        
        # Regulatory updates
        "New regulatory requirements were implemented successfully with minimal operational disruption. Compliance costs were managed through process improvements and technology solutions. Regulatory capital ratios exceeded minimum requirements.",
        
        "Tax reform impacts were incorporated into financial planning and reporting processes. Effective tax rate optimization strategies were implemented while maintaining full compliance with regulations.",
        
        # Investor relations
        "Investor feedback emphasized the importance of transparent communication and consistent execution of strategic plans. Management remains committed to regular updates and engagement with the investment community.",
        
        "Share price performance reflected market recognition of operational improvements and strategic progress. Trading volumes remained stable with institutional ownership increasing during the quarter.",
        
        # Economic commentary
        "Macroeconomic indicators suggest continued moderate growth with controlled inflation expectations. Central bank policies remain accommodative supporting business investment and consumer spending.",
        
        "Global trade dynamics and geopolitical developments create uncertainties requiring active risk management and scenario planning. Supply chain resilience remains a strategic priority."
    ]
    
    return financial_texts


def get_sample_draft_texts() -> List[str]:
    """
    Get sample draft financial texts that need refinement.
    
    Returns:
        List of draft financial text samples (lower quality)
    """
    
    draft_texts = [
        "Q3 results good. Revenue up. Profit ok too. Market doing fine.",
        
        "Company made money this quarter. Sales were higher than before. Costs went down some.",
        
        "Business is doing well. Numbers look good. Management happy with performance.",
        
        "Stock price went up. Investors like the company. Future looks bright maybe.",
        
        "Earnings beat expectations. Revenue growth strong. Margins improved slightly.",
        
        "Cash flow positive this quarter. Balance sheet looks healthy. Debt levels manageable.",
        
        "Market conditions challenging but company adapting. Strategy working so far.",
        
        "Investment returns decent. Portfolio performance stable. Risk management ongoing.",
        
        "New products launching soon. Market response expected positive. Sales team optimistic.",
        
        "Regulatory changes coming. Company preparing for compliance. Costs may increase.",
        
        "Merger completed successfully. Integration proceeding as planned. Synergies materializing.",
        
        "Economic outlook uncertain. Company remains cautious but hopeful. Plans flexible.",
        
        "Technology upgrades implemented. Efficiency gains realized. Customer satisfaction improved.",
        
        "International expansion continues. New markets showing promise. Local partnerships formed.",
        
        "Credit quality stable. Default rates low. Provision levels adequate for now."
    ]
    
    return draft_texts


def get_financial_terminology() -> Dict[str, List[str]]:
    """
    Get financial terminology categorized by type.
    
    Returns:
        Dictionary with categorized financial terms
    """
    
    terminology = {
        'performance_metrics': [
            'revenue', 'profit', 'earnings', 'income', 'margin', 'roi', 'roe', 'roa',
            'ebitda', 'eps', 'pe_ratio', 'cash_flow', 'operating_income', 'net_income'
        ],
        
        'financial_statements': [
            'balance_sheet', 'income_statement', 'cash_flow_statement', 'equity_statement',
            'assets', 'liabilities', 'equity', 'working_capital', 'retained_earnings'
        ],
        
        'market_terms': [
            'market_cap', 'share_price', 'dividend', 'yield', 'volatility', 'beta',
            'trading_volume', 'market_share', 'valuation', 'price_earnings'
        ],
        
        'business_operations': [
            'operations', 'strategy', 'growth', 'expansion', 'acquisition', 'merger',
            'efficiency', 'productivity', 'innovation', 'competitive_advantage'
        ],
        
        'risk_management': [
            'risk_assessment', 'credit_risk', 'market_risk', 'operational_risk',
            'compliance', 'regulation', 'hedging', 'diversification', 'insurance'
        ],
        
        'time_periods': [
            'quarterly', 'annual', 'year_over_year', 'quarter_over_quarter',
            'fiscal_year', 'calendar_year', 'monthly', 'semi_annual'
        ]
    }
    
    return terminology


def get_sample_text_pairs() -> List[Dict[str, str]]:
    """
    Get sample pairs of draft and refined financial texts.
    
    Returns:
        List of dictionaries with 'draft' and 'refined' keys
    """
    
    text_pairs = [
        {
            'draft': "Company did good this quarter. Money up.",
            'refined': "The company delivered strong quarterly performance with revenue growth exceeding expectations and improved profitability metrics."
        },
        
        {
            'draft': "Sales were ok. Costs down a bit. Profit better.",
            'refined': "Revenue remained stable while operational efficiency improvements resulted in reduced costs and enhanced profit margins."
        },
        
        {
            'draft': "Stock went up. People buying more. Good news.",
            'refined': "Share price appreciation reflected increased investor confidence and positive market sentiment following strong fundamental performance."
        },
        
        {
            'draft': "Market tough but we doing fine. Plan working.",
            'refined': "Despite challenging market conditions, the company maintained resilient performance through effective strategic execution and adaptive operational management."
        },
        
        {
            'draft': "New product selling well. Customers like it. More sales coming.",
            'refined': "The recently launched product line achieved strong market acceptance with positive customer feedback driving increased sales momentum and revenue growth."
        }
    ]
    
    return text_pairs


def get_financial_contexts() -> List[str]:
    """
    Get different financial contexts for text generation.
    
    Returns:
        List of financial context descriptions
    """
    
    contexts = [
        "quarterly_earnings_report",
        "annual_shareholder_letter",
        "investor_presentation",
        "financial_analysis_report",
        "market_commentary",
        "risk_assessment_document",
        "strategic_planning_memo",
        "budget_review_summary",
        "acquisition_announcement",
        "dividend_policy_statement",
        "credit_rating_analysis",
        "economic_outlook_report",
        "sector_analysis_brief",
        "performance_dashboard_summary",
        "regulatory_compliance_update"
    ]
    
    return contexts


# Utility functions for data management
def filter_texts_by_length(texts: List[str], min_words: int = 10, max_words: int = 200) -> List[str]:
    """
    Filter texts by word count.
    
    Args:
        texts: List of texts to filter
        min_words: Minimum word count
        max_words: Maximum word count
        
    Returns:
        Filtered list of texts
    """
    filtered_texts = []
    
    for text in texts:
        word_count = len(text.split())
        if min_words <= word_count <= max_words:
            filtered_texts.append(text)
    
    return filtered_texts


def get_training_data(num_samples: int = None) -> List[str]:
    """
    Get training data with optional sample size limit.
    
    Args:
        num_samples: Maximum number of samples to return
        
    Returns:
        List of training texts
    """
    all_texts = get_sample_financial_texts()
    
    if num_samples is not None:
        return all_texts[:num_samples]
    
    return all_texts


def get_test_data() -> Dict[str, List[str]]:
    """
    Get test data for evaluation.
    
    Returns:
        Dictionary with draft and refined text lists
    """
    pairs = get_sample_text_pairs()
    
    return {
        'draft_texts': [pair['draft'] for pair in pairs],
        'refined_texts': [pair['refined'] for pair in pairs]
    }
