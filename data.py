"""
Sample financial texts for training the Quasar diffusion model.
This module provides realistic financial text data for demonstration.
"""

def get_sample_financial_texts():
    """Get sample financial texts for training."""
    return [
        "The Company reported quarterly revenue of $2.3 billion, representing a 12% increase year-over-year.",
        "Operating expenses for the quarter totaled $450 million, compared to $420 million in the prior year period.",
        "Gross margin improved to 42.5% from 38.2% in the previous quarter due to operational efficiency gains.",
        "Cash and cash equivalents totaled $1.2 billion as of December 31, providing adequate liquidity.",
        "The effective tax rate for the period was 21.5%, compared to 23.1% in the prior year.",
        "Free cash flow generation of $180 million demonstrates strong operational performance.",
        "The Company's debt-to-equity ratio improved to 0.35 from 0.42 in the previous quarter.",
        "Return on invested capital increased to 14.2% from 12.8% year-over-year.",
    ]

def get_sample_text_pairs():
    """Get sample pairs of draft and refined financial texts."""
    return [
        {
            "draft": "Sales went up this quarter by quite a bit compared to last year.",
            "refined": "Revenue increased 15.3% year-over-year to $2.1 billion in the current quarter."
        },
        {
            "draft": "Our costs were higher than expected but still manageable.",
            "refined": "Operating expenses of $340 million exceeded guidance by 8% but remained within acceptable parameters."
        },
    ]