"""
Simple Text Processor for Financial Text Diffusion Model

A lightweight text processor that works without external ML dependencies
"""

import re
import numpy as np
from typing import List, Dict, Union
import string

class SimpleTextProcessor:
    """Simple text processor for financial document processing"""
    
    def __init__(self):
        # Financial terms and patterns
        self.financial_terms = {
            'revenue', 'profit', 'loss', 'earnings', 'ebitda', 'cash flow',
            'assets', 'liabilities', 'equity', 'roi', 'margin', 'growth',
            'quarter', 'fiscal', 'financial', 'investment', 'return',
            'market', 'share', 'dividend', 'expense', 'cost', 'budget'
        }
        
        # Common financial abbreviations
        self.financial_abbrevs = {
            'Q1': 'first quarter',
            'Q2': 'second quarter', 
            'Q3': 'third quarter',
            'Q4': 'fourth quarter',
            'YoY': 'year over year',
            'QoQ': 'quarter over quarter',
            'EBITDA': 'earnings before interest, taxes, depreciation, and amortization',
            'ROI': 'return on investment',
            'ROE': 'return on equity',
            'P&L': 'profit and loss',
            'CAPEX': 'capital expenditure',
            'OPEX': 'operating expenditure'
        }
        
        # Simple vocabulary for embeddings
        self.vocab = {}
        self.inverse_vocab = {}
        self.embedding_dim = 384
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess financial text for better quality"""
        if not text or not text.strip():
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Expand financial abbreviations
        for abbrev, expansion in self.financial_abbrevs.items():
            text = re.sub(r'\b' + re.escape(abbrev) + r'\b', expansion, text, flags=re.IGNORECASE)
        
        # Clean up common formatting issues
        text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
        text = re.sub(r'\t+', ' ', text)  # Replace tabs with space
        text = re.sub(r'[^\w\s.,!?;:\-\(\)%$]', '', text)  # Keep basic punctuation
        
        # Ensure proper sentence endings
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b|\$\d+(?:\.\d+)?[kmb]?|\d+(?:\.\d+)?%?', text)
        return tokens
    
    def build_vocab(self, texts: List[str], max_vocab_size: int = 5000):
        """Build vocabulary from texts"""
        word_freq = {}
        
        # Special tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        for token in special_tokens:
            self.vocab[token] = len(self.vocab)
            self.inverse_vocab[len(self.inverse_vocab)] = token
        
        # Count word frequencies
        for text in texts:
            words = self.simple_tokenize(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words[:max_vocab_size - len(special_tokens)]:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
                self.inverse_vocab[len(self.inverse_vocab)] = word
    
    def text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to a simple embedding representation"""
        # Simple hash-based embedding
        words = self.simple_tokenize(text)
        embedding = np.zeros(self.embedding_dim)
        
        for i, word in enumerate(words):
            hash_val = hash(word) % self.embedding_dim
            embedding[hash_val] += 1.0 / (i + 1)  # Position weighting
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def embedding_to_text(self, embedding: np.ndarray) -> str:
        """Convert embedding back to text (simplified approach)"""
        # Simple heuristic based on embedding values
        avg_value = np.mean(embedding)
        std_value = np.std(embedding)
        
        # Generate text based on embedding characteristics
        if avg_value > 0.1:
            base_text = "The financial performance shows positive indicators"
        elif avg_value < -0.1:
            base_text = "The financial metrics indicate challenging conditions"
        else:
            base_text = "The financial analysis reveals mixed results"
        
        if std_value > 0.3:
            base_text += " with significant variability across different measures."
        else:
            base_text += " with consistent patterns across key indicators."
        
        return base_text
    
    def calculate_text_metrics(self, text1: str, text2: str) -> Dict[str, float]:
        """Calculate basic text comparison metrics"""
        words1 = len(text1.split())
        words2 = len(text2.split())
        
        length_ratio = words2 / max(1, words1)
        
        # Simple readability score (inverse of average word length)
        avg_word_len = np.mean([len(w) for w in text2.split()]) if text2.split() else 0
        readability = max(0, 1 - (avg_word_len - 5) / 10)
        
        # Word diversity (unique words / total words)
        words = text2.split()
        word_diversity = len(set(words)) / len(words) if words else 0
        
        return {
            'length_ratio': length_ratio,
            'readability': readability,
            'word_diversity': word_diversity
        }
    
    def extract_financial_entities(self, text: str) -> List[str]:
        """Extract financial entities and terms from text"""
        entities = []
        words = self.simple_tokenize(text.lower())
        
        # Find financial terms
        for word in words:
            if word in self.financial_terms:
                entities.append(word)
        
        # Find numbers that might be financial values
        numbers = re.findall(r'\$?[\d,]+\.?\d*[BMK]?', text)
        entities.extend(numbers)
        
        return list(set(entities))