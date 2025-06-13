"""
Text Processing Utilities for Financial Text Diffusion Model.

This module handles text preprocessing, embedding generation, and conversion
between text and embedding spaces for the diffusion model.
"""

import torch
import numpy as np
import re
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string


class TextProcessor:
    """
    Text processor for financial document processing and embedding generation.
    
    Handles conversion between text and embedding spaces, text preprocessing,
    and various text manipulation tasks required for the diffusion model.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the text processor.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.embedding_model = None
        self._initialize_model()
        self._download_nltk_data()
        
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
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess financial text for better embedding quality.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
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
    
    def text_to_embedding(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Convert text to embeddings using the sentence transformer.
        
        Args:
            text: Single text string or list of strings
            
        Returns:
            Tensor of embeddings [batch_size, embedding_dim]
        """
        try:
            if isinstance(text, str):
                text = [text]
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(t) for t in text]
            
            # Filter out empty texts
            processed_texts = [t for t in processed_texts if t.strip()]
            
            if not processed_texts:
                # Return zero embedding if no valid text
                return torch.zeros(1, self.embedding_dim)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                processed_texts,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
    
    def embedding_to_text(self, embedding: torch.Tensor, 
                         num_candidates: int = 10) -> str:
        """
        Convert embedding back to text using nearest neighbor search.
        
        This is a simplified approach. In practice, you might want to use
        a more sophisticated decoder or maintain a database of embeddings.
        
        Args:
            embedding: Input embedding [embedding_dim] or [1, embedding_dim]
            num_candidates: Number of candidate texts to consider
            
        Returns:
            Generated text string
        """
        try:
            # Ensure embedding is 2D
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            
            # For now, generate a placeholder text based on embedding properties
            # In a real implementation, you would use a more sophisticated method
            embedding_np = embedding.detach().cpu().numpy().flatten()
            
            # Simple heuristic based on embedding values
            avg_value = np.mean(embedding_np)
            std_value = np.std(embedding_np)
            
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
            
        except Exception as e:
            return f"Error in text generation: {str(e)}"
    
    def similarity_search(self, query_embedding: torch.Tensor, 
                         candidate_embeddings: torch.Tensor,
                         top_k: int = 5) -> torch.Tensor:
        """
        Find most similar embeddings using cosine similarity.
        
        Args:
            query_embedding: Query embedding [embedding_dim]
            candidate_embeddings: Candidate embeddings [num_candidates, embedding_dim]
            top_k: Number of top results to return
            
        Returns:
            Indices of top-k most similar candidates
        """
        # Normalize embeddings
        query_norm = torch.nn.functional.normalize(query_embedding.unsqueeze(0), dim=1)
        candidate_norm = torch.nn.functional.normalize(candidate_embeddings, dim=1)
        
        # Compute cosine similarity
        similarities = torch.mm(query_norm, candidate_norm.t()).squeeze(0)
        
        # Get top-k indices
        top_k = min(top_k, len(similarities))
        _, top_indices = torch.topk(similarities, top_k)
        
        return top_indices
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for processing.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            # Fallback to simple splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def extract_financial_entities(self, text: str) -> List[str]:
        """
        Extract financial entities and terms from text.
        
        Args:
            text: Input text
            
        Returns:
            List of financial entities found
        """
        entities = []
        words = word_tokenize(text.lower())
        
        # Find financial terms
        for word in words:
            if word in self.financial_terms:
                entities.append(word)
        
        # Find numbers that might be financial values
        numbers = re.findall(r'\$?[\d,]+\.?\d*[BMK]?', text)
        entities.extend(numbers)
        
        return list(set(entities))
    
    def calculate_text_complexity(self, text: str) -> dict:
        """
        Calculate various text complexity metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with complexity metrics
        """
        if not text or not text.strip():
            return {'word_count': 0, 'sentence_count': 0, 'avg_word_length': 0, 'avg_sentence_length': 0}
        
        sentences = self.split_into_sentences(text)
        words = word_tokenize(text.lower())
        
        # Remove punctuation for word analysis
        words = [w for w in words if w not in string.punctuation]
        
        metrics = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0,
            'financial_entity_count': len(self.extract_financial_entities(text))
        }
        
        return metrics
    
    def batch_process_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Process multiple texts in batches for efficiency.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Tensor of all embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.text_to_embedding(batch)
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, self.embedding_dim)
    
    def get_processor_info(self) -> dict:
        """
        Get information about the text processor configuration.
        
        Returns:
            Dictionary with processor information
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'financial_terms_count': len(self.financial_terms),
            'abbreviations_count': len(self.financial_abbrevs)
        }
