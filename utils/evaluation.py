"""
Evaluation utilities for the Financial Text Diffusion Model.

This module provides comprehensive evaluation metrics and analysis tools
for assessing the quality and performance of the diffusion model.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
import textstat


class ModelEvaluator:
    """
    Comprehensive evaluator for the financial text diffusion model.
    
    Provides various metrics to assess text quality, semantic similarity,
    and domain-specific performance for financial text refinement.
    """
    
    def __init__(self):
        """Initialize the evaluator with required components."""
        self._setup_nltk()
        self._setup_rouge()
        
        # Financial domain-specific terms
        self.financial_indicators = {
            'positive': ['profit', 'growth', 'increase', 'gain', 'improvement', 
                        'success', 'strong', 'positive', 'rise', 'boost'],
            'negative': ['loss', 'decline', 'decrease', 'drop', 'fall', 
                        'weak', 'poor', 'negative', 'reduction', 'shortfall'],
            'neutral': ['stable', 'consistent', 'maintained', 'steady', 'flat']
        }
    
    def _setup_nltk(self):
        """Setup NLTK components."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            print(f"Warning: Could not setup NLTK: {e}")
    
    def _setup_rouge(self):
        """Setup ROUGE scorer."""
        try:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        except Exception as e:
            print(f"Warning: Could not setup ROUGE: {e}")
            self.rouge_scorer = None
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """
        Calculate BLEU score between reference and candidate texts.
        
        Args:
            reference: Reference (original) text
            candidate: Candidate (refined) text
            
        Returns:
            BLEU score (0-1)
        """
        try:
            # Tokenize texts
            reference_tokens = nltk.word_tokenize(reference.lower())
            candidate_tokens = nltk.word_tokenize(candidate.lower())
            
            # Calculate BLEU with smoothing
            smoothing = SmoothingFunction().method1
            score = sentence_bleu(
                [reference_tokens], 
                candidate_tokens,
                smoothing_function=smoothing
            )
            
            return score
            
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores between reference and candidate texts.
        
        Args:
            reference: Reference (original) text
            candidate: Candidate (refined) text
            
        Returns:
            Dictionary with ROUGE scores
        """
        if self.rouge_scorer is None:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
            
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_readability_score(self, text: str) -> float:
        """
        Calculate readability score using Flesch Reading Ease.
        
        Args:
            text: Input text
            
        Returns:
            Readability score (higher = more readable)
        """
        try:
            if not text.strip():
                return 0.0
            
            score = textstat.flesch_reading_ease(text)
            return max(0.0, score / 100.0)  # Normalize to 0-1
            
        except Exception as e:
            print(f"Error calculating readability: {e}")
            return 0.0
    
    def calculate_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Calculate various text complexity metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with complexity metrics
        """
        if not text.strip():
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'lexical_diversity': 0
            }
        
        try:
            # Basic tokenization
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text.lower())
            words = [w for w in words if w.isalpha()]  # Keep only alphabetic words
            
            # Calculate metrics
            word_count = len(words)
            sentence_count = len(sentences)
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            lexical_diversity = len(set(words)) / word_count if word_count > 0 else 0
            
            return {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'lexical_diversity': lexical_diversity
            }
            
        except Exception as e:
            print(f"Error calculating text complexity: {e}")
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'lexical_diversity': 0
            }
    
    def calculate_financial_sentiment(self, text: str) -> Dict[str, float]:
        """
        Calculate financial sentiment based on domain-specific terms.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        text_lower = text.lower()
        
        # Count sentiment indicators
        positive_count = sum(1 for term in self.financial_indicators['positive'] 
                           if term in text_lower)
        negative_count = sum(1 for term in self.financial_indicators['negative'] 
                           if term in text_lower)
        neutral_count = sum(1 for term in self.financial_indicators['neutral'] 
                          if term in text_lower)
        
        total_count = positive_count + negative_count + neutral_count
        
        if total_count == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        return {
            'positive': positive_count / total_count,
            'negative': negative_count / total_count,
            'neutral': neutral_count / total_count
        }
    
    def calculate_semantic_coherence(self, text: str) -> float:
        """
        Calculate semantic coherence score based on sentence similarity.
        
        Args:
            text: Input text
            
        Returns:
            Coherence score (0-1)
        """
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) < 2:
                return 1.0  # Single sentence is perfectly coherent
            
            # Simple coherence metric based on word overlap
            coherence_scores = []
            
            for i in range(len(sentences) - 1):
                sent1_words = set(nltk.word_tokenize(sentences[i].lower()))
                sent2_words = set(nltk.word_tokenize(sentences[i + 1].lower()))
                
                # Calculate Jaccard similarity
                intersection = len(sent1_words & sent2_words)
                union = len(sent1_words | sent2_words)
                
                if union > 0:
                    coherence_scores.append(intersection / union)
                else:
                    coherence_scores.append(0.0)
            
            return np.mean(coherence_scores) if coherence_scores else 0.0
            
        except Exception as e:
            print(f"Error calculating semantic coherence: {e}")
            return 0.0
    
    def evaluate_refinement(self, original: str, refined: str) -> Dict[str, float]:
        """
        Comprehensive evaluation of text refinement quality.
        
        Args:
            original: Original text
            refined: Refined text
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        # Basic similarity metrics
        bleu_score = self.calculate_bleu_score(original, refined)
        rouge_scores = self.calculate_rouge_scores(original, refined)
        
        # Text quality metrics
        original_readability = self.calculate_readability_score(original)
        refined_readability = self.calculate_readability_score(refined)
        readability_improvement = refined_readability - original_readability
        
        # Complexity analysis
        original_complexity = self.calculate_text_complexity(original)
        refined_complexity = self.calculate_text_complexity(refined)
        
        # Length and structure changes
        length_ratio = (refined_complexity['word_count'] / 
                       max(1, original_complexity['word_count']))
        
        # Semantic coherence
        original_coherence = self.calculate_semantic_coherence(original)
        refined_coherence = self.calculate_semantic_coherence(refined)
        coherence_improvement = refined_coherence - original_coherence
        
        # Financial sentiment analysis
        original_sentiment = self.calculate_financial_sentiment(original)
        refined_sentiment = self.calculate_financial_sentiment(refined)
        
        # Lexical diversity improvement
        diversity_improvement = (refined_complexity['lexical_diversity'] - 
                               original_complexity['lexical_diversity'])
        
        return {
            'bleu_score': bleu_score,
            'rouge_score': rouge_scores['rougeL'],  # Use ROUGE-L as primary
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'readability': refined_readability,
            'readability_improvement': readability_improvement,
            'length_ratio': length_ratio,
            'coherence': refined_coherence,
            'coherence_improvement': coherence_improvement,
            'word_diversity': refined_complexity['lexical_diversity'],
            'diversity_improvement': diversity_improvement,
            'sentiment_positive': refined_sentiment['positive'],
            'sentiment_negative': refined_sentiment['negative'],
            'sentiment_neutral': refined_sentiment['neutral']
        }
    
    def calculate_text_metrics(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Calculate basic text comparison metrics.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with comparison metrics
        """
        # Calculate basic metrics
        complexity1 = self.calculate_text_complexity(text1)
        complexity2 = self.calculate_text_complexity(text2)
        
        length_ratio = (complexity2['word_count'] / 
                       max(1, complexity1['word_count']))
        
        readability = self.calculate_readability_score(text2)
        word_diversity = complexity2['lexical_diversity']
        
        return {
            'length_ratio': length_ratio,
            'readability': readability,
            'word_diversity': word_diversity
        }
    
    def batch_evaluate(self, original_texts: List[str], 
                      refined_texts: List[str]) -> Dict[str, List[float]]:
        """
        Evaluate multiple text pairs in batch.
        
        Args:
            original_texts: List of original texts
            refined_texts: List of refined texts
            
        Returns:
            Dictionary with lists of evaluation metrics
        """
        if len(original_texts) != len(refined_texts):
            raise ValueError("Original and refined text lists must have same length")
        
        results = {
            'bleu_scores': [],
            'rouge_scores': [],
            'readability_scores': [],
            'coherence_scores': [],
            'length_ratios': []
        }
        
        for original, refined in zip(original_texts, refined_texts):
            evaluation = self.evaluate_refinement(original, refined)
            
            results['bleu_scores'].append(evaluation['bleu_score'])
            results['rouge_scores'].append(evaluation['rouge_score'])
            results['readability_scores'].append(evaluation['readability'])
            results['coherence_scores'].append(evaluation['coherence'])
            results['length_ratios'].append(evaluation['length_ratio'])
        
        return results
    
    def generate_report(self, original_texts: List[str], 
                       refined_texts: List[str]) -> Dict[str, float]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            original_texts: List of original texts
            refined_texts: List of refined texts
            
        Returns:
            Dictionary with summary statistics
        """
        batch_results = self.batch_evaluate(original_texts, refined_texts)
        
        # Calculate summary statistics
        report = {}
        
        for metric_name, values in batch_results.items():
            if values:  # Check if list is not empty
                report[f'avg_{metric_name}'] = np.mean(values)
                report[f'std_{metric_name}'] = np.std(values)
                report[f'min_{metric_name}'] = np.min(values)
                report[f'max_{metric_name}'] = np.max(values)
            else:
                report[f'avg_{metric_name}'] = 0.0
                report[f'std_{metric_name}'] = 0.0
                report[f'min_{metric_name}'] = 0.0
                report[f'max_{metric_name}'] = 0.0
        
        # Add overall quality score (weighted combination)
        if batch_results['bleu_scores'] and batch_results['rouge_scores']:
            quality_scores = [
                0.3 * bleu + 0.3 * rouge + 0.2 * readability + 0.2 * coherence
                for bleu, rouge, readability, coherence in zip(
                    batch_results['bleu_scores'],
                    batch_results['rouge_scores'],
                    batch_results['readability_scores'],
                    batch_results['coherence_scores']
                )
            ]
            report['overall_quality'] = np.mean(quality_scores)
        else:
            report['overall_quality'] = 0.0
        
        return report
    
    def compare_models(self, original_texts: List[str],
                      model1_outputs: List[str],
                      model2_outputs: List[str],
                      model1_name: str = "Model 1",
                      model2_name: str = "Model 2") -> Dict[str, Dict[str, float]]:
        """
        Compare two models' outputs.
        
        Args:
            original_texts: Original input texts
            model1_outputs: First model's outputs
            model2_outputs: Second model's outputs
            model1_name: Name for first model
            model2_name: Name for second model
            
        Returns:
            Dictionary comparing both models
        """
        model1_report = self.generate_report(original_texts, model1_outputs)
        model2_report = self.generate_report(original_texts, model2_outputs)
        
        return {
            model1_name: model1_report,
            model2_name: model2_report,
            'winner': {
                metric: model1_name if model1_report.get(metric, 0) > model2_report.get(metric, 0) else model2_name
                for metric in ['avg_bleu_scores', 'avg_rouge_scores', 'overall_quality']
            }
        }
