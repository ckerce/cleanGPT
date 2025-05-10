# ./utils/token_statistics.py
"""
Token Statistics Utility
Analyzes token usage in datasets to help optimize tokenization
"""

import os
import json
from collections import Counter
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..tokenizers import BaseTokenizer

logger = logging.getLogger(__name__)

class TokenUsageAnalyzer:
    """
    Analyzes token usage in a dataset using a specific tokenizer.
    
    This is useful for:
    - Understanding token distribution
    - Creating efficient reduced vocabularies
    - Comparing tokenization strategies
    """
    
    def __init__(self, tokenizer: BaseTokenizer):
        """
        Initialize the analyzer with a tokenizer.
        
        Args:
            tokenizer: The tokenizer to analyze
        """
        self.tokenizer = tokenizer
        self.token_counter = Counter()
        self.total_tokens = 0
        self.num_samples = 0
        self.token_sequences = []
        self.avg_sequence_length = 0
        
    def analyze_texts(self, texts: List[str], 
                      max_samples: Optional[int] = None,
                      store_sequences: bool = False,
                      show_progress: bool = True) -> Dict[str, Any]:
        """
        Analyze a list of text samples.
        
        Args:
            texts: List of text strings to analyze
            max_samples: Maximum number of samples to process (None = all)
            store_sequences: Whether to store token sequences (memory intensive)
            show_progress: Whether to show a progress bar
            
        Returns:
            Dictionary with token usage statistics
        """
        # Reset counters
        self.token_counter = Counter()
        self.total_tokens = 0
        self.num_samples = 0
        if store_sequences:
            self.token_sequences = []
        
        # Limit samples if specified
        if max_samples is not None:
            texts = texts[:max_samples]
        
        # Process each text
        iterator = tqdm(texts, desc="Analyzing texts") if show_progress else texts
        for text in iterator:
            # Tokenize the text
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # Update counters
            self.token_counter.update(tokens)
            self.total_tokens += len(tokens)
            self.num_samples += 1
            
            # Store sequences if requested
            if store_sequences:
                self.token_sequences.append(token_ids)
        
        # Calculate average sequence length
        self.avg_sequence_length = self.total_tokens / self.num_samples if self.num_samples > 0 else 0
        
        # Prepare results
        results = {
            'total_tokens': self.total_tokens,
            'unique_tokens': len(self.token_counter),
            'num_samples': self.num_samples,
            'avg_sequence_length': self.avg_sequence_length,
            'vocab_coverage': len(self.token_counter) / self.tokenizer.vocab_size,
            'most_common_tokens': self.token_counter.most_common(20)
        }
        
        logger.info(f"Analysis complete: {self.total_tokens} tokens from {self.num_samples} samples")
        logger.info(f"Unique tokens: {len(self.token_counter)} ({results['vocab_coverage']:.2%} of vocab)")
        
        return results
    
    def get_token_frequencies(self, as_percentage: bool = False) -> Dict[str, Union[int, float]]:
        """
        Get frequency of each token.
        
        Args:
            as_percentage: Return frequencies as percentages rather than counts
            
        Returns:
            Dictionary mapping tokens to their frequencies
        """
        if as_percentage:
            return {token: count / self.total_tokens for token, count in self.token_counter.items()}
        else:
            return dict(self.token_counter)
    
    def get_coverage_thresholds(self) -> Dict[float, int]:
        """
        Get vocabulary sizes needed for different coverage thresholds.
        
        Returns:
            Dictionary mapping coverage percentages to vocab sizes
        """
        # Sort tokens by frequency (descending)
        sorted_tokens = sorted(self.token_counter.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate cumulative frequencies
        cumulative_freq = 0
        coverage_thresholds = {}
        
        # Check coverage at different thresholds
        for i, (token, freq) in enumerate(sorted_tokens):
            cumulative_freq += freq
            coverage = cumulative_freq / self.total_tokens
            
            # Record vocab size at standard coverage thresholds
            for threshold in [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999]:
                if coverage >= threshold and threshold not in coverage_thresholds:
                    coverage_thresholds[threshold] = i + 1
        
        return coverage_thresholds
    
    def create_reduced_vocab(self, coverage: float = 0.95) -> Dict[str, int]:
        """
        Create a reduced vocabulary that covers a percentage of tokens.
        
        Args:
            coverage: Desired coverage as a percentage (0.0-1.0)
            
        Returns:
            Reduced vocabulary mapping tokens to IDs
        """
        # Sort tokens by frequency (descending)
        sorted_tokens = sorted(self.token_counter.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate how many tokens needed for coverage
        cumulative_freq = 0
        vocab_size = 0
        
        for token, freq in sorted_tokens:
            cumulative_freq += freq
            vocab_size += 1
            
            if cumulative_freq / self.total_tokens >= coverage:
                break
        
        # Create the reduced vocabulary
        reduced_vocab = {}
        
        # First add special tokens from the original tokenizer
        special_tokens = []
        for attr in ['pad_token', 'eos_token', 'bos_token', 'unk_token']:
            if hasattr(self.tokenizer, attr):
                token = getattr(self.tokenizer, attr)
                if token and token not in reduced_vocab:
                    special_tokens.append(token)
        
        # Add special tokens first
        for i, token in enumerate(special_tokens):
            reduced_vocab[token] = i
        
        # Then add the most frequent tokens
        for token, _ in sorted_tokens[:vocab_size]:
            if token not in reduced_vocab:  # Skip if it's a special token we already added
                reduced_vocab[token] = len(reduced_vocab)
        
        logger.info(f"Created reduced vocabulary with {len(reduced_vocab)} tokens ({coverage:.1%} coverage)")
        return reduced_vocab
    
    def plot_token_distribution(self, 
                                top_n: int = 100, 
                                log_scale: bool = True,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the distribution of token frequencies.
        
        Args:
            top_n: How many top tokens to include
            log_scale: Whether to use log scale for y-axis
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Get top tokens and their frequencies
        top_tokens = self.token_counter.most_common(top_n)
        tokens, counts = zip(*top_tokens)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(tokens)), counts)
        
        # Set log scale if requested
        if log_scale:
            plt.yscale('log')
        
        # Set labels and title
        plt.xlabel('Token Rank')
        plt.ylabel('Frequency (log scale)' if log_scale else 'Frequency')
        plt.title(f'Top {top_n} Token Frequencies')
        
        # Set x-axis ticks and labels
        step = max(1, top_n // 10)  # Show at most 10 ticks for readability
        plt.xticks(range(0, len(tokens), step), range(1, len(tokens)+1, step))
        
        # Add a grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the plot if requested
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            logger.info(f"Plot saved to: {save_path}")
        
        return plt.gcf()
    
    def plot_coverage_curve(self, 
                            max_tokens: Optional[int] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the token coverage curve.
        
        Args:
            max_tokens: Maximum number of tokens to include
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Sort tokens by frequency
        sorted_tokens = sorted(self.token_counter.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to max_tokens if specified
        if max_tokens:
            sorted_tokens = sorted_tokens[:max_tokens]
        
        # Calculate cumulative coverage
        cumulative_freq = np.cumsum([freq for _, freq in sorted_tokens])
        coverage = cumulative_freq / self.total_tokens
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(coverage)+1), coverage)
        
        # Add markers at common thresholds
        thresholds = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
        for threshold in thresholds:
            # Find the index where coverage exceeds the threshold
            idx = next((i for i, c in enumerate(coverage) if c >= threshold), None)
            if idx is not None:
                plt.plot(idx+1, threshold, 'ro')
                plt.annotate(f"{threshold:.0%}: {idx+1} tokens", 
                           (idx+1, threshold), 
                           textcoords="offset points",
                           xytext=(0, 10), 
                           ha='center')
        
        # Set labels and title
        plt.xlabel('Number of Tokens (ordered by frequency)')
        plt.ylabel('Cumulative Coverage')
        plt.title('Token Coverage Curve')
        
        # Add grid and limit axes
        plt.grid(linestyle='--', alpha=0.7)
        plt.xlim(left=0)
        plt.ylim(0, 1.05)
        
        # Save the plot if requested
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            logger.info(f"Coverage plot saved to: {save_path}")
        
        return plt.gcf()
    
    def save_analysis(self, directory: str):
        """
        Save the analysis results to files.
        
        Args:
            directory: Directory to save the results
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save token frequencies
        freq_file = os.path.join(directory, "token_frequencies.json")
        with open(freq_file, 'w', encoding='utf-8') as f:
            # Convert token counter to a dictionary for JSON serialization
            json.dump(dict(self.token_counter), f, ensure_ascii=False, indent=2)
        
        # Save analysis summary
        summary = {
            'total_tokens': self.total_tokens,
            'unique_tokens': len(self.token_counter),
            'num_samples': self.num_samples,
            'avg_sequence_length': self.avg_sequence_length,
            'vocab_coverage': len(self.token_counter) / self.tokenizer.vocab_size,
            'coverage_thresholds': self.get_coverage_thresholds()
        }
        
        summary_file = os.path.join(directory, "analysis_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Generate and save plots
        self.plot_token_distribution(save_path=os.path.join(directory, "token_distribution.png"))
        self.plot_coverage_curve(save_path=os.path.join(directory, "coverage_curve.png"))
        
        logger.info(f"Analysis results saved to: {directory}")


def analyze_dataset_with_tokenizer(texts: List[str], 
                                tokenizer: BaseTokenizer,
                                max_samples: Optional[int] = None,
                                output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a dataset using a specific tokenizer and return statistics.
    
    Args:
        texts: List of text samples to analyze
        tokenizer: Tokenizer to use for analysis
        max_samples: Maximum number of samples to analyze (None = all)
        output_dir: Directory to save analysis results (None = don't save)
        
    Returns:
        Dictionary with token usage statistics
    """
    # Create analyzer
    analyzer = TokenUsageAnalyzer(tokenizer)
    
    # Run analysis
    results = analyzer.analyze_texts(texts, max_samples=max_samples)
    
    # Save results if output directory provided
    if output_dir:
        analyzer.save_analysis(output_dir)
    
    return results
