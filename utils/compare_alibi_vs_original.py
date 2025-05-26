# ./utils/compare_alibi_vs_original.py
"""
Comparison script between the original Token-Factored Transformer and the ALiBi version.
This script demonstrates the key differences in parameter count, memory usage, and 
length extrapolation capabilities.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Dict, List, Tuple

# Import model system
import sys
sys.path.append('..')

from model import get_model, MODEL_REGISTRY
from config import GPTConfig
from config_alibi import GPTConfigALiBi
from mytokenizers import create_tokenizer


class ModelComparator:
    """
    Class to compare the original and ALiBi versions of the Token-Factored Transformer.
    """
    
    def __init__(self, base_config_dict: Dict):
        """
        Initialize comparator with base configuration.
        
        Args:
            base_config_dict: Dictionary of configuration parameters
        """
        self.base_config = base_config_dict
        
        # Create configurations for both models
        self.original_config = GPTConfig(**base_config_dict)
        self.alibi_config = GPTConfigALiBi(**base_config_dict)
        
        # Update ALiBi config for fair comparison
        self.alibi_config.max_position_embeddings = base_config_dict.get('block_size', 256) * 4
        
        # Create tokenizer
        self.tokenizer = create_tokenizer('gpt2')
        self.original_config.update_from_tokenizer(self.tokenizer)
        self.alibi_config.update_from_tokenizer(self.tokenizer)
        
        # Initialize models using the model registry
        self.original_model = get_model("Factored", self.original_config)
        self.alibi_model = get_model("FactoredALiBi", self.alibi_config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.original_model.to(self.device)
        self.alibi_model.to(self.device)
    
    def compare_parameter_counts(self) -> Dict[str, int]:
        """Compare parameter counts between models."""
        original_params = self.original_model.get_num_params(non_embedding=False)
        original_params_no_emb = self.original_model.get_num_params(non_embedding=True)
        alibi_params = self.alibi_model.get_num_params(non_embedding=False)
        
        # Calculate positional embedding parameters in original model
        pos_emb_params = original_params - original_params_no_emb
        
        return {
            'original_total': original_params,
            'original_no_pos_emb': original_params_no_emb,
            'positional_embeddings': pos_emb_params,
            'alibi_total': alibi_params,
            'parameter_saving': original_params - alibi_params,
            'saving_percentage': ((original_params - alibi_params) / original_params) * 100
        }
    
    def compare_memory_usage(self, sequence_lengths: List[int]) -> Dict[str, List[float]]:
        """
        Compare memory usage for different sequence lengths.
        
        Args:
            sequence_lengths: List of sequence lengths to test
            
        Returns:
            Dictionary with memory usage data
        """
        results = {
            'sequence_lengths': sequence_lengths,
            'original_memory': [],
            'alibi_memory': [],
            'original_max_len': self.original_config.block_size,
            'alibi_max_len': self.alibi_config.max_position_embeddings
        }
        
        batch_size = 1  # Use batch size 1 for memory comparison
        
        for seq_len in sequence_lengths:
            # Test original model (only up to its block_size)
            if seq_len <= self.original_config.block_size:
                original_memory = self._measure_memory_usage(
                    self.original_model, batch_size, seq_len
                )
                results['original_memory'].append(original_memory)
            else:
                results['original_memory'].append(None)  # Cannot handle this length
            
            # Test ALiBi model (can handle longer sequences)
            if seq_len <= self.alibi_config.max_position_embeddings:
                alibi_memory = self._measure_memory_usage(
                    self.alibi_model, batch_size, seq_len
                )
                results['alibi_memory'].append(alibi_memory)
            else:
                results['alibi_memory'].append(None)  # Exceeds max length
        
        return results
    
    def _measure_memory_usage(self, model: nn.Module, batch_size: int, seq_len: int) -> float:
        """
        Measure memory usage for a given model and sequence length.
        
        Args:
            model: The model to test
            batch_size: Batch size for the test
            seq_len: Sequence length for the test
            
        Returns:
            Memory usage in MB
        """
        if not torch.cuda.is_available():
            return 0.0  # Cannot measure memory on CPU easily
        
        model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create dummy input
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=self.device)
        
        with torch.no_grad():
            try:
                outputs = model(input_ids)
                torch.cuda.synchronize()
                memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
                return memory_used
            except RuntimeError as e:
                if "out of memory" in str(e):
                    return float('inf')
                else:
                    raise e
    
    def compare_inference_speed(self, sequence_lengths: List[int], num_runs: int = 5) -> Dict[str, List[float]]:
        """
        Compare inference speed between models.
        
        Args:
            sequence_lengths: List of sequence lengths to test
            num_runs: Number of runs to average over
            
        Returns:
            Dictionary with timing data
        """
        results = {
            'sequence_lengths': sequence_lengths,
            'original_times': [],
            'alibi_times': [],
            'original_std': [],
            'alibi_std': []
        }
        
        batch_size = 1
        
        for seq_len in sequence_lengths:
            print(f"Testing inference speed for sequence length {seq_len}...")
            
            # Test original model
            if seq_len <= self.original_config.block_size:
                times = []
                for _ in range(num_runs):
                    time_taken = self._measure_inference_time(self.original_model, batch_size, seq_len)
                    if time_taken is not None:
                        times.append(time_taken)
                
                if times:
                    results['original_times'].append(np.mean(times))
                    results['original_std'].append(np.std(times))
                else:
                    results['original_times'].append(None)
                    results['original_std'].append(None)
            else:
                results['original_times'].append(None)
                results['original_std'].append(None)
            
            # Test ALiBi model
            if seq_len <= self.alibi_config.max_position_embeddings:
                times = []
                for _ in range(num_runs):
                    time_taken = self._measure_inference_time(self.alibi_model, batch_size, seq_len)
                    if time_taken is not None:
                        times.append(time_taken)
                
                if times:
                    results['alibi_times'].append(np.mean(times))
                    results['alibi_std'].append(np.std(times))
                else:
                    results['alibi_times'].append(None)
                    results['alibi_std'].append(None)
            else:
                results['alibi_times'].append(None)
                results['alibi_std'].append(None)
        
        return results
    
    def _measure_inference_time(self, model: nn.Module, batch_size: int, seq_len: int) -> float:
        """
        Measure inference time for a given model and sequence length.
        
        Args:
            model: The model to test
            batch_size: Batch size for the test
            seq_len: Sequence length for the test
            
        Returns:
            Inference time in milliseconds, or None if failed
        """
        model.eval()
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=self.device)
        
        # Warmup run
        with torch.no_grad():
            try:
                _ = model(input_ids)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except RuntimeError:
                return None
        
        # Actual timing
        start_time = time.time()
        with torch.no_grad():
            try:
                _ = model(input_ids)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                return (end_time - start_time) * 1000  # Convert to milliseconds
            except RuntimeError:
                return None
    
    def test_length_extrapolation(self, test_prompt: str = "The quick brown fox") -> Dict[str, any]:
        """
        Test length extrapolation capabilities.
        
        Args:
            test_prompt: Prompt to use for generation testing
            
        Returns:
            Dictionary with extrapolation test results
        """
        results = {
            'prompt': test_prompt,
            'original_results': {},
            'alibi_results': {}
        }
        
        # Tokenize prompt
        input_ids = torch.tensor([self.tokenizer.encode(test_prompt)], device=self.device)
        prompt_len = input_ids.size(1)
        
        # Test different generation lengths
        generation_lengths = [32, 64, 128, 256, 512]
        
        print(f"Testing length extrapolation with prompt: '{test_prompt}' ({prompt_len} tokens)")
        
        for gen_len in generation_lengths:
            total_len = prompt_len + gen_len
            
            # Test original model
            if total_len <= self.original_config.block_size:
                try:
                    start_time = time.time()
                    with torch.no_grad():
                        generated = self.original_model.generate(
                            input_ids, max_new_tokens=gen_len, temperature=0.8, top_k=40
                        )
                    generation_time = time.time() - start_time
                    
                    generated_text = self.tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
                    results['original_results'][gen_len] = {
                        'success': True,
                        'time': generation_time,
                        'text': generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
                        'total_length': total_len
                    }
                except Exception as e:
                    results['original_results'][gen_len] = {
                        'success': False,
                        'error': str(e),
                        'total_length': total_len
                    }
            else:
                results['original_results'][gen_len] = {
                    'success': False,
                    'error': f'Exceeds block_size ({self.original_config.block_size})',
                    'total_length': total_len
                }
            
            # Test ALiBi model
            if total_len <= self.alibi_config.max_position_embeddings:
                try:
                    start_time = time.time()
                    with torch.no_grad():
                        generated = self.alibi_model.generate(
                            input_ids, max_new_tokens=gen_len, temperature=0.8, top_k=40
                        )
                    generation_time = time.time() - start_time
                    
                    generated_text = self.tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
                    results['alibi_results'][gen_len] = {
                        'success': True,
                        'time': generation_time,
                        'text': generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
                        'total_length': total_len
                    }
                except Exception as e:
                    results['alibi_results'][gen_len] = {
                        'success': False,
                        'error': str(e),
                        'total_length': total_len
                    }
            else:
                results['alibi_results'][gen_len] = {
                    'success': False,
                    'error': f'Exceeds max_position_embeddings ({self.alibi_config.max_position_embeddings})',
                    'total_length': total_len
                }
        
        return results
    
    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report."""
        report = []
        report.append("=" * 80)
        report.append("TOKEN-FACTORED TRANSFORMER: ALiBi vs ORIGINAL COMPARISON REPORT")
        report.append("=" * 80)
        
        # Model configurations
        report.append("\n📐 MODEL CONFIGURATIONS:")
        report.append(f"Original Model:")
        report.append(f"  - Layers: {self.original_config.n_layer}")
        report.append(f"  - Heads: {self.original_config.n_head}")
        report.append(f"  - Embedding Dim: {self.original_config.n_embd}")
        report.append(f"  - Block Size: {self.original_config.block_size}")
        report.append(f"  - Positional Embeddings: Learned")
        
        report.append(f"\nALiBi Model:")
        report.append(f"  - Layers: {self.alibi_config.n_layer}")
        report.append(f"  - Heads: {self.alibi_config.n_head}")
        report.append(f"  - Embedding Dim: {self.alibi_config.n_embd}")
        report.append(f"  - Block Size: {self.alibi_config.block_size}")
        report.append(f"  - Max Position: {self.alibi_config.max_position_embeddings}")
        report.append(f"  - Positional Embeddings: ALiBi (no learned parameters)")
        
        # Parameter comparison
        param_stats = self.compare_parameter_counts()
        report.append("\n📊 PARAMETER COMPARISON:")
        report.append(f"Original Model Total:     {param_stats['original_total']:,} parameters")
        report.append(f"ALiBi Model Total:        {param_stats['alibi_total']:,} parameters")
        report.append(f"Positional Embeddings:    {param_stats['positional_embeddings']:,} parameters")
        report.append(f"Parameter Savings:        {param_stats['parameter_saving']:,} parameters")
        report.append(f"Savings Percentage:       {param_stats['saving_percentage']:.2f}%")
        
        return "\n".join(report)
    
    def plot_comparisons(self, save_path: str = None):
        """
        Create visualization plots comparing the models.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Token-Factored Transformer: ALiBi vs Original Comparison', fontsize=16)
        
        # Parameter comparison
        param_stats = self.compare_parameter_counts()
        
        ax1 = axes[0, 0]
        models = ['Original', 'ALiBi']
        params = [param_stats['original_total'], param_stats['alibi_total']]
        colors = ['#ff7f0e', '#2ca02c']
        
        bars = ax1.bar(models, params, color=colors, alpha=0.7)
        ax1.set_title('Total Parameters')
        ax1.set_ylabel('Parameters')
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Add value labels on bars
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{param/1e6:.2f}M', ha='center', va='bottom')
        
        # Memory usage comparison
        sequence_lengths = [64, 128, 256, 512, 1024]
        memory_data = self.compare_memory_usage(sequence_lengths)
        
        ax2 = axes[0, 1]
        valid_lengths = []
        original_mem = []
        alibi_mem = []
        
        for i, seq_len in enumerate(memory_data['sequence_lengths']):
            if (memory_data['original_memory'][i] is not None and 
                memory_data['alibi_memory'][i] is not None):
                valid_lengths.append(seq_len)
                original_mem.append(memory_data['original_memory'][i])
                alibi_mem.append(memory_data['alibi_memory'][i])
        
        if valid_lengths and torch.cuda.is_available():
            ax2.plot(valid_lengths, original_mem, 'o-', label='Original', color='#ff7f0e')
            ax2.plot(valid_lengths, alibi_mem, 's-', label='ALiBi', color='#2ca02c')
            ax2.set_title('Memory Usage vs Sequence Length')
            ax2.set_xlabel('Sequence Length')
            ax2.set_ylabel('Memory (MB)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'CUDA not available\nfor memory measurement', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Memory Usage (CUDA Required)')
        
        # Length capability comparison
        ax3 = axes[1, 0]
        capabilities = ['Training\nLength', 'Max Inference\nLength']
        original_caps = [self.original_config.block_size, self.original_config.block_size]
        alibi_caps = [self.alibi_config.block_size, self.alibi_config.max_position_embeddings]
        
        x = np.arange(len(capabilities))
        width = 0.35
        
        ax3.bar(x - width/2, original_caps, width, label='Original', color='#ff7f0e', alpha=0.7)
        ax3.bar(x + width/2, alibi_caps, width, label='ALiBi', color='#2ca02c', alpha=0.7)
        
        ax3.set_title('Sequence Length Capabilities')
        ax3.set_xlabel('Capability Type')
        ax3.set_ylabel('Max Tokens')
        ax3.set_xticks(x)
        ax3.set_xticklabels(capabilities)
        ax3.legend()
        
        # Add value labels
        for i, (orig, alibi) in enumerate(zip(original_caps, alibi_caps)):
            ax3.text(i - width/2, orig + 10, str(orig), ha='center', va='bottom')
            ax3.text(i + width/2, alibi + 10, str(alibi), ha='center', va='bottom')
        
        # Architecture differences
        ax4 = axes[1, 1]
        features = ['Learned\nPos. Emb.', 'Length\nExtrapolation', 'Parameter\nEfficiency']
        original_scores = [1, 0, 0]  # Has pos emb, no extrapolation, less efficient
        alibi_scores = [0, 1, 1]    # No pos emb, has extrapolation, more efficient
        
        x = np.arange(len(features))
        ax4.bar(x - width/2, original_scores, width, label='Original', color='#ff7f0e', alpha=0.7)
        ax4.bar(x + width/2, alibi_scores, width, label='ALiBi', color='#2ca02c', alpha=0.7)
        
        ax4.set_title('Architecture Features')
        ax4.set_xlabel('Feature')
        ax4.set_ylabel('Has Feature (1=Yes, 0=No)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(features)
        ax4.set_ylim(0, 1.2)
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()


def run_comprehensive_comparison():
    """Run a comprehensive comparison between the models."""
    
    # Configuration for comparison
    config_dict = {
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 384,
        'block_size': 256,
        'vocab_size': 50257,  # GPT-2 vocab size
        'dropout': 0.1,
        'bias': False,
    }
    
    print("Initializing models for comparison...")
    comparator = ModelComparator(config_dict)
    
    # Generate and print report
    report = comparator.generate_comparison_report()
    print(report)
    
    # Test length extrapolation
    print("\n🚀 TESTING LENGTH EXTRAPOLATION:")
    print("-" * 50)
    
    extrapolation_results = comparator.test_length_extrapolation(
        test_prompt="In a distant galaxy far away, there lived a curious alien species"
    )
    
    for gen_len in [32, 64, 128, 256]:
        print(f"\nGeneration Length: {gen_len} tokens")
        
        # Original model results
        orig_result = extrapolation_results['original_results'][gen_len]
        print(f"Original: {'✓' if orig_result['success'] else '✗'}", end="")
        if orig_result['success']:
            print(f" ({orig_result['time']:.3f}s)")
        else:
            print(f" - {orig_result['error']}")
        
        # ALiBi model results
        alibi_result = extrapolation_results['alibi_results'][gen_len]
        print(f"ALiBi:    {'✓' if alibi_result['success'] else '✗'}", end="")
        if alibi_result['success']:
            print(f" ({alibi_result['time']:.3f}s)")
        else:
            print(f" - {alibi_result['error']}")
    
    # Create comparison plots
    print("\n📊 GENERATING COMPARISON PLOTS...")
    comparator.plot_comparisons('token_factored_comparison.png')
    
    # Summary
    param_stats = comparator.compare_parameter_counts()
    print(f"\n🎯 KEY TAKEAWAYS:")
    print(f"✓ ALiBi saves {param_stats['parameter_saving']:,} parameters ({param_stats['saving_percentage']:.1f}%)")
    print(f"✓ ALiBi can extrapolate to {comparator.alibi_config.max_position_embeddings} tokens vs {comparator.original_config.block_size} for original")
    print(f"✓ Both models use the same factored architecture (xt/xe streams)")
    print(f"✓ ALiBi provides better length generalization without additional parameters")


if __name__ == "__main__":
    run_comprehensive_comparison()
