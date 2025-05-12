# ./examples/token_analysis.py

"""
Example script for token usage analysis
Demonstrates how to analyze tokenization strategies
"""

import os
import argparse
import logging
from datasets import load_dataset, DownloadMode

from mytokenizers import create_tokenizer
from utils.token_statistics import TokenUsageAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze token usage in a dataset")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset name (e.g., 'roneneldan/TinyStories', 'wikitext')")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration (optional)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to analyze")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of samples to analyze")

    # Tokenizer arguments
    parser.add_argument("--tokenizer_type", type=str, default="gpt2",
                        choices=["character", "gpt2"],
                        help="Type of tokenizer to analyze")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to pretrained tokenizer (optional)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./token_analysis",
                        help="Output directory for analysis results")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and display plots")
    parser.add_argument("--create_reduced_vocab", action="store_true",
                        help="Create a reduced vocabulary from token usage")
    parser.add_argument("--coverage", type=float, default=0.95,
                        help="Target coverage for reduced vocabulary (0.0-1.0)")

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the dataset
    logger.info(f"Loading dataset: {args.dataset}" +
                (f" ({args.dataset_config})" if args.dataset_config else ""))

    try:
        # Load dataset with more flexible configuration
        if args.dataset_config:
            dataset = load_dataset(
                args.dataset,
                args.dataset_config,
                split=f"{args.split}[:{args.max_samples}]",
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
            )
        else:
            dataset = load_dataset(
                args.dataset,
                split=f"{args.split}[:{args.max_samples}]",
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
            )

        logger.info(f"Loaded {len(dataset)} text samples")

        # Extract text samples (handle different dataset structures)
        if 'text' in dataset.column_names:
            text_samples = dataset['text']
        elif 'story' in dataset.column_names:
            text_samples = dataset['story']
        else:
            raise ValueError(f"Could not find text column. Available columns: {dataset.column_names}")

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Create tokenizer
    logger.info(f"Creating {args.tokenizer_type} tokenizer")
    if args.tokenizer_path:
        tokenizer = create_tokenizer(args.tokenizer_type, from_pretrained=args.tokenizer_path)
    else:
        tokenizer = create_tokenizer(args.tokenizer_type)

    # Create analyzer
    analyzer = TokenUsageAnalyzer(tokenizer)

    # Run analysis
    logger.info("Running token usage analysis...")
    results = analyzer.analyze_texts(text_samples, show_progress=True)

    # Print key results
    logger.info("\nAnalysis Results:")
    logger.info(f"  Total tokens: {results['total_tokens']}")
    logger.info(f"  Unique tokens: {results['unique_tokens']}")
    logger.info(f"  Vocab coverage: {results['vocab_coverage']:.2%}")
    logger.info(f"  Average sequence length: {results['avg_sequence_length']:.2f}")
    logger.info("\nMost common tokens:")
    for token, count in results['most_common_tokens'][:10]:
        if isinstance(token, str):
            token_repr = repr(token)  # Use repr to show escape sequences
        else:
            token_repr = str(token)
        logger.info(f"  {token_repr}: {count}")

    # Get coverage thresholds
    coverage_thresholds = analyzer.get_coverage_thresholds()
    logger.info("\nCoverage Thresholds:")
    for coverage, vocab_size in sorted(coverage_thresholds.items()):
        logger.info(f"  {coverage:.1%} coverage requires {vocab_size} tokens")

    # Save analysis results
    logger.info("Saving analysis results...")
    analyzer.save_analysis(args.output_dir)

    # Display plots if requested
    if args.plot:
        logger.info("Generating plots...")
        import matplotlib.pyplot as plt

        analyzer.plot_token_distribution(top_n=100)
        plt.figure()
        analyzer.plot_coverage_curve()
        plt.show()

    # Create reduced vocabulary if requested
    if args.create_reduced_vocab:
        logger.info(f"Creating reduced vocabulary with {args.coverage:.1%} coverage...")
        reduced_vocab = analyzer.create_reduced_vocab(coverage=args.coverage)

        # Save reduced vocabulary
        import json
        vocab_path = os.path.join(args.output_dir, f"reduced_vocab_{args.coverage:.2f}.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(reduced_vocab, f, ensure_ascii=False, indent=2)

        logger.info(f"Reduced vocabulary with {len(reduced_vocab)} tokens saved to: {vocab_path}")

    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
