# -*- coding: utf-8 -*-
############################################
#                                          #
#  Data Loading and Preparation Utilities  #
#                                          #
############################################

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

def load_and_prepare_data(dataset_name, dataset_config, tokenizer_name,
                          max_samples, max_seq_length, batch_size):
    """
    Loads dataset, tokenizer, preprocesses data, and returns DataLoader & tokenizer.
    """
    # --- Load Dataset ---
    print(f"\nLoading dataset: {dataset_name} ({dataset_config})...")
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=f'train[:{max_samples}]', trust_remote_code=True)
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise # Re-raise exception to stop execution

    # --- Load Tokenizer ---
    print(f"\nLoading tokenizer: {tokenizer_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            print("Tokenizer lacks default pad token, setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

    # --- Preprocess ---
    print("\nPreprocessing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer"
    )
    print("Dataset tokenized.")

    # --- Collator ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print("Data collator created (Causal LM).")

    # --- DataLoader ---
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    print("DataLoader created.")

    return dataloader, tokenizer
