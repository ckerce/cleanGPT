#! /bin/bash

python examples/train_vanilla_gpt2.py \
	--dataset "wikimedia/wikipedia" \
	--dataset_config "20231101.en" \
	--block_size 128 \
	--batch_size 128 \
	--output_dr "./outputs/vanilla" \
	--tokenizer_type gpt2 \
	--num_epochs 3 \
	--max_samples 5000000 \
	--n_layer 6 \
	--n_head 6 \
	--n_embd 768 \
	--dropout 0.1 \
	--learning_rate 0.0005 \
	--weight_decay 0.01 
