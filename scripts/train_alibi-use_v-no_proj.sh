#!/bin/bash

python ./examples/train_factored_alibi_example_v2.py \
  --dataset "wikimedia/wikipedia" \
  --dataset_config "20231101.en" \
  --preset medium \
  --block_size 256 \
  --batch_size 128 \
  --max_position_embeddings 512 \
  --tokenizer_type gpt2 \
  --output_dir "./outputs/alibi-use_v-use_proj" \
  --test_generation \
  --use_v \
  --num_epochs 12 \
  --max_samples 1000000
