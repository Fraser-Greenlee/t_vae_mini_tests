python train.py \
    --funnel_name "funnel-transformer/small" \
    --t5 "t5-small" \
    --dataset_path 'tokenized_toy_problems/letter_ordering.py' \
    --dataset_config_args '{"num_chars": 3, "seq_len": 4}' \
    --n_latent_tokens 1 \
    --latent_size 2 \
    --max_train_samples 10000 \
    --max_eval_samples 100 \
    --output_dir outputs/letter_ordering/small \
