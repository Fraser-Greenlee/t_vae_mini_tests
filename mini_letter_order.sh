python train_without_tokenizer.py \
    --funnel_name "funnel-transformer/small" \
    --t5 "t5-small" \
    --dataset_path 'tokenized_toy_problems.letter_ordering' \
    --n_latent_tokens 1 \
    --latent_size 2 \
    --max_train_samples 10000 \
    --max_eval_samples 100 \

