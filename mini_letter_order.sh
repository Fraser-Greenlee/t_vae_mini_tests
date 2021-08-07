python train_without_tokenizer.py \
    --t5 "funnel-transformer/small" \
    --use_t5_encoder \
    --dataset_path 'tokenized_toy_problems.letter_ordering' \
    --max_train_samples 10000 \
    --max_eval_samples 100 \
    
