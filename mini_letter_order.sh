./venv/bin/python train.py \
    --do_train \

    --encoder_model_name_or_path "t5-small" \
    --encoder_config_overrides "tokenizer_class=ByT5Tokenizer" \

    --decoder_model_name_or_path "t5-small" \
    --decoder_config_overrides "tokenizer_class=ByT5Tokenizer" \

    --config_overrides="latent_size=2" \

    --dataset_path 'tokenized_toy_problems/letter_ordering.py' \
    --tokenizer_name 'google/byt5-base' \
    --dataset_config_args '{"num_chars": 3, "seq_len": 4}' \

    --max_train_samples 10000 \
    --max_eval_samples 100 \
    --output_dir outputs/letter_ordering/small \
