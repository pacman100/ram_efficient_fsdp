accelerate launch --config_file config.yaml fsdp_ram_efficient.py \
--model_name_or_path "bert-base-uncased" \
--with_tracking \
--ram_efficient