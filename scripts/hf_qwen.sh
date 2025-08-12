export MODEL_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
time python hf_generate.py --compile --checkpoint_path $MODEL_REPO --prompt "The fifth smallest factor of 2012 is" --max_new_tokens 1000 --num_samples 1  --temperature 0.00001
